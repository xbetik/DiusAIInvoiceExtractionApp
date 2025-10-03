#!/usr/bin/env python3
"""
ML-based Invoice Extraction (Donut, OCR-free vision model)
- Safe prompting & BOS handling to avoid torch.embedding index errors
- Automatic fallback paths if the finetuned checkpoint rejects the prompt
"""
import os, json, argparse, logging
from typing import Any, Dict, List
from PIL import Image
import torch

from transformers import DonutProcessor, VisionEncoderDecoderModel

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("invoice-ml")

def _to_images(input_path: str) -> List[Image.Image]:
    if input_path.lower().endswith(".pdf"):
        if convert_from_path is None:
            raise RuntimeError("Install pdf2image for PDFs")
        return convert_from_path(input_path, dpi=200)
    return [Image.open(input_path).convert("RGB")]

def _normalize(record: Dict[str, Any]) -> Dict[str, Any]:
    supplier = record.get("supplier", {}) or {}
    customer = record.get("customer", {}) or {}
    out = {
        "document_id": record.get("document_id"),
        "source_type": record.get("source_type"),
        "supplier": {
            "name": supplier.get("name"),
            "vat_id": supplier.get("vat_id"),
            "ico": supplier.get("ico"),
            "address": supplier.get("address"),
            "iban": supplier.get("iban"),
        },
        "customer": {
            "name": customer.get("name"),
            "vat_id": customer.get("vat_id"),
            "ico": customer.get("ico"),
            "address": customer.get("address"),
        },
        "invoice_number": record.get("invoice_number"),
        "issue_date": record.get("issue_date"),
        "due_date": record.get("due_date"),
        "delivery_date": record.get("delivery_date"),
        "currency": record.get("currency"),
        "total_amount": record.get("total_amount"),
        "vat_amount": record.get("vat_amount"),
        "subtotal_amount": record.get("subtotal_amount"),
        "payment_method": record.get("payment_method"),
        "lines": record.get("lines", []) if isinstance(record.get("lines"), list) else [],
        "notes": record.get("notes"),
        "confidence": float(record.get("confidence") or 0.0),
        "missing_fields": [],
        "status": "incomplete",
    }
    missing = []
    if not out["supplier"].get("name"): missing.append("supplier.name")
    if not out["customer"].get("name"): missing.append("customer.name")
    if not out["invoice_number"]: missing.append("invoice_number")
    if not out["issue_date"]: missing.append("issue_date")
    if out["total_amount"] in (None, "", []): missing.append("total_amount")
    out["missing_fields"] = missing
    out["status"] = "complete" if not missing else "incomplete"
    if not record.get("confidence"):
        out["confidence"] = max(0.3, 0.95 - 0.05*len(missing))
    return out

class DonutInvoiceExtractor:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.processor = DonutProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # decoder config
        self.vocab_size = getattr(getattr(self.model.config, "decoder", self.model.config), "vocab_size")
        self.decoder_start_token_id = self.model.config.decoder_start_token_id
        self.pad_token_id = self.model.config.pad_token_id
        self.eos_token_id = self.model.config.eos_token_id

        # Validate docvqa control tokens (if present)
        self.ctrl_tokens = ["<s_docvqa>", "<s_question>", "</s_question>", "<s_answer>"]
        self.ctrl_ids = []
        for tok in self.ctrl_tokens:
            tid = self.processor.tokenizer.convert_tokens_to_ids(tok)
            self.ctrl_ids.append(tid)

        log.info(f"decoder vocab_size={self.vocab_size}, bos={self.decoder_start_token_id}, "
                 f"eos={self.eos_token_id}, pad={self.pad_token_id}")

    def _prompt_docvqa(self) -> str:
        # Official DocVQA structure
        return (
            "<s_docvqa>"
            "<s_question>Extract supplier, customer, invoice_number, issue_date (YYYY-MM-DD), "
            "due_date (YYYY-MM-DD), delivery_date (YYYY-MM-DD), currency (ISO3), subtotal_amount, "
            "vat_amount, total_amount, payment_method, and lines "
            "[{description, quantity, unit, unit_price, vat_rate, line_total}] as a JSON object.</s_question>"
            "<s_answer>"
        )

    def _ids_from_text(self, text: str) -> torch.Tensor:
        ids = self.processor.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        # Prepend BOS explicitly if the model defines one
        if self.decoder_start_token_id is not None:
            bos = torch.tensor([[self.decoder_start_token_id]], dtype=ids.dtype)
            ids = torch.cat([bos, ids], dim=1)
        # Range check
        if int(ids.max()) >= self.vocab_size or int(ids.min()) < 0:
            raise RuntimeError(
                f"Out-of-range token id in prompt (min={int(ids.min())}, "
                f"max={int(ids.max())}, vocab={self.vocab_size})."
            )
        return ids.to(self.device)

    def _generate(self, pixel_values: torch.Tensor, decoder_input_ids: torch.Tensor) -> str:
        with torch.no_grad():
            out = self.model.generate(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=512,
                num_beams=2,
                early_stopping=True,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]] if self.processor.tokenizer.unk_token_id is not None else None,
            )
        return self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0]

    def _infer_once(self, image: Image.Image, mode: str) -> Dict[str, Any]:
        enc = self.processor(image, return_tensors="pt")
        pixel_values = enc["pixel_values"].to(self.device)

        if mode == "docvqa":
            # ensure this checkpoint has the docvqa tokens
            if any(tid is None or tid < 0 or tid >= self.vocab_size for tid in self.ctrl_ids):
                raise RuntimeError("This checkpoint does not define DocVQA control tokens.")
            dec_ids = self._ids_from_text(self._prompt_docvqa())
        elif mode == "bos_only":
            # Start with BOS only (no prompt)
            if self.decoder_start_token_id is not None:
                dec_ids = torch.tensor([[self.decoder_start_token_id]], device=self.device)
            else:
                # fallback: use tokenizer's bos or pad
                bos = self.processor.tokenizer.bos_token_id or self.pad_token_id or 0
                dec_ids = torch.tensor([[bos]], device=self.device)
        else:
            raise ValueError(mode)

        seq = self._generate(pixel_values, dec_ids)

        # JSON substring parse
        try:
            s, e = seq.find("{"), seq.rfind("}")
            if s != -1 and e != -1:
                return json.loads(seq[s:e+1])
        except Exception:
            pass
        return {"raw": seq}

    def extract(self, path: str) -> Dict[str, Any]:
        pages = _to_images(path)
        img = pages[0]

        # 1) Try DocVQA prompt
        try:
            data = self._infer_once(img, "docvqa")
        except Exception as e:
            log.warning(f"DocVQA prompt failed ({type(e).__name__}: {e}); retrying with BOS-only.")
            # 2) Retry BOS-only
            try:
                data = self._infer_once(img, "bos_only")
            except Exception as e2:
                log.warning(f"BOS-only failed on {self.model_id} ({type(e2).__name__}: {e2}); "
                            f"retrying with base Donut.")
                # 3) Last resort: base Donut + "<s>" start
                base_id = "naver-clova-ix/donut-base"
                base_proc = DonutProcessor.from_pretrained(base_id)
                base_model = VisionEncoderDecoderModel.from_pretrained(base_id).to(self.device).eval()
                enc = base_proc(img, return_tensors="pt")
                pixel_values = enc["pixel_values"].to(self.device)
                # Build minimal start: tokenizer BOS or "<s>"
                start = base_proc.tokenizer.bos_token or "<s>"
                dec_ids = base_proc.tokenizer(start, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
                with torch.no_grad():
                    out = base_model.generate(
                        pixel_values=pixel_values,
                        decoder_input_ids=dec_ids,
                        max_new_tokens=256,
                        num_beams=2,
                        early_stopping=True,
                    )
                seq = base_proc.batch_decode(out.sequences, skip_special_tokens=True)[0]
                try:
                    s, e = seq.find("{"), seq.rfind("}")
                    data = json.loads(seq[s:e+1]) if s != -1 and e != -1 else {"raw": seq}
                except Exception:
                    data = {"raw": seq}

        data["document_id"] = os.path.basename(path)
        data["source_type"] = "pdf" if path.lower().endswith(".pdf") else "image"
        return _normalize(data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to invoice image/PDF")
    ap.add_argument("-o", "--output", default="invoice_ml.json")
    ap.add_argument("--model", default="naver-clova-ix/donut-base-finetuned-docvqa")
    args = ap.parse_args()

    extractor = DonutInvoiceExtractor(model_id=args.model)
    rec = extractor.extract(args.input)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print(json.dumps(rec, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
