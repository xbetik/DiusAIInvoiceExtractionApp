#!/usr/bin/env python3
"""
ML Invoice Extraction via Hugging Face DocVQA pipeline (Donut).
- Avoids manual .generate() token plumbing (no torch.embedding crash).
- Asks the model targeted questions and assembles a normalized JSON.
"""
import os, json, argparse, logging
from typing import Dict, Any, List, Tuple
from PIL import Image

from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("invoice-ml-pipe")

# ---------- Utilities ----------
def _to_image(path: str) -> Image.Image:
    if path.lower().endswith(".pdf"):
        # Light PDF support: require pdf2image + poppler or ask user to convert to PNG/JPG
        try:
            from pdf2image import convert_from_path
        except Exception as e:
            raise RuntimeError("Install pdf2image (and Poppler on Windows) or pass a PNG/JPG.") from e
        return convert_from_path(path, dpi=200)[0]
    return Image.open(path).convert("RGB")

def _norm_amount(x: str) -> float | None:
    if not x: return None
    s = x.replace("\u00A0", " ").strip()
    s = s.replace(" ", "")
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s.strip("EUR€$£CZKPLNCHFHUF"))
    except Exception:
        return None

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
        "currency": record.get("currency") or "EUR",
        "total_amount": _norm_amount(str(record.get("total_amount"))) if record.get("total_amount") is not None else None,
        "vat_amount": _norm_amount(str(record.get("vat_amount"))) if record.get("vat_amount") is not None else None,
        "subtotal_amount": _norm_amount(str(record.get("subtotal_amount"))) if record.get("subtotal_amount") is not None else None,
        "payment_method": record.get("payment_method"),
        "lines": record.get("lines") if isinstance(record.get("lines"), list) else [],
        "notes": record.get("notes"),
        "confidence": float(record.get("confidence") or 0.0),
        "missing_fields": [],
        "status": "incomplete"
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

# ---------- Core ----------
def ask(docvqa, image: Image.Image, question: str) -> str:
    # The DocVQA pipeline returns a dict with 'answer'
    res = docvqa(image=image, question=question)
    # Older/other versions may return list [{'score','answer'}]; handle both:
    if isinstance(res, list) and res:
        return res[0].get("answer", "")
    return res.get("answer", "")

def extract_invoice(image_path: str, model_id: str) -> Dict[str, Any]:
    image = _to_image(image_path)

    # Device: let HF pick automatically; remove device_map if you prefer CPU only
    docvqa = pipeline(
        "document-question-answering",
        model=model_id,
        tokenizer=model_id,
        # device_map="auto"   # uncomment for GPU if available
    )

    # Ask focused questions (more robust than a single “return JSON” prompt)
    supplier_name   = ask(docvqa, image, "What is the supplier name?")
    supplier_vat    = ask(docvqa, image, "What is the supplier VAT number?")
    supplier_ico    = ask(docvqa, image, "What is the supplier IČO?")
    supplier_addr   = ask(docvqa, image, "What is the supplier address?")
    supplier_iban   = ask(docvqa, image, "What is the IBAN?")
    customer_name   = ask(docvqa, image, "What is the customer name?")
    customer_vat    = ask(docvqa, image, "What is the customer VAT number?")
    customer_ico    = ask(docvqa, image, "What is the customer IČO?")
    customer_addr   = ask(docvqa, image, "What is the customer address?")
    invoice_number  = ask(docvqa, image, "What is the invoice number?")
    issue_date      = ask(docvqa, image, "What is the issue date in format YYYY-MM-DD?")
    due_date        = ask(docvqa, image, "What is the due date in format YYYY-MM-DD?")
    delivery_date   = ask(docvqa, image, "What is the delivery date in format YYYY-MM-DD?")
    currency        = ask(docvqa, image, "What is the currency (3-letter ISO)?")
    subtotal_amount = ask(docvqa, image, "What is the subtotal amount?")
    vat_amount      = ask(docvqa, image, "What is the VAT amount?")
    total_amount    = ask(docvqa, image, "What is the total amount to pay?")
    payment_method  = ask(docvqa, image, "What is the payment method?")

    record = {
        "document_id": os.path.basename(image_path),
        "source_type": "pdf" if image_path.lower().endswith(".pdf") else "image",
        "supplier": {
            "name": supplier_name,
            "vat_id": supplier_vat,
            "ico": supplier_ico,
            "address": supplier_addr,
            "iban": supplier_iban,
        },
        "customer": {
            "name": customer_name,
            "vat_id": customer_vat,
            "ico": customer_ico,
            "address": customer_addr,
        },
        "invoice_number": invoice_number,
        "issue_date": issue_date,
        "due_date": due_date,
        "delivery_date": delivery_date,
        "currency": currency,
        "total_amount": total_amount,
        "vat_amount": vat_amount,
        "subtotal_amount": subtotal_amount,
        "payment_method": payment_method,
        "lines": [],   # can be extracted in a follow-up pass if needed
        "notes": None,
        "confidence": 0.0,
    }
    return _normalize(record)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to invoice image/PDF")
    ap.add_argument("-o", "--output", default="invoice_ml.json")
    ap.add_argument("--model", default="naver-clova-ix/donut-base-finetuned-docvqa")
    args = ap.parse_args()

    rec = extract_invoice(args.input, args.model)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print(json.dumps(rec, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
