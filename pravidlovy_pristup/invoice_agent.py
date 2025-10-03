#!/usr/bin/env python3
"""
Invoice AI Agent: OCR + parsing + validation -> JSON ready for accounting import.

Dependencies (install via pip):
  pip install pytesseract pillow pdf2image python-dateutil rapidfuzz jsonschema
Optional (for better OCR): tesseract-ocr system package & appropriate language packs (e.g., eng, slk, ces, deu).
"""
from __future__ import annotations
import re, os, io, json, math, uuid, logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import date
from dateutil import parser as dtparser
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

try:
    import pytesseract
    from PIL import Image, ImageOps
except Exception:
    pytesseract = None
    Image = None

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

try:
    from rapidfuzz import fuzz, process
except Exception:
    fuzz = None
    process = None

try:
    from jsonschema import validate as jsonschema_validate
except Exception:
    jsonschema_validate = None

# ------------------------ Utilities ------------------------
CURRENCY_SYMBOLS = {
    "€": "EUR", "$": "USD", "£": "GBP", "Kč": "CZK", "zł": "PLN", "Ft": "HUF", "CHF": "CHF"
}
ISO_CURRENCIES = {"EUR","USD","GBP","CZK","PLN","HUF","CHF","AUD","CAD","JPY","NOK","SEK","DKK","RON","BGN"}

VAT_PAT = re.compile(r'\b(?:SK|CZ|HU|PL|DE|AT|GB|FR|NL|IT|ES|PT|RO|BG|BE|DK|SE|NO|IE|LU|LT|LV|EE)[A-Z0-9]{2,12}\b', re.I)
ICO_PAT = re.compile(r'\bIČO[:\s]*([0-9]{6,10})\b', re.I)
IBAN_PAT = re.compile(r'\b[A-Z]{2}[0-9]{2}[A-Z0-9]{1,30}\b')
INVOICE_NO_PAT = re.compile(r'\b(?:Invoice|Faktúra|Faktura|Rechnung|Účtenka)\s*[:#]?\s*([A-Z0-9\-\/]{3,})', re.I)
DATE_PAT = re.compile(r'\b(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})\b')
ISO_DATE_PAT = re.compile(r'\b(20\d{2}-\d{2}-\d{2})\b')
AMOUNT_PAT = re.compile(r'(?<![A-Za-z])([0-9]{1,3}(?:[ .\u00A0,][0-9]{3})*|[0-9]+)([.,][0-9]{2})?\s*(€|EUR|\$|USD|GBP|£|CZK|Kč|PLN|zł|HUF|Ft|CHF)\b')
TOTAL_KEYWORDS = re.compile(r'\b(Total|Celkom|Suma|K zaplateniu|Grand\s*Total|Amount\s*Due)\b', re.I)
SUBTOTAL_KEYWORDS = re.compile(r'\b(Subtotal|Medzisúčet|Zwischensumme)\b', re.I)
VAT_KEYWORDS = re.compile(r'\b(VAT|DPH|MwSt)\b', re.I)
DUE_DATE_KEYS = re.compile(r'\b(Due\s*Date|Splatnosť|Splatnost|Fälligkeitsdatum)\b', re.I)
ISSUE_DATE_KEYS = re.compile(r'\b(Issue\s*Date|Dátum\s*vystavenia|Vystavená|Ausstellungsdatum)\b', re.I)
DELIVERY_DATE_KEYS = re.compile(r'\b(Dodanie|Delivery\s*Date|Leistungsdatum)\b', re.I)
SUPPLIER_KEYS = re.compile(r'\b(Dodávateľ|Supplier|Verkäufer|Fakturant)\b', re.I)
CUSTOMER_KEYS = re.compile(r'\b(Odb(ě|e)rateľ|Customer|Käufer|ODBERATEL|Odberateľ|Bill\s*to)\b', re.I)

def norm_decimal(s: str) -> Optional[float]:
    """Normalize European/US formatted amounts to float."""
    if s is None:
        return None
    s = s.strip()
    # remove spaces non-breaking
    s = s.replace('\u00A0',' ').replace(' ','')
    # If both comma and dot present, assume dot thousands, comma decimal
    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.','')
            s = s.replace(',','.')
        else:
            s = s.replace(',','')
    else:
        # Single separator: if comma, make decimal point
        if ',' in s and '.' not in s:
            s = s.replace(',','.')
        # if dot present, leave
    try:
        return float(s)
    except Exception:
        return None

def try_parse_date(s: str) -> Optional[str]:
    try:
        d = dtparser.parse(s, dayfirst=True, fuzzy=True).date()
        return d.isoformat()
    except Exception:
        return None

def guess_currency(text: str) -> Optional[str]:
    for sym, iso in CURRENCY_SYMBOLS.items():
        if sym in text:
            return iso
    for iso in ISO_CURRENCIES:
        if re.search(rf'\b{iso}\b', text):
            return iso
    return None

def extract_block(text: str, start_regex: re.Pattern, window_lines: int = 6) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if start_regex.search(line):
            return '\n'.join(lines[i:i+window_lines])
    return ""

def score_confidence(fields: Dict[str, object], missing: List[str]) -> float:
    filled = len([k for k,v in fields.items() if v not in (None, "", [], {})])
    total = filled + len(missing)
    if total == 0:
        return 0.0
    base = filled / total
    # Boost if key identifiers present
    boost = 0.0
    if fields.get("supplier",{}).get("vat_id"): boost += 0.05
    if fields.get("customer",{}).get("vat_id"): boost += 0.05
    if fields.get("iban"): boost += 0.05
    if fields.get("invoice_number"): boost += 0.05
    return max(0.0, min(1.0, base + boost))

# ------------------------ OCR Layer ------------------------
def ocr_from_file(path: str, lang: str = "eng+slk+ces+deu") -> str:
    """Return text from image/pdf; if pytesseract or pdf2image missing, raise helpful error."""
    if path.lower().endswith(".pdf"):
        if convert_from_path is None:
            raise RuntimeError("pdf2image not installed. Install it or supply text instead of PDF.")
        pages = convert_from_path(path, dpi=300)
        text_pages = [pytesseract.image_to_string(p, lang=lang) for p in pages]
        return "\n\n".join(text_pages)
    else:
        if Image is None or pytesseract is None:
            raise RuntimeError("pytesseract/Pillow not installed. Install them or supply text instead of image.")
        img = Image.open(path)
        # preprocess: grayscale & increase contrast slightly
        img = ImageOps.grayscale(img)
        text = pytesseract.image_to_string(img, lang=lang)
        return text

# ------------------------ Extraction ------------------------
def extract_entities(text: str) -> Dict[str, object]:
    # Supplier & customer blocks (heuristic by keywords)
    supplier_block = extract_block(text, SUPPLIER_KEYS) or text[:500]
    customer_block = extract_block(text, CUSTOMER_KEYS)
    # Names: first non-empty line in block
    def first_name(block: str) -> Optional[str]:
        for line in block.splitlines():
            t = line.strip()
            if len(t) > 2 and not any(k in t.lower() for k in ["supplier","dodávateľ","verkäufer","fakturant","customer","odberateľ","bill to"]):
                return t
        return None

    supplier = {
        "name": first_name(supplier_block),
        "vat_id": (VAT_PAT.search(supplier_block or text) or VAT_PAT.search(text) or [None,None])[0],
        "ico": (ICO_PAT.search(supplier_block or text).group(1) if ICO_PAT.search(supplier_block or text) else None),
        "address": None,
        "iban": (IBAN_PAT.search(text).group(0) if IBAN_PAT.search(text) else None),
        "bank_account": None,
        "email": None,
        "phone": None,
    }

    customer = {
        "name": first_name(customer_block) if customer_block else None,
        "vat_id": (VAT_PAT.search(customer_block) or [None,None])[0] if customer_block else None,
        "ico": (ICO_PAT.search(customer_block).group(1) if customer_block and ICO_PAT.search(customer_block) else None),
        "address": None
    }

    # Invoice number
    inv_no_match = INVOICE_NO_PAT.search(text)
    invoice_number = inv_no_match.group(1).strip() if inv_no_match else None

    # Dates
    def find_date_with_key(key_re: re.Pattern) -> Optional[str]:
        block = extract_block(text, key_re, window_lines=3)
        for m in ISO_DATE_PAT.finditer(block):
            d = try_parse_date(m.group(1))
            if d: return d
        for m in DATE_PAT.finditer(block):
            d = try_parse_date(m.group(1))
            if d: return d
        return None

    issue_date = find_date_with_key(ISSUE_DATE_KEYS) or (try_parse_date((ISO_DATE_PAT.search(text) or DATE_PAT.search(text) or [None,None])[0]) if (ISO_DATE_PAT.search(text) or DATE_PAT.search(text)) else None)
    due_date = find_date_with_key(DUE_DATE_KEYS)
    delivery_date = find_date_with_key(DELIVERY_DATE_KEYS)

    # Amounts & currency
    currency = guess_currency(text) or "EUR"
    total_amount = None; subtotal_amount = None; vat_amount = None

    # look around keywords
    def amount_near(key_re: re.Pattern) -> Optional[float]:
        block = extract_block(text, key_re, window_lines=3)
        for m in AMOUNT_PAT.finditer(block):
            val = norm_decimal(m.group(1) + (m.group(2) or ""))
            if val is not None:
                return val
        # fallback: first numeric in block
        nums = re.findall(r'([0-9][0-9 .,\u00A0]+)', block)
        for n in nums[::-1]:
            v = norm_decimal(n)
            if v is not None:
                return v
        return None

    total_amount = amount_near(TOTAL_KEYWORDS)
    subtotal_amount = amount_near(SUBTOTAL_KEYWORDS)
    vat_amount = amount_near(VAT_KEYWORDS)

    # Lines (very heuristic): look for rows like "Desc .... qty x price = total"
    lines = []
    for raw_line in text.splitlines():
        if len(raw_line.strip()) < 4: 
            continue
        # Capture something like: "Item A 2 x 15,00 = 30,00"
        m = re.search(r'(.+?)\s+([0-9]+(?:[.,][0-9]+)?)\s*[x×]\s*([0-9]+(?:[.,][0-9]+)?)\s*[=≈]\s*([0-9]+(?:[.,][0-9]+)?)', raw_line)
        if m:
            desc = m.group(1).strip()
            qty = norm_decimal(m.group(2))
            price = norm_decimal(m.group(3))
            total = norm_decimal(m.group(4))
            lines.append({
                "description": desc,
                "quantity": qty or 1.0,
                "unit": "",
                "unit_price": price or (total / qty if qty else None),
                "vat_rate": None,
                "line_total": total or (qty * price if qty and price else None)
            })

    # Fallback: if no lines detected, create one from total
    if not lines and total_amount:
        lines = [{
            "description": "Goods/Services",
            "quantity": 1.0,
            "unit": "",
            "unit_price": total_amount,
            "vat_rate": None,
            "line_total": total_amount
        }]

    # Compute missing fields
    core_missing = []
    if not supplier.get("name"): core_missing.append("supplier.name")
    if not customer.get("name"): core_missing.append("customer.name")
    if not invoice_number: core_missing.append("invoice_number")
    if not issue_date: core_missing.append("issue_date")
    if not total_amount: core_missing.append("total_amount")

    status = "complete" if not core_missing else "incomplete"

    # Build record
    record = {
        "document_id": None,
        "source_type": None,
        "supplier": supplier,
        "customer": customer,
        "invoice_number": invoice_number,
        "issue_date": issue_date,
        "due_date": due_date,
        "delivery_date": delivery_date,
        "currency": currency,
        "total_amount": total_amount,
        "vat_amount": vat_amount,
        "subtotal_amount": subtotal_amount,
        "payment_method": None,
        "lines": lines,
        "notes": None,
        "confidence": 0.0,
        "missing_fields": core_missing,
        "status": status
    }
    record["confidence"] = score_confidence(record, core_missing)
    return record

# ------------------------ Public API ------------------------
def process_document(input_path: Optional[str] = None, *, raw_text: Optional[str] = None, document_id: Optional[str] = None, source_type: Optional[str] = None, lang: str = "eng+slk+ces+deu") -> Dict[str, object]:
    """
    Process a single invoice (image/pdf/text) and return a normalized dict.
    """
    if not document_id:
        document_id = os.path.basename(input_path) if input_path else str(uuid.uuid4())
    if input_path and not source_type:
        source_type = "pdf" if input_path.lower().endswith(".pdf") else "image"
    if raw_text and not source_type:
        source_type = "text"

    if raw_text is None:
        if input_path is None:
            raise ValueError("Provide either input_path or raw_text.")
        text = ocr_from_file(input_path, lang=lang)
    else:
        text = raw_text

    record = extract_entities(text)
    record["document_id"] = document_id
    record["source_type"] = source_type

    return record

def save_json(record: Dict[str, object], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

def load_schema(schema_path: str) -> Dict[str, object]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_record(record: Dict[str, object], schema_path: str) -> Tuple[bool, Optional[str]]:
    if jsonschema_validate is None:
        return True, None
    schema = load_schema(schema_path)
    try:
        jsonschema_validate(instance=record, schema=schema)
        return True, None
    except Exception as e:
        return False, str(e)

# ------------------------ CLI ------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Extract invoice data to JSON")
    ap.add_argument("input", help="Path to invoice image/PDF, or '-' to read text from stdin")
    ap.add_argument("-o", "--output", default="invoice.json", help="Output JSON path")
    ap.add_argument("--schema", default=None, help="Path to JSON schema to validate against")
    ap.add_argument("--lang", default="eng+slk+ces+deu", help="Tesseract language packs to use")
    args = ap.parse_args()

    if args.input == "-":
        raw_text = sys.stdin.read()
        record = process_document(raw_text=raw_text, source_type="text")
    else:
        record = process_document(input_path=args.input, lang=args.lang)

    if args.schema:
        ok, err = validate_record(record, args.schema)
        if not ok:
            logging.warning("Schema validation warning: %s", err)

    save_json(record, args.output)
    print(json.dumps(record, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    import sys
    main()
