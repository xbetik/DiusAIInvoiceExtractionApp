
# Invoice AI Agent

This agent extracts key invoice fields from images, PDFs, or raw text, determines completeness,
and outputs a normalized JSON ready for import to an accounting system.

## Features
- OCR via Tesseract (`pytesseract`) and `pdf2image` for PDFs
- Heuristic field extraction for EU/Slovak/Czech invoices (VAT IDs, IČO, IBAN, dates, amounts)
- Currency detection (€, EUR, CZK, PLN, HUF, CHF, USD, GBP, ...)
- Line-item parsing (simple "qty x price = total" pattern; falls back to single line)
- Completeness check: supplier, customer, invoice_number, issue_date, total_amount
- Confidence score + list of missing fields
- JSON Schema validation (optional)

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install pytesseract pillow pdf2image python-dateutil rapidfuzz jsonschema
# Install Tesseract on your system + language packs (eng, slk, ces, deu recommended)
```

## Usage
```bash
python invoice_agent.py input.jpg -o out.json --schema invoice_schema.json
# or read text from stdin:
cat invoice.txt | python invoice_agent.py - -o out.json
```

## Output (example)
```json
{
  "document_id": "input.jpg",
  "source_type": "image",
  "supplier": {"name":"Acme s.r.o.", "vat_id":"SK1234567890", "ico":"12345678", "address": null, "iban":"SK6802000000001234567891", "bank_account": null, "email": null, "phone": null},
  "customer": {"name":"Privatbanka, a.s.", "vat_id":"SK12...", "ico": null, "address": null},
  "invoice_number": "2025-001",
  "issue_date": "2025-09-08",
  "due_date": "2025-09-22",
  "delivery_date": null,
  "currency": "EUR",
  "total_amount": 1200.5,
  "vat_amount": 200.08,
  "subtotal_amount": 1000.42,
  "payment_method": null,
  "lines": [
    {"description":"Consulting services", "quantity":1.0, "unit":"", "unit_price":1200.5, "vat_rate":20.0, "line_total":1200.5}
  ],
  "notes": null,
  "confidence": 0.82,
  "missing_fields": [],
  "status": "complete"
}
```

## Extending
- Plug in a layout-aware OCR (e.g., paddleocr) or a transformer model for field tagging.
- Add vendor-specific templates by mapping known VAT IDs to field pickers.
- Enrich line parsing by detecting table structures from coordinates (requires OCR bounding boxes).

## Licensing
MIT
