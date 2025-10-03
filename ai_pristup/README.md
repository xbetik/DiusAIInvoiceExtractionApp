
# ML-based Invoice Extraction (Donut)

This is an **AI, layout-aware** alternative to rule-based parsing.
It uses a Hugging Face **Donut** model to read the image and *generate your JSON directly*.

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
> First run will download model weights (hundreds of MB).

## Run
```powershell
python ml_agent.py .\sample_invoice.jpg -o invoice_ml.json --model naver-clova-ix/donut-base-finetuned-docvqa
```

## Notes
- Works best if later fine-tuned on your invoices.
- Normalizes the model output to `invoice_schema.json`, sets `status`/`missing_fields`, and a basic `confidence`.
