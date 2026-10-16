# Diagnosis reports

Every completed diagnosis is archived under ignored `artifacts/runtime/diagnoses/`. Report creation is
separate from inference, so an optional PDF failure cannot break diagnosis.

- `GET /api/v1/reports/diagnosis/{id}.html` — mandatory, portable report.
- `GET /api/v1/reports/diagnosis/{id}.json` — complete typed evidence export.
- `GET /api/v1/reports/diagnosis/{id}.pdf` — optional ReportLab document.

Install PDF support with `python -m pip install -e '.[reports]'`. If absent or disabled using
`REPORT_ENABLE_PDF=false`, the API returns 503 and directs the user to HTML. Reports include visual
artifacts, model/checkpoint/mode, confidence and calibration status, processing, Grad-CAM disclaimer,
severity status, grounded narrative, sources, timings, warnings, and a decision-support disclaimer.
A numerical severity value is withheld unless the evidence status is `validated`.

An archived report can also be generated without starting the API:

```bash
python -m crop_ai.reports.generate --diagnosis-id <id> --format html
python -m crop_ai.reports.generate --diagnosis-id <id> --format pdf
```
