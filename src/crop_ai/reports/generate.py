from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from crop_ai.reports.html_report import render_html_report
from crop_ai.reports.pdf_report import render_pdf_report

SAFE_ID = re.compile(r"^[0-9a-fA-F-]{16,64}$")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a report from an archived diagnosis.")
    parser.add_argument("--diagnosis-id", required=True)
    parser.add_argument("--format", choices=["html", "pdf", "json"], default="html")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not SAFE_ID.fullmatch(arguments.diagnosis_id):
        parser.error("Invalid diagnosis identifier.")
    source = Path("artifacts/runtime/diagnoses") / f"{arguments.diagnosis_id}.json"
    if not source.is_file():
        parser.error(f"Archived diagnosis does not exist: {arguments.diagnosis_id}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    output = arguments.output or (
        Path("artifacts/runtime/reports") / f"diagnosis-{arguments.diagnosis_id}.{arguments.format}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.format == "pdf":
        output.write_bytes(render_pdf_report(payload))
    elif arguments.format == "html":
        output.write_text(render_html_report(payload), encoding="utf-8")
    else:
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {arguments.format.upper()} report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
