# ruff: noqa: E501
from __future__ import annotations

from html import escape
from typing import Any


def _text(value: Any, fallback: str = "Unavailable") -> str:
    return escape(str(value)) if value not in (None, "") else fallback


def _items(values: list[Any]) -> str:
    if not values:
        return '<p class="muted">No grounded items are available.</p>'
    return "<ul>" + "".join(f"<li>{_text(value)}</li>" for value in values) + "</ul>"


def render_html_report(payload: dict[str, Any]) -> str:
    prediction = payload.get("prediction", {})
    severity = payload.get("severity", {})
    evidence = payload.get("evidence", {})
    advisory = payload.get("advisory", {})
    artifacts = payload.get("artifacts", {})
    timings = payload.get("timings_ms", {})
    sources = advisory.get("sources", [])
    severity_value = (
        f"{float(severity['affected_area_percent']):.1f}%"
        if severity.get("status") == "validated"
        and severity.get("affected_area_percent") is not None
        else "Numerical severity withheld: estimate is not validated."
    )
    source_html = (
        "".join(
            "<li><strong>{}</strong> — {}{}{}{}</li>".format(
                _text(item.get("source_id")),
                _text(item.get("title")),
                f" · {_text(item.get('organization'))}" if item.get("organization") else "",
                f" · page {int(item['page'])}" if item.get("page") else "",
                (
                    f' · <a href="{escape(str(item["url"]), quote=True)}">source link</a>'
                    if item.get("url")
                    else ""
                ),
            )
            for item in sources
        )
        or '<li class="muted">No retrieved sources support this report.</li>'
    )
    timing_html = "".join(
        f"<tr><td>{_text(name)}</td><td>{float(value):.2f} ms</td></tr>"
        for name, value in timings.items()
    )
    warning_html = _items(payload.get("warnings", []))
    artifact_names = {
        "original_image": "Uploaded image",
        "preprocessed_image": "Preprocessed image",
        "leaf_mask": "Leaf mask",
        "leaf_isolated": "Isolated leaf",
        "gradcam": "Grad-CAM",
    }
    artifact_html = (
        "".join(
            f'<figure><img class="leaf" src="{escape(str(artifacts[key]), quote=True)}" '
            f'alt="{label}"><figcaption>{label}</figcaption></figure>'
            for key, label in artifact_names.items()
            if isinstance(artifacts.get(key), str) and artifacts[key].startswith("data:image/")
        )
        or '<p class="muted">Visual artifacts unavailable.</p>'
    )
    top_predictions = _items(
        [
            f"{item.get('label', 'Unknown')} — {float(item.get('confidence', 0)) * 100:.1f}%"
            for item in prediction.get("top_k", [])
        ]
    )
    preprocessing_mode = _text(evidence.get("preprocessing", {}).get("mode"))
    device = _text(payload.get("runtime", {}).get("device"))
    gradcam_status = _text(evidence.get("provenance", {}).get("gradcam"))
    diagnosis_id = _text(payload.get("session_id"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Crop Disease Detection and Diagnosis Using Multimodal AI — {diagnosis_id}</title>
<style>
body{{font-family:Arial,sans-serif;color:#18352b;margin:0;background:#f4f5ed}}main{{max-width:960px;margin:0 auto;padding:36px}}
header,.card{{background:white;border:1px solid #d8ddcf;border-radius:16px;padding:24px;margin-bottom:16px}}header{{background:#12372a;color:white}}
h1{{margin:0 0 8px}}h2{{font-size:19px}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}figure{{margin:0}}figcaption{{font-weight:bold;padding:6px 0}}.leaf{{width:100%;max-height:360px;object-fit:contain}}
.label{{font-size:12px;text-transform:uppercase;letter-spacing:.1em;color:#65766f}}.value{{font-size:24px;font-weight:bold;margin:8px 0}}.warning{{border-left:4px solid #b45d32}}.muted{{color:#65766f}}table{{width:100%;border-collapse:collapse}}td{{padding:7px;border-bottom:1px solid #e4e7df}}a{{color:#1d6548}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}main{{padding:14px}}}}
</style></head><body><main>
<header><h1>Crop Disease Detection and Diagnosis Using Multimodal AI</h1><p>Diagnosis report · ID {diagnosis_id} · {_text(evidence.get("timestamp"))}</p></header>
<section class="card"><h2>Input and OpenCV processing</h2><div class="gallery">{artifact_html}</div><p>Preprocessing: {preprocessing_mode} · Leaf segmentation: {_text(evidence.get("segmentation_status"))}</p></section>
<section class="grid"><article class="card"><p class="label">Computer vision analysis</p>
<p class="label">Model output</p><p class="value">{_text(prediction.get("disease"), "Abstained / unavailable")}</p>
<p>Crop: {_text(prediction.get("crop"))}</p><p>Confidence: {float(prediction.get("confidence", 0)) * 100:.1f}%</p>
<p>Raw confidence: {float(prediction.get("raw_confidence") or 0) * 100:.1f}% · Calibration: {_text(prediction.get("calibration_status"))}</p><p>Abstained: {_text(prediction.get("abstained"))}</p><p>Model: {_text(evidence.get("model_id"))} · checkpoint {_text(prediction.get("checkpoint_id"))} · mode {_text(evidence.get("model_mode"))} · device {device}</p></article><article class="card"><h2>Top predictions</h2>{top_predictions}</article></section>
<section class="card warning"><h2>Explainability</h2><p>Grad-CAM status: {gradcam_status}. Grad-CAM identifies image regions that influenced the classifier. It is not a validated disease segmentation mask and must not be interpreted as severity.</p></section>
<section class="card warning"><h2>Severity and uncertainty</h2><p><strong>{_text(severity.get("category"))}</strong></p><p>{escape(severity_value)}</p><p>{_text(severity.get("warning"), "No severity note supplied.")}</p></section>
<section class="card"><h2>Grounded decision-support summary</h2><p>{_text(advisory.get("summary"))}</p><h3>Symptoms</h3>{_items(advisory.get("symptoms", []))}<h3>Possible causes</h3>{_items(advisory.get("possible_causes", []))}<h3>Prevention</h3>{_items(advisory.get("prevention", []))}<h3>Management</h3>{_items(advisory.get("management", []))}<p><strong>When to seek expert help:</strong> {_text(advisory.get("when_to_seek_expert_help"))}</p></section>
<section class="card"><h2>Cited sources</h2><ol>{source_html}</ol></section>
<section class="grid"><article class="card"><h2>Pipeline timings</h2><table>{timing_html}</table></article><article class="card warning"><h2>Warnings and limitations</h2>{warning_html}</article></section>
<section class="card warning"><strong>Important:</strong> This academic research prototype does not replace laboratory confirmation or advice from a qualified agricultural expert. Verify treatment, dosage, legal, and local recommendations before acting.</section>
</main></body></html>"""
