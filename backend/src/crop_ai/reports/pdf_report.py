from __future__ import annotations

import base64
import io
from typing import Any
from xml.sax.saxutils import escape


class PDFReportUnavailable(RuntimeError):
    pass


def _safe(value: Any) -> str:
    return escape(str(value))


def _decode_data_url(value: str) -> io.BytesIO | None:
    if not value.startswith("data:image/") or ";base64," not in value:
        return None
    try:
        return io.BytesIO(base64.b64decode(value.split(",", 1)[1], validate=True))
    except (ValueError, base64.binascii.Error):
        return None


def render_pdf_report(payload: dict[str, Any]) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Image,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise PDFReportUnavailable("Install the 'reports' extra to enable PDF reports.") from exc

    output = io.BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            textColor=colors.HexColor("#12372a"),
            alignment=TA_CENTER,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Warning",
            parent=styles["BodyText"],
            backColor=colors.HexColor("#f2e6db"),
            borderColor=colors.HexColor("#b45d32"),
            borderWidth=1,
            borderPadding=8,
            leading=15,
        )
    )
    story: list[Any] = [
        Paragraph(
            "Crop Disease Detection and Diagnosis Using Multimodal AI",
            styles["ReportTitle"],
        ),
        Paragraph(
            "Research prototype · ID "
            f"{_safe(payload.get('session_id', 'unavailable'))} · "
            f"{_safe(payload.get('evidence', {}).get('timestamp', 'time unavailable'))}",
            styles["BodyText"],
        ),
        Spacer(1, 8),
    ]
    image_data = _decode_data_url(str(payload.get("artifacts", {}).get("original_image", "")))
    if image_data:
        story.extend(
            [Image(image_data, width=90 * mm, height=62 * mm, kind="proportional"), Spacer(1, 8)]
        )
    prediction = payload.get("prediction", {})
    evidence = payload.get("evidence", {})
    rows = [
        ["Predicted crop", prediction.get("crop") or "Unavailable"],
        ["Predicted condition", prediction.get("disease") or "Abstained / unavailable"],
        ["Confidence", f"{float(prediction.get('confidence', 0)) * 100:.1f}%"],
        ["Raw confidence", f"{float(prediction.get('raw_confidence') or 0) * 100:.1f}%"],
        ["Calibration", prediction.get("calibration_status") or "Unavailable"],
        ["Abstained", str(prediction.get("abstained", False))],
        ["Model", evidence.get("model_id") or "Unavailable"],
        ["Checkpoint", prediction.get("checkpoint_id") or "Unavailable"],
        ["Runtime mode", evidence.get("model_mode") or "Unavailable"],
        ["Runtime device", payload.get("runtime", {}).get("device") or "Unavailable"],
    ]
    table = Table(rows, colWidths=[48 * mm, 110 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8efd9")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d8ddcf")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([table, Spacer(1, 12)])
    top_predictions = prediction.get("top_k", [])
    story.extend(
        [
            Paragraph("Top predictions", styles["Heading2"]),
            Paragraph(
                "<br/>".join(
                    f"• {_safe(item.get('label', 'Unknown'))} — "
                    f"{float(item.get('confidence', 0)) * 100:.1f}%"
                    for item in top_predictions
                )
                or "No ranked predictions are available.",
                styles["BodyText"],
            ),
            Spacer(1, 8),
            Paragraph("OpenCV processing and explainability", styles["Heading2"]),
            Paragraph(
                "Preprocessing: "
                f"{_safe(evidence.get('preprocessing', {}).get('mode', 'Unavailable'))}. "
                "Leaf segmentation: "
                f"{_safe(evidence.get('segmentation_status', 'Unavailable'))}. "
                "Grad-CAM status: "
                f"{_safe(evidence.get('provenance', {}).get('gradcam', 'Unavailable'))}. "
                "Grad-CAM identifies regions that influenced the classifier; it is not a "
                "validated disease segmentation mask or a severity measurement.",
                styles["BodyText"],
            ),
            Spacer(1, 8),
        ]
    )
    severity = payload.get("severity", {})
    severity_text = "Numerical severity withheld because the estimate is not validated."
    if severity.get("status") == "validated" and severity.get("affected_area_percent") is not None:
        severity_text = f"Validated affected area: {float(severity['affected_area_percent']):.1f}%"
    story.extend(
        [
            Paragraph("Severity and uncertainty", styles["Heading2"]),
            Paragraph(
                escape(
                    f"{severity.get('category', 'Unavailable')}. {severity_text} "
                    f"{severity.get('warning') or ''}"
                ),
                styles["Warning"],
            ),
            Spacer(1, 8),
        ]
    )
    advisory = payload.get("advisory", {})
    story.extend(
        [
            PageBreak(),
            Paragraph("Grounded decision-support summary", styles["Heading2"]),
            Paragraph(
                _safe(advisory.get("summary") or "No grounded summary is available."),
                styles["BodyText"],
            ),
            Spacer(1, 6),
        ]
    )
    for heading, key in (
        ("Symptoms", "symptoms"),
        ("Possible causes", "possible_causes"),
        ("Prevention", "prevention"),
        ("Management", "management"),
    ):
        story.append(Paragraph(heading, styles["Heading3"]))
        values = advisory.get(key, [])
        story.append(
            Paragraph(
                "<br/>".join(f"• {_safe(value)}" for value in values)
                if values
                else "No grounded items are available.",
                styles["BodyText"],
            )
        )
    story.extend(
        [
            Spacer(1, 8),
            Paragraph(
                "<b>When to seek expert help:</b> "
                + escape(
                    str(
                        advisory.get("when_to_seek_expert_help")
                        or "Consult a qualified agricultural expert before treatment decisions."
                    )
                ),
                styles["BodyText"],
            ),
            PageBreak(),
            Paragraph("Sources, provenance, and limitations", styles["Heading1"]),
        ]
    )
    sources = advisory.get("sources", [])
    if sources:
        for item in sources:
            details = " · ".join(
                str(value)
                for value in (
                    item.get("source_id"),
                    item.get("title"),
                    item.get("organization"),
                    f"page {item['page']}" if item.get("page") else None,
                    item.get("url"),
                )
                if value
            )
            story.append(Paragraph(f"• {_safe(details)}", styles["BodyText"]))
    else:
        story.append(Paragraph("No retrieved sources support this report.", styles["Warning"]))
    story.extend([Spacer(1, 10), Paragraph("Warnings", styles["Heading2"])])
    for warning in payload.get("warnings", []):
        story.append(Paragraph(f"• {_safe(warning)}", styles["BodyText"]))
    story.extend(
        [
            Spacer(1, 12),
            Paragraph(
                "This academic research prototype does not replace laboratory confirmation or "
                "advice from a qualified agricultural expert. Verify treatment, dosage, legal, "
                "and local recommendations before acting.",
                styles["Warning"],
            ),
        ]
    )
    document.build(story)
    return output.getvalue()
