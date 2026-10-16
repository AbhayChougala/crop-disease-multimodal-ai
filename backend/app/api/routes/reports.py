from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse

from backend.app.core.config import get_settings
from backend.app.services.diagnosis_archive import load_diagnosis
from crop_ai.reports.html_report import render_html_report
from crop_ai.reports.pdf_report import PDFReportUnavailable, render_pdf_report

router = APIRouter(tags=["reports"])


def _load(diagnosis_id: str) -> dict:
    try:
        payload = load_diagnosis(diagnosis_id)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=404, detail="Diagnosis report was not found.") from exc
    if payload is None:
        raise HTTPException(status_code=404, detail="Diagnosis report was not found.")
    return payload


@router.get("/reports/diagnosis/{diagnosis_id}.html", response_class=HTMLResponse)
async def html_report(diagnosis_id: str) -> HTMLResponse:
    return HTMLResponse(
        render_html_report(_load(diagnosis_id)),
        headers={"Content-Disposition": f'inline; filename="diagnosis-{diagnosis_id}.html"'},
    )


@router.get("/reports/diagnosis/{diagnosis_id}.json")
async def json_report(diagnosis_id: str) -> JSONResponse:
    return JSONResponse(
        _load(diagnosis_id),
        headers={"Content-Disposition": f'attachment; filename="diagnosis-{diagnosis_id}.json"'},
    )


@router.get("/reports/diagnosis/{diagnosis_id}/json", include_in_schema=False)
async def json_report_alias(diagnosis_id: str) -> JSONResponse:
    return await json_report(diagnosis_id)


@router.get("/reports/diagnosis/{diagnosis_id}.pdf")
async def pdf_report(diagnosis_id: str) -> Response:
    if not get_settings().report_enable_pdf:
        raise HTTPException(
            status_code=503, detail="PDF reports are disabled; use the HTML report."
        )
    try:
        content = render_pdf_report(_load(diagnosis_id))
    except PDFReportUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"{exc} Use the HTML report instead.") from exc
    return Response(
        content,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="diagnosis-{diagnosis_id}.pdf"'},
    )
