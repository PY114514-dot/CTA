"""Weekly report parsing endpoint (L4 ground-truth channel)."""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.report_parser import parse_weekly_report

router = APIRouter(tags=["周报解析"])


@router.post("/api/report/parse-weekly")
async def parse_weekly_report_endpoint(file: UploadFile = File(...)) -> dict:
    """Parse a CTA weekly report (PDF or tall image) into a structured profile.

    Accepts PDF, PNG, or JPEG.  The report is sliced, OCR'd, and parsed
    into a ProfileSnapshot containing performance metrics, exposure,
    factor contributions, sector P&L, and variety-factor matrix.
    """
    allowed_types = {
        "application/pdf",
        "image/png",
        "image/jpeg",
        "application/octet-stream",
    }
    if file.content_type not in allowed_types:
        if not (file.filename or "").lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
            raise HTTPException(status_code=415, detail="仅支持 PDF、PNG 或 JPEG 文件")

    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"文件读取失败: {exc}") from exc

    if len(content) < 100:
        raise HTTPException(status_code=422, detail="文件内容过小，请检查上传")

    try:
        snapshot = parse_weekly_report(content, filename=file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"周报解析失败: {exc}") from exc

    return snapshot.to_dict()
