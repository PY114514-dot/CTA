"""Weekly report parser — public interface.

Pipeline: PDF/image → slice → OCR → structured extraction → ProfileSnapshot.
"""

from app.services.report_parser.pdf_slicer import slice_report, SliceInfo
from app.services.report_parser.structure_extractor import extract_profile
from app.services.report_parser.schema import ProfileSnapshot

__all__ = ["parse_weekly_report", "slice_report", "extract_profile", "ProfileSnapshot", "SliceInfo"]


def parse_weekly_report(
    file_bytes: bytes,
    filename: str = "",
    slice_height: int | None = None,
) -> ProfileSnapshot:
    """End-to-end: parse a weekly report file into a structured snapshot.

    Parameters
    ----------
    file_bytes : raw file content (PDF, PNG, or JPEG)
    filename : original filename for metadata
    slice_height : override default slice height

    Returns
    -------
    ProfileSnapshot with extracted fields.
    """
    slices = slice_report(file_bytes, slice_height=slice_height)
    snapshot = extract_profile(slices, source_file=filename)
    return snapshot
