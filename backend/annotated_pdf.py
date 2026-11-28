"""Embed reviewable PDF highlights from stored extraction provenance."""

import ctypes
import io
import json
import math
from collections import defaultdict
from contextlib import closing
from typing import Any

import pypdfium2 as pdfium
from pypdfium2 import raw
from pdfium_support import pypdfium2_lock

_COLORS = {
    "approved": (46, 180, 100),
    "verified": (46, 180, 100),
    "supported": (46, 180, 100),
    "overridden": (70, 140, 240),
}
_REVIEW_COLOR = (255, 190, 55)
_DEVICE_SIZE = 100_000


def _note_value(text: str):
    encoded = (text.replace("\x00", "") + "\x00").encode("utf-16-le", errors="replace")
    return (ctypes.c_ushort * (len(encoded) // 2)).from_buffer_copy(encoded)


def _page_quad(page, box: dict) -> raw.FS_QUADPOINTSF | None:
    try:
        left, top, right, bottom, width, height = (
            float(box[key]) for key in ("l", "t", "r", "b", "w", "h")
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(n) for n in (left, top, right, bottom, width, height)):
        return None
    if width <= 0 or height <= 0:
        return None

    left, right = (round(_DEVICE_SIZE * max(0, min(1, x / width))) for x in (left, right))
    top, bottom = (round(_DEVICE_SIZE * max(0, min(1, y / height))) for y in (top, bottom))
    if left >= right or top >= bottom:
        return None

    # Match the viewer's top-left page fractions. PDFium converts them into
    # unrotated PDF page coordinates, including a nonzero page crop origin.
    points = []
    for x, y in ((left, top), (right, top), (left, bottom), (right, bottom)):
        px, py = ctypes.c_double(), ctypes.c_double()
        if not raw.FPDF_DeviceToPage(page, 0, 0, _DEVICE_SIZE, _DEVICE_SIZE, 0, x, y, px, py):
            return None
        points.extend((px.value, py.value))
    return raw.FS_QUADPOINTSF(*points)


def render_annotated_pdf(source: bytes, evidence: list[dict[str, Any]]) -> tuple[bytes, int]:
    by_page = defaultdict(list)
    for cell in evidence:
        bbox = cell.get("bbox")
        boxes = bbox if isinstance(bbox, list) else [bbox]
        grouped = defaultdict(list)
        for box in boxes:
            if isinstance(box, dict) and type(box.get("page")) is int:
                grouped[box["page"]].append(box)
        for page_number, locations in grouped.items():
            by_page[page_number].append((cell, locations))

    count = 0
    with pypdfium2_lock, pdfium.PdfDocument(source) as document:
        for number, entries in by_page.items():
            if not 1 <= number <= len(document):
                continue
            with closing(document[number - 1]) as page:
                for cell, boxes in entries:
                    quads = [quad for box in boxes if (quad := _page_quad(page, box)) is not None]
                    if not quads:
                        continue
                    xs = [getattr(quad, f"x{i}") for quad in quads for i in range(1, 5)]
                    ys = [getattr(quad, f"y{i}") for quad in quads for i in range(1, 5)]
                    rectangle = raw.FS_RECTF(min(xs), max(ys), max(xs), min(ys))
                    annotation = raw.FPDFPage_CreateAnnot(page, raw.FPDF_ANNOT_HIGHLIGHT)
                    if not annotation:
                        raise RuntimeError("Could not create PDF highlight")
                    try:
                        status = cell["critic_status"]
                        value = cell["value"]
                        if not isinstance(value, str):
                            value = json.dumps(value, ensure_ascii=False) if value is not None else "No value"
                        note = (
                            f"Criterion: {cell['criterion_id']}\nValue: {value}\n"
                            f"Status: {status.replace('_', ' ')}\nQuote: {cell['verbatim_quote']}"
                        )
                        if cell.get("critic_notes"):
                            note += f"\nNotes: {cell['critic_notes']}"
                        color = _COLORS.get(status, _REVIEW_COLOR)
                        if not (
                            raw.FPDFAnnot_SetRect(annotation, rectangle)
                            and all(raw.FPDFAnnot_AppendAttachmentPoints(annotation, quad) for quad in quads)
                            and raw.FPDFAnnot_SetColor(annotation, raw.FPDFANNOT_COLORTYPE_Color, *color, 255)
                            and raw.FPDFAnnot_SetStringValue(annotation, b"T", _note_value("SLR Audit Studio"))
                            and raw.FPDFAnnot_SetStringValue(annotation, b"Contents", _note_value(note))
                        ):
                            raise RuntimeError("Could not write PDF highlight")
                        count += 1
                    finally:
                        raw.FPDFPage_CloseAnnot(annotation)
        if not count:
            return b"", 0
        output = io.BytesIO()
        document.save(output)
        return output.getvalue(), count
