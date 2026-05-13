"""Embedded figure extraction (Tier A) and optional calibration-based digitization (Tier B)."""

from __future__ import annotations

import hashlib
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

from fasting_atlas.schemas import FigureEvidence, FigureExtracted

_FIG_LINE = re.compile(r"^\s*(Figure|Fig\.?)\s*\d+[a-z]?\b.*$", re.IGNORECASE | re.MULTILINE)


def extract_figures(
    pdf_path: str,
    paper_id: str,
    artifacts_root: str | Path,
    *,
    tier: str = "a",
    calibration_dir: str | Path | None = None,
    digitize_flag: bool = False,
) -> list[FigureExtracted]:
    """
    tier: ``a`` — inventory + PNG assets; ``b`` — also load calibration JSON and fill ``digitized_series``.
    """
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("Figure extraction requires pymupdf: pip install pymupdf") from exc

    artifacts_root = Path(artifacts_root)
    out_dir = artifacts_root / paper_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_root = Path(calibration_dir) if calibration_dir else None

    doc = fitz.open(pdf_path)
    figures: list[FigureExtracted] = []
    seen_xref: set[int] = set()
    fig_counter = 0

    for page in doc:
        page_num = page.number + 1
        page_text = page.get_text("text") or ""
        caption = _find_figure_caption(page_text)

        for img in page.get_images(full=True):
            xref = int(img[0])
            if xref in seen_xref:
                continue
            seen_xref.add(xref)

            rects = page.get_image_rects(xref)
            bbox: list[float] = []
            if rects:
                r = rects[0]
                bbox = [float(r.x0), float(r.y0), float(r.x1), float(r.y1)]

            try:
                img_dict = doc.extract_image(xref)
            except (ValueError, RuntimeError):
                continue
            image_bytes = img_dict.get("image") or b""
            if not image_bytes:
                continue

            try:
                from PIL import Image

                im = Image.open(BytesIO(image_bytes))
                buf = BytesIO()
                im.save(buf, format="PNG")
                image_bytes = buf.getvalue()
            except Exception:
                pass

            sha = hashlib.sha256(image_bytes).hexdigest()
            fig_counter += 1
            figure_id = f"fig_p{page_num}_i{fig_counter}"
            fname = f"{figure_id}_xref{xref}.png"
            img_path = out_dir / fname
            img_path.write_bytes(image_bytes)

            evidence: list[FigureEvidence] = []
            if caption:
                evidence.append(
                    FigureEvidence(
                        source_file=pdf_path,
                        page=page_num,
                        quote=caption[:500],
                        bbox=bbox or None,
                    )
                )

            fe = FigureExtracted(
                figure_id=figure_id,
                page=page_num,
                bbox=bbox,
                caption=caption,
                image_sha256=sha,
                image_path=str(img_path.resolve()),
                extraction_method="embedded_image",
                confidence=0.85 if caption else 0.55,
                evidence=evidence,
            )

            if tier == "b" and digitize_flag:
                _apply_digitization(fe, cal_root or out_dir)

            figures.append(fe)

    doc.close()
    return figures


def _find_figure_caption(page_text: str) -> str | None:
    for line in page_text.splitlines():
        if _FIG_LINE.match(line.strip()):
            return line.strip()[:800]
    return None


def _apply_digitization(fig: FigureExtracted, cal_root: Path) -> None:
    cal_file = cal_root / f"{fig.figure_id}_calibration.json"
    if not cal_file.is_file():
        template = _default_calibration_template(fig)
        img_parent = Path(fig.image_path).parent if fig.image_path else cal_root
        tpl_path = img_parent / f"{fig.figure_id}_calibration.TEMPLATE.json"
        tpl_path.write_text(json.dumps(template, indent=2), encoding="utf-8")
        return

    data = json.loads(cal_file.read_text(encoding="utf-8"))
    pixels = _sample_curve_pixels(fig.image_path or "")
    mapped = [pixel_series_to_data(px, py, data) for px, py in pixels]
    fig.digitized_series = [{"x": xy[0], "y": xy[1]} for xy in mapped if xy]
    fig.digitization_metadata = {
        "calibration_file": str(cal_file),
        "source": "calibration_json",
        "raw_pixel_samples": len(pixels),
    }


def _default_calibration_template(fig: FigureExtracted) -> dict[str, Any]:
    return {
        "figure_id": fig.figure_id,
        "pixel_x0": 0,
        "pixel_y0": 0,
        "pixel_x1": 400,
        "pixel_y1": 300,
        "data_x0": 0.0,
        "data_x1": 1.0,
        "data_y0": 0.0,
        "data_y1": 1.0,
        "notes": "Edit pixel corners to axis box in image pixel space; set data_* to axis numeric endpoints.",
    }


def pixel_series_to_data(px: float, py: float, cal: dict[str, Any]) -> tuple[float, float] | None:
    """Linear map from pixel (px, py) to data coordinates using axis-aligned calibration box."""
    try:
        px0 = float(cal["pixel_x0"])
        py0 = float(cal["pixel_y0"])
        px1 = float(cal["pixel_x1"])
        py1 = float(cal["pixel_y1"])
        dx0 = float(cal["data_x0"])
        dx1 = float(cal["data_x1"])
        dy0 = float(cal["data_y0"])
        dy1 = float(cal["data_y1"])
    except (KeyError, TypeError, ValueError):
        return None
    if px1 == px0 or py1 == py0:
        return None
    x = dx0 + (px - px0) * (dx1 - dx0) / (px1 - px0)
    y = dy0 + (py - py0) * (dy1 - dy0) / (py1 - py0)
    return (x, y)


def _sample_curve_pixels(image_path: str, max_points: int = 40) -> list[tuple[float, float]]:
    """Bounded CV: intensity ridge sample (optional opencv)."""
    if not image_path or not Path(image_path).is_file():
        return []
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError:
        return []

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    h, w = img.shape[:2]
    points: list[tuple[float, float]] = []
    for col in range(0, w, max(1, w // max_points)):
        col_strip = img[:, col]
        row = int(col_strip.argmin()) if col_strip.size else 0
        points.append((float(col), float(row)))
    return points[:max_points]
