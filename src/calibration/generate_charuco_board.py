"""Generate a printable ChArUco board asset for the scan-space calibration workflow."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.charuco_calibration import build_charuco_board

OUTPUT_DIR = PROJECT_ROOT / "calibration_results" / "targets"

PAGE_WIDTH_MM = 210.0
PAGE_HEIGHT_MM = 297.0
PRINT_DPI = 300
PAGE_MARGIN_MM = 15.0


def mm_to_px(length_mm, dpi=PRINT_DPI):
    """Convert a physical length in millimetres into pixels for the target DPI."""
    return int(round((float(length_mm) / 25.4) * float(dpi)))


def main():
    """Render the fixed ChArUco board into an A4-sized printable PNG."""
    spec, board, _detector = build_charuco_board()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    board_width_mm = int(spec["squares_x"]) * float(spec["square_length_mm"])
    board_height_mm = int(spec["squares_y"]) * float(spec["square_length_mm"])

    board_width_px = mm_to_px(board_width_mm)
    board_height_px = mm_to_px(board_height_mm)
    page_width_px = mm_to_px(PAGE_WIDTH_MM)
    page_height_px = mm_to_px(PAGE_HEIGHT_MM)
    margin_px = mm_to_px(PAGE_MARGIN_MM)

    if board_width_px > (page_width_px - (2 * margin_px)):
        raise RuntimeError("Board width does not fit inside the configured A4 page margins.")
    if board_height_px > (page_height_px - (2 * margin_px)):
        raise RuntimeError("Board height does not fit inside the configured A4 page margins.")

    board_image = board.generateImage((board_width_px, board_height_px))
    page = np.full((page_height_px, page_width_px), 255, dtype="uint8")

    offset_x = (page_width_px - board_width_px) // 2
    offset_y = (page_height_px - board_height_px) // 2
    page[offset_y:offset_y + board_height_px, offset_x:offset_x + board_width_px] = board_image

    file_stem = (
        f"charuco_{spec['squares_x']}x{spec['squares_y']}_"
        f"{int(round(spec['square_length_mm']))}mm_"
        f"{int(round(spec['marker_length_mm']))}mm_"
        f"{PRINT_DPI}dpi"
    )
    png_path = OUTPUT_DIR / f"{file_stem}.png"
    metadata_path = OUTPUT_DIR / f"{file_stem}.json"

    cv2.imwrite(str(png_path), page)

    metadata = {
        "board_spec": spec,
        "output_png": str(png_path),
        "page": {
            "format": "A4",
            "width_mm": PAGE_WIDTH_MM,
            "height_mm": PAGE_HEIGHT_MM,
            "dpi": PRINT_DPI,
            "margin_mm": PAGE_MARGIN_MM,
        },
        "board_size_mm": {
            "width_mm": board_width_mm,
            "height_mm": board_height_mm,
        },
        "board_size_px": {
            "width_px": board_width_px,
            "height_px": board_height_px,
        },
        "print_instructions": [
            "Print at 100% scale.",
            "Disable fit-to-page or shrink-to-fit.",
            f"Verify one printed square measures {float(spec['square_length_mm']):.3f} mm.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved printable ChArUco board to {png_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
