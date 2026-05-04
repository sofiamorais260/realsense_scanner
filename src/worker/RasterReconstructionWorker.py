"""Worker-thread object for raster reconstruction."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from src.calibration.charuco_calibration import CalibrationError
from src.controllers.raster_reconstruction_controller import RasterReconstructionError
from src.ui.topography_tools import TopographyTools


class RasterReconstructionWorker(QObject):
    """Run the expensive raster reconstruction off the Qt UI thread."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        run_dir,
        scan_calibration,
        reconstruction_controller,
    ):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.scan_calibration = dict(scan_calibration or {})
        self.reconstruction_controller = reconstruction_controller

    @pyqtSlot()
    def run(self):
        try:
            self.progress.emit("Building stitched raster reconstruction...")
            reconstruction_root = self.run_dir / "reconstruction"
            topography_tools = TopographyTools(output_root=reconstruction_root)
            result = self.reconstruction_controller.reconstruct_run(
                run_dir=self.run_dir,
                scan_calibration=self.scan_calibration,
                topography_tools=topography_tools,
                show_preview=False,
            )
            self.finished.emit(result)
        except (CalibrationError, RasterReconstructionError) as exc:
            self.failed.emit(f"Raster reconstruction failed: {exc}")
        except Exception as exc:
            self.failed.emit(f"Raster reconstruction failed unexpectedly: {exc}")
