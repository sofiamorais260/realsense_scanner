"""Qt workflow for reconstructing raster runs without blocking the main window."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import QProgressDialog

from src.worker.RasterReconstructionWorker import RasterReconstructionWorker
from src.ui.RasterReconstructionViewerDialog import RasterReconstructionViewerDialog


class RasterReconstructionWorkflow(QObject):
    """Own the background reconstruction task, progress UI, and result viewer."""

    message = pyqtSignal(str)

    def __init__(
        self,
        *,
        reconstruction_controller,
        artifact_controller,
        parent=None,
    ):
        super().__init__(parent)
        self.reconstruction_controller = reconstruction_controller
        self.artifact_controller = artifact_controller
        self._thread = None
        self._worker = None
        self._progress_dialog = None
        self._viewer_dialog = None
        self._active_run_dir = None

    @property
    def is_active(self):
        return self._thread is not None and self._thread.isRunning()

    def start(self, *, run_dir, scan_calibration):
        if self.is_active:
            raise RuntimeError("Raster reconstruction is already running.")

        self._active_run_dir = Path(run_dir)
        self._progress_dialog = QProgressDialog(
            "Building raster reconstruction...",
            None,
            0,
            0,
            self.parent(),
        )
        self._progress_dialog.setWindowTitle("Raster Reconstruction")
        self._progress_dialog.setWindowModality(Qt.NonModal)
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.setAutoClose(False)
        self._progress_dialog.setAutoReset(False)
        self._progress_dialog.show()

        self._thread = QThread(self)
        self._worker = RasterReconstructionWorker(
            run_dir=self._active_run_dir,
            scan_calibration=scan_calibration,
            reconstruction_controller=self.reconstruction_controller,
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_progress(self, text):
        text = str(text)
        self.message.emit(text)
        if self._progress_dialog is not None:
            self._progress_dialog.setLabelText(text)

    def _on_finished(self, result):
        result = dict(result or {})
        try:
            self.artifact_controller.attach_reconstruction_outputs_to_run_dir(
                run_dir=self._active_run_dir,
                output_paths=result["output_paths"],
            )
        except Exception as exc:
            print(f"Failed to attach reconstruction outputs to raster metadata: {exc}")

        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None

        message = str(result.get("message") or "Raster reconstruction complete.")
        try:
            self._viewer_dialog = RasterReconstructionViewerDialog(
                topography=result["topography"],
                output_paths=result["output_paths"],
                scan_path=result.get("scan_path"),
                parent=self.parent(),
            )
            self._viewer_dialog.show()
            self._viewer_dialog.raise_()
            self._viewer_dialog.activateWindow()
        except Exception as exc:
            message = (
                f"{message}\n"
                f"Reconstruction was saved, but the viewer could not open: {exc}"
            )
        print(message)
        self.message.emit(message)

    def _on_failed(self, message):
        message = str(message)
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None
        print(message)
        self.message.emit(message)

    def _cleanup_thread(self):
        self._thread = None
        self._worker = None
        self._active_run_dir = None
