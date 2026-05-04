# =====================================================
# DepthProfileAnalysisDialog.py
#
# Lightweight popup for the quick depth-profile
# validation analysis. Shows the combined analysis
# figure, key numbers, and lets the user save one PNG.
#
# =====================================================

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QVBoxLayout,
)


class DepthProfileAnalysisDialog(QDialog):
    """Show one quick depth-profile analysis view without writing many files."""

    def __init__(self, analysis_payload, parent=None):
        super().__init__(parent)
        self.analysis_payload = analysis_payload or {}
        self._png_bytes = self.analysis_payload.get("png_bytes", b"")
        self._default_save_dir = Path(self.analysis_payload.get("default_save_dir", str(Path.cwd())))
        self._default_save_name = self.analysis_payload.get(
            "default_save_name",
            "depth_profile_quick_analysis.png",
        )
        self._source_pixmap = None

        self.setWindowTitle("Depth Profile Analysis")
        self.resize(920, 720)

        self.summary_text = QPlainTextEdit(self)
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlainText("\n".join(self.analysis_payload.get("summary_lines", [])))
        self.summary_text.setFixedHeight(180)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setBackgroundRole(self.image_label.backgroundRole())

        self.image_scroll = QScrollArea(self)
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setWidget(self.image_label)
        self._set_image_from_bytes(self._png_bytes)

        self.save_button = QPushButton("Save PNG", self)
        self.save_button.clicked.connect(self._save_png)
        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.accept)

        button_row = QHBoxLayout()
        button_row.addWidget(self.save_button)
        button_row.addStretch(1)
        button_row.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.summary_text)
        layout.addWidget(self.image_scroll, 1)
        layout.addLayout(button_row)

    def _set_image_from_bytes(self, png_bytes):
        """Decode the in-memory PNG into a scrollable Qt pixmap."""
        if not png_bytes:
            self.image_label.setText("No analysis image is available.")
            return

        image = QImage.fromData(png_bytes, "PNG")
        if image.isNull():
            self.image_label.setText("The analysis image could not be decoded.")
            return

        self._source_pixmap = QPixmap.fromImage(image)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        """Scale the quick-analysis image to the current dialog viewport."""
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return

        if not hasattr(self, "image_scroll") or self.image_scroll is None:
            return

        viewport_size = self.image_scroll.viewport().size()
        if viewport_size.width() <= 0 or viewport_size.height() <= 0:
            return

        scaled_pixmap = self._source_pixmap.scaled(
            viewport_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())

    def resizeEvent(self, a0):
        """Keep the analysis figure scaled to the dialog whenever it is resized."""
        super().resizeEvent(a0)
        self._update_scaled_pixmap()

    def _save_png(self):
        """Save the currently displayed analysis image as one standalone PNG."""
        if not self._png_bytes:
            return

        default_path = self._default_save_dir / self._default_save_name
        default_path.parent.mkdir(parents=True, exist_ok=True)
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Quick Analysis PNG",
            str(default_path),
            "PNG Files (*.png)",
        )
        if not output_path:
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(self._png_bytes)



