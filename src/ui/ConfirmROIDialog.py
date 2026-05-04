#
# =====================================================
# ConfirmROIDialog.py
#
# Small helper dialog used to accept, retry, or cancel
# an ROI selection while keeping the preview visible.
#
# ================================================

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout


class ConfirmROIDialog(QDialog):
    """Ask the user wether to keep, retry, or cancel the selected ROI."""

    def __init__(self, parent=None):
        """Build a small confirmation dialog for ROI selection"""
        super().__init__(parent)
        
        #Dialog state
        self.setWindowTitle("Confirm ROI")
        self.setWindowModality(Qt.NonModal) # pyright: ignore[reportAttributeAccessIssue]
        self.selection = "cancel"

        #Action Buttons
        self.button_box = QDialogButtonBox()
        ok_button = self.button_box.addButton(QDialogButtonBox.Ok)
        retry_button = self.button_box.addButton("Retry", QDialogButtonBox.ActionRole)
        cancel_button = self.button_box.addButton(QDialogButtonBox.Cancel)

        ok_button.clicked.connect(self._accept_roi) # pyright: ignore[reportOptionalMemberAccess]
        retry_button.clicked.connect(self._retry_roi) # pyright: ignore[reportOptionalMemberAccess]
        cancel_button.clicked.connect(self.reject) # pyright: ignore[reportOptionalMemberAccess]

        #Layout.
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Keep this ROI selection?"))
        layout.addWidget(self.button_box)
        self.setLayout(layout)
    
    def _accept_roi(self):
        """Accept the current ROI selection"""
        self.selection = "ok"
        self.accept()

    def _retry_roi(self):
        """Reject the current ROI and request a new selection"""
        self.selection = "retry"
        self.done(QDialog.Accepted)



