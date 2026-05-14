"""Interactive viewer for one stitched raster reconstruction."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class RasterReconstructionViewerDialog(QDialog):
    """Inspect one stitched raster reconstruction as mesh, point cloud, and height map."""

    WINDOW_TITLE = "Raster Reconstruction Viewer"
    MAX_POINT_CLOUD_POINTS = 45000
    MAX_SURFACE_SAMPLES_PER_AXIS = 170

    def __init__(self, *, topography, output_paths, scan_path=None, parent=None):
        super().__init__(parent)
        self.topography = dict(topography or {})
        self.output_paths = dict(output_paths or {})

        self._height_map_mm = np.asarray(self.topography["height_map_mm"], dtype="float32")
        self._valid_mask = np.asarray(self.topography["valid_mask"], dtype=bool)

        raw_x_map = np.asarray(self.topography["x_map_mm"], dtype="float32")
        raw_y_map = np.asarray(self.topography["y_map_mm"], dtype="float32")
        self._local_x_mm, self._local_y_mm = self._build_local_xy_maps_mm(
            x_map_mm=raw_x_map,
            y_map_mm=raw_y_map,
            valid_mask=self._valid_mask,
        )
        if not np.any(self._valid_mask):
            raise ValueError("Raster reconstruction viewer received no valid height samples.")

        # Origin offset — needed to convert scan path to local coordinates.
        self._x_origin_mm = float(np.min(raw_x_map[self._valid_mask]))
        self._y_origin_mm = float(np.min(raw_y_map[self._valid_mask]))

        # Build scan path overlay (Layers 1 + 2).
        self._scan_path_local_x = None
        self._scan_path_local_y = None
        self._scan_path_z = None
        self._scan_path_line_indices = None
        if scan_path is not None:
            self._build_scan_path_local(scan_path)

        self.setWindowTitle(self.WINDOW_TITLE)
        self.setWindowModality(Qt.NonModal)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.resize(1500, 900)

        root_layout = QVBoxLayout()

        header_label = QLabel(self._build_header_text())
        header_label.setWordWrap(True)
        header_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root_layout.addWidget(header_label)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_view_tabs())
        splitter.addWidget(self._build_info_panel())
        splitter.setSizes([1120, 360])
        root_layout.addWidget(splitter, 1)

        self.hover_label = QLabel("Hover over the 2D probe map to inspect X/Y/height.")
        self.hover_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root_layout.addWidget(self.hover_label)

        self.setLayout(root_layout)

    def _build_view_tabs(self):
        tabs = QTabWidget()
        tabs.addTab(
            self._build_matplotlib_panel(self._draw_mesh_plot),
            "3D Mesh",
        )
        tabs.addTab(
            self._build_matplotlib_panel(self._draw_point_cloud_plot),
            "Point Cloud (preview)",
        )
        tabs.addTab(
            self._build_matplotlib_panel(
                self._draw_heatmap_plot,
                connect_hover=True,
            ),
            "2D Probe Map",
        )
        tabs.addTab(
            self._build_point_cloud_launcher_tab(),
            "3D Viewer",
        )
        return tabs

    def _build_matplotlib_panel(self, draw_callback, *, connect_hover=False):
        figure = Figure(figsize=(10.8, 7.2), constrained_layout=True)
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)
        draw_callback(figure, canvas)
        if connect_hover:
            canvas.mpl_connect("motion_notify_event", self._on_heatmap_hover)
            self.heatmap_canvas = canvas

        panel = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(canvas, 1)
        panel.setLayout(layout)
        return panel

    def _build_point_cloud_launcher_tab(self):
        """Build the tab that launches the Open3D interactive point cloud viewer."""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        desc = QLabel(
            "<b>Interactive 3D Point Cloud Viewer</b><br><br>"
            "Opens the full tissue point cloud in a separate window using Open3D — "
            "the same renderer used by the Intel RealSense Viewer. "
            "You can freely orbit, zoom, and pan around the tissue in 3D.<br><br>"
            "Choose how to colour the points:"
        )
        desc.setWordWrap(True)
        desc.setTextFormat(Qt.RichText)
        layout.addWidget(desc)
        layout.addSpacing(12)

        # ── Layer buttons ────────────────────────────────────────────────────
        layer_group = QGroupBox("Colour layer")
        layer_layout = QHBoxLayout()

        btn_rgb = QPushButton("📷  Camera RGB")
        btn_rgb.setToolTip(
            "Colour each 3D point with its actual camera colour.\n"
            "Shows the tissue texture and colour in true 3D."
        )
        btn_rgb.clicked.connect(
            lambda: self._launch_point_cloud_viewer("point_cloud_ply_path", "Camera RGB")
        )
        layer_layout.addWidget(btn_rgb)

        btn_height = QPushButton("🌈  Height")
        btn_height.setToolTip(
            "Colour each point by its height above the tray (viridis scale).\n"
            "Blue = low, yellow/green = high. Best for topography."
        )
        btn_height.clicked.connect(
            lambda: self._launch_npz_layer_viewer("height", "Height")
        )
        layer_layout.addWidget(btn_height)

        btn_lines = QPushButton("🔢  Scan Lines")
        btn_lines.setToolTip(
            "Colour each point by which scan line captured it.\n"
            "Useful to check scan coverage and density."
        )
        btn_lines.clicked.connect(
            lambda: self._launch_npz_layer_viewer("lines", "Scan Lines")
        )
        layer_layout.addWidget(btn_lines)

        layer_group.setLayout(layer_layout)
        layout.addWidget(layer_group)
        layout.addSpacing(12)

        # ── Status label ─────────────────────────────────────────────────────
        self._point_cloud_status = QLabel(
            "Click a layer button above to open the 3D viewer.\n"
            "The viewer opens in a separate window — you can keep this dialog open."
        )
        self._point_cloud_status.setWordWrap(True)
        self._point_cloud_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._point_cloud_status)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def _launch_npz_layer_viewer(self, layer, layer_name):
        """Generate a coloured PLY from point_cloud.npz on demand, then open it.

        The generated PLY is cached alongside the NPZ (e.g. ``*_height.ply``)
        so subsequent opens are instant.  *layer* is one of ``"height"`` or
        ``"lines"``.
        """
        npz_key = "point_cloud_npz_path"
        npz_path = self.output_paths.get(npz_key)
        if not npz_path or not Path(str(npz_path)).exists():
            self._point_cloud_status.setText(
                f"Point cloud NPZ not found — re-run the reconstruction to generate it."
            )
            return

        npz_path = Path(str(npz_path))
        ply_path = npz_path.parent / f"{npz_path.stem}_{layer}.ply"
        if not ply_path.exists():
            try:
                self._point_cloud_status.setText(
                    f"Generating '{layer_name}' colour layer from NPZ…"
                )
                data = np.load(str(npz_path))
                xyz = data["xyz"]
                if layer == "height":
                    rgb = self._height_to_rgb(xyz[:, 2])
                else:
                    rgb = self._line_index_to_rgb(data["line_index"])
                self._write_coloured_ply(ply_path, xyz, rgb)
            except Exception as exc:
                self._point_cloud_status.setText(
                    f"Failed to generate '{layer_name}' PLY: {exc}"
                )
                return

        self._launch_point_cloud_viewer_path(str(ply_path), layer_name)

    def _launch_point_cloud_viewer(self, ply_key, layer_name):
        """Launch the Open3D point cloud viewer for a PLY stored in output_paths."""
        ply_path = self.output_paths.get(ply_key)
        if not ply_path or not Path(str(ply_path)).exists():
            self._point_cloud_status.setText(
                f"Point cloud file not found for layer '{layer_name}'.\n"
                "Re-run the reconstruction to generate it — make sure at least one scan "
                "was done after this session's colour-frame save was enabled."
            )
            return
        self._launch_point_cloud_viewer_path(str(ply_path), layer_name)

    def _launch_point_cloud_viewer_path(self, ply_path_str_in, layer_name):
        """Open an Open3D viewer subprocess for a PLY at the given path."""
        ply_path_str = str(Path(ply_path_str_in)).replace("\\", "\\\\")
        viewer_code = (
            "import sys\n"
            "try:\n"
            "    import open3d as o3d\n"
            "except ImportError:\n"
            "    print('open3d not installed. Run: pip install open3d', flush=True)\n"
            "    sys.exit(1)\n"
            f"pcd = o3d.io.read_point_cloud(r'{ply_path_str}')\n"
            f"o3d.visualization.draw_geometries(\n"
            f"    [pcd],\n"
            f"    window_name='Tissue Point Cloud — {layer_name}',\n"
            f"    width=1280, height=800,\n"
            f"    point_show_normal=False,\n"
            f")\n"
        )
        try:
            subprocess.Popen(
                [sys.executable, "-c", viewer_code],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._point_cloud_status.setText(
                f"Opening '{layer_name}' point cloud viewer…\n"
                f"File: {ply_path_str_in}\n\n"
                "Controls in the viewer window:\n"
                "  • Left-drag  →  orbit\n"
                "  • Right-drag →  pan\n"
                "  • Scroll     →  zoom\n"
                "  • R          →  reset view\n"
                "  • Q / Esc    →  close viewer"
            )
        except Exception as exc:
            self._point_cloud_status.setText(
                f"Failed to launch viewer: {exc}\n"
                "Make sure open3d is installed: pip install open3d"
            )

    def _build_info_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()

        info_label = QLabel("Reconstruction Outputs")
        info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(info_label)

        info_text = QPlainTextEdit()
        info_text.setReadOnly(True)
        info_text.setPlainText(self._build_output_summary_text())
        layout.addWidget(info_text, 1)

        hint_label = QLabel(
            "Controls: use the toolbar to pan/zoom/save. Drag the 3D views to orbit. "
            "Use the 2D probe map for precise X/Y/height readout."
        )
        hint_label.setWordWrap(True)
        hint_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(hint_label)

        panel.setLayout(layout)
        return panel

    def _draw_mesh_plot(self, figure, _canvas):
        ax = figure.add_subplot(1, 1, 1, projection="3d")
        x_mm = np.asarray(self._local_x_mm, dtype="float32")
        y_mm = np.asarray(self._local_y_mm, dtype="float32")
        z_mm = np.asarray(self._height_map_mm, dtype="float32")
        valid_mask = np.asarray(self._valid_mask, dtype=bool)

        row_stride = max(1, int(np.ceil(z_mm.shape[0] / self.MAX_SURFACE_SAMPLES_PER_AXIS)))
        col_stride = max(1, int(np.ceil(z_mm.shape[1] / self.MAX_SURFACE_SAMPLES_PER_AXIS)))
        z_masked = np.ma.masked_invalid(np.where(valid_mask, z_mm, np.nan))
        colors = cm.get_cmap("viridis")(self._normalize_heights(z_mm, valid_mask))
        colors[..., 3] = np.where(valid_mask, 1.0, 0.0)

        ax.plot_surface(
            x_mm,
            y_mm,
            z_masked,
            facecolors=colors,
            linewidth=0.0,
            antialiased=True,
            shade=False,
            rstride=row_stride,
            cstride=col_stride,
        )

        # Layer 2 — scan path overlay.
        if self._scan_path_local_x is not None and self._scan_path_local_x.size > 0:
            valid_pts = np.isfinite(self._scan_path_z)
            if np.any(valid_pts):
                px = self._scan_path_local_x[valid_pts]
                py = self._scan_path_local_y[valid_pts]
                # Lift path slightly above surface so it's always visible.
                pz = self._scan_path_z[valid_pts] + 0.8
                # Connect all points as a single path line.
                ax.plot(
                    px, py, pz,
                    color="white", linewidth=0.7, alpha=0.55, zorder=4,
                )
                # Colour each sample dot by scan-line index for easy row identification.
                line_idx = self._scan_path_line_indices[valid_pts].astype("float32")
                sc = ax.scatter(
                    px, py, pz,
                    c=line_idx, cmap="plasma",
                    s=14, alpha=0.95, linewidths=0, zorder=5,
                )
                figure.colorbar(
                    sc, ax=ax, shrink=0.55, pad=0.10, label="Scan line index",
                )

        self._style_3d_axes(ax, title="Tissue Surface + Scan Path")

    def _draw_point_cloud_plot(self, figure, _canvas):
        ax = figure.add_subplot(1, 1, 1, projection="3d")
        x_points = self._local_x_mm[self._valid_mask].reshape(-1)
        y_points = self._local_y_mm[self._valid_mask].reshape(-1)
        z_points = self._height_map_mm[self._valid_mask].reshape(-1)

        if x_points.size > self.MAX_POINT_CLOUD_POINTS:
            stride = int(np.ceil(x_points.size / self.MAX_POINT_CLOUD_POINTS))
            x_points = x_points[::stride]
            y_points = y_points[::stride]
            z_points = z_points[::stride]

        scatter = ax.scatter(
            x_points,
            y_points,
            z_points,
            c=z_points,
            cmap="viridis",
            s=2.0,
            linewidths=0.0,
            alpha=0.85,
        )
        figure.colorbar(scatter, ax=ax, shrink=0.70, pad=0.08, label="Height (mm)")
        self._style_3d_axes(ax, title=f"Point Cloud ({int(x_points.size)} displayed points)")

    def _draw_heatmap_plot(self, figure, _canvas):
        self.heatmap_ax = figure.add_subplot(1, 1, 1)
        x_valid = self._local_x_mm[self._valid_mask]
        y_valid = self._local_y_mm[self._valid_mask]
        extent = (
            float(np.min(x_valid)),
            float(np.max(x_valid)),
            float(np.max(y_valid)),
            float(np.min(y_valid)),
        )
        heatmap = np.ma.masked_invalid(np.where(self._valid_mask, self._height_map_mm, np.nan))
        image = self.heatmap_ax.imshow(
            heatmap,
            origin="upper",
            extent=extent,
            interpolation="none",
            cmap="viridis",
            aspect="equal",
        )
        self.heatmap_ax.set_title("2D Probe Map")
        self.heatmap_ax.set_xlabel("X (mm)")
        self.heatmap_ax.set_ylabel("Y (mm)")
        self.heatmap_crosshair_h = self.heatmap_ax.axhline(
            y=float(np.mean(y_valid)),
            color="white",
            linewidth=0.7,
            alpha=0.0,
        )
        self.heatmap_crosshair_v = self.heatmap_ax.axvline(
            x=float(np.mean(x_valid)),
            color="white",
            linewidth=0.7,
            alpha=0.0,
        )
        figure.colorbar(
            image,
            ax=self.heatmap_ax,
            fraction=0.046,
            pad=0.04,
            label="Height above plane (mm)",
        )

        # Layer 2 — scan path overlay on the 2D map.
        if self._scan_path_local_x is not None and self._scan_path_local_x.size > 0:
            self.heatmap_ax.plot(
                self._scan_path_local_x,
                self._scan_path_local_y,
                color="white", linewidth=0.6, alpha=0.45, zorder=4,
            )
            line_idx = self._scan_path_line_indices.astype("float32")
            self.heatmap_ax.scatter(
                self._scan_path_local_x,
                self._scan_path_local_y,
                c=line_idx, cmap="plasma",
                s=8, alpha=0.9, linewidths=0, zorder=5,
            )

    def _build_scan_path_local(self, scan_path):
        """Convert the raw scan path (stitched tray-delta space) to local viewer coordinates."""
        raw_x = np.asarray(scan_path.get("x_mm") or [], dtype="float32")
        raw_y = np.asarray(scan_path.get("y_mm") or [], dtype="float32")
        if raw_x.size == 0:
            return
        # Shift to the same local origin used by the height map / mesh.
        local_x = raw_x - self._x_origin_mm
        local_y = raw_y - self._y_origin_mm
        path_z = self._interpolate_height_at_path_points(local_x, local_y)

        raw_line_idx = scan_path.get("line_indices") or []
        line_idx = np.asarray(
            [float(v) if v is not None else float("nan") for v in raw_line_idx],
            dtype="float32",
        )
        # Replace NaN line indices with 0 so colormap works.
        line_idx = np.where(np.isfinite(line_idx), line_idx, 0.0)

        self._scan_path_local_x = local_x
        self._scan_path_local_y = local_y
        self._scan_path_z = path_z
        self._scan_path_line_indices = line_idx

    def _interpolate_height_at_path_points(self, local_x, local_y):
        """Nearest-neighbour height lookup for scan path points on the regular grid."""
        h, w = self._height_map_mm.shape
        valid_local_x = self._local_x_mm[self._valid_mask]
        valid_local_y = self._local_y_mm[self._valid_mask]
        if valid_local_x.size == 0:
            return np.full(local_x.shape, np.nan, dtype="float32")

        # Estimate grid step from the span and grid dimensions.
        x_span = float(np.max(valid_local_x) - np.min(valid_local_x))
        y_span = float(np.max(valid_local_y) - np.min(valid_local_y))
        grid_step_x = x_span / max(w - 1, 1)
        grid_step_y = y_span / max(h - 1, 1)

        path_z = np.full(local_x.shape, np.nan, dtype="float32")
        for i in range(local_x.size):
            col = int(round(float(local_x[i]) / grid_step_x)) if grid_step_x > 0 else 0
            row = int(round(float(local_y[i]) / grid_step_y)) if grid_step_y > 0 else 0
            col = max(0, min(col, w - 1))
            row = max(0, min(row, h - 1))
            if self._valid_mask[row, col]:
                path_z[i] = float(self._height_map_mm[row, col])
        return path_z


    def _style_3d_axes(self, ax, *, title):
        ax.set_title(title)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Height (mm)")
        ax.view_init(elev=28.0, azim=-130.0)
        self._set_equal_3d_box_aspect(ax)

    def _set_equal_3d_box_aspect(self, ax):
        x_valid = self._local_x_mm[self._valid_mask]
        y_valid = self._local_y_mm[self._valid_mask]
        z_valid = self._height_map_mm[self._valid_mask]
        spans = np.asarray(
            [
                max(float(np.max(x_valid) - np.min(x_valid)), 1.0),
                max(float(np.max(y_valid) - np.min(y_valid)), 1.0),
                max(float(np.max(z_valid) - np.min(z_valid)), 1.0),
            ],
            dtype="float64",
        )
        try:
            ax.set_box_aspect(spans / float(np.max(spans)))
        except Exception:
            pass

    def _on_heatmap_hover(self, event):
        if event.inaxes != self.heatmap_ax or event.xdata is None or event.ydata is None:
            self._hide_heatmap_crosshair()
            self.hover_label.setText("Hover over the 2D probe map to inspect X/Y/height.")
            self.heatmap_canvas.draw_idle()
            return

        row_index, col_index = self._find_nearest_valid_cell(event.xdata, event.ydata)
        if row_index is None or col_index is None:
            self._hide_heatmap_crosshair()
            self.hover_label.setText("No valid reconstruction sample at this location.")
            self.heatmap_canvas.draw_idle()
            return

        x_mm = float(self._local_x_mm[row_index, col_index])
        y_mm = float(self._local_y_mm[row_index, col_index])
        z_mm = float(self._height_map_mm[row_index, col_index])
        self.heatmap_crosshair_h.set_ydata([y_mm, y_mm])
        self.heatmap_crosshair_v.set_xdata([x_mm, x_mm])
        self.heatmap_crosshair_h.set_alpha(0.8)
        self.heatmap_crosshair_v.set_alpha(0.8)
        self.hover_label.setText(
            f"X {x_mm:.2f} mm | Y {y_mm:.2f} mm | Height {z_mm:.2f} mm"
        )
        self.heatmap_canvas.draw_idle()

    def _hide_heatmap_crosshair(self):
        self.heatmap_crosshair_h.set_alpha(0.0)
        self.heatmap_crosshair_v.set_alpha(0.0)

    def _find_nearest_valid_cell(self, x_mm, y_mm):
        x_map_mm = np.asarray(self._local_x_mm, dtype="float32")
        y_map_mm = np.asarray(self._local_y_mm, dtype="float32")
        valid_indices = np.argwhere(self._valid_mask)
        if valid_indices.size == 0:
            return None, None
        valid_x = x_map_mm[self._valid_mask]
        valid_y = y_map_mm[self._valid_mask]
        deltas = np.square(valid_x - float(x_mm)) + np.square(valid_y - float(y_mm))
        best_index = int(np.argmin(deltas))
        row_index, col_index = valid_indices[best_index]
        return int(row_index), int(col_index)

    def _build_header_text(self):
        valid_values_mm = self._height_map_mm[self._valid_mask]
        x_span = float(np.nanmax(self._local_x_mm) - np.nanmin(self._local_x_mm))
        y_span = float(np.nanmax(self._local_y_mm) - np.nanmin(self._local_y_mm))
        return (
            "Interactive stitched reconstruction. "
            f"Peak {float(np.max(valid_values_mm)):.2f} mm | "
            f"Median {float(np.median(valid_values_mm)):.2f} mm | "
            f"Footprint {x_span:.1f} x {y_span:.1f} mm | "
            f"Valid samples {int(valid_values_mm.size)}"
        )

    def _build_output_summary_text(self):
        lines = []
        for key, value in sorted(self.output_paths.items()):
            lines.append(f"{key}: {self._format_output_value(value)}")
        return "\n".join(lines)

    @staticmethod
    def _format_output_value(value):
        if value is None:
            return "N/A"
        return str(value)

    # ── Static helpers for on-demand layer PLY generation ─────────────────────

    @staticmethod
    def _height_to_rgb(heights):
        """Map height values to RGB using a viridis-like colormap (float [0..1])."""
        h = np.asarray(heights, dtype="float32")
        finite = np.isfinite(h)
        rgb = np.zeros((len(h), 3), dtype="float32")
        if np.any(finite):
            h_min = float(np.min(h[finite]))
            h_max = float(np.max(h[finite]))
            span = h_max - h_min if h_max > h_min else 1.0
            t = np.where(finite, (h - h_min) / span, 0.5).astype("float32")
            rgb[:, 0] = np.clip(1.5 * t - 0.25, 0, 1)           # R
            rgb[:, 1] = np.clip(np.sin(t * np.pi), 0, 1)         # G
            rgb[:, 2] = np.clip(1.0 - 1.5 * t + 0.25, 0, 1)     # B
        return rgb

    @staticmethod
    def _line_index_to_rgb(line_indices):
        """Map scan line indices to distinct RGB colours (tab10 palette)."""
        palette = np.array([
            [0.122, 0.467, 0.706], [1.000, 0.498, 0.055],
            [0.173, 0.627, 0.173], [0.839, 0.153, 0.157],
            [0.580, 0.404, 0.741], [0.549, 0.337, 0.294],
            [0.890, 0.467, 0.761], [0.498, 0.498, 0.498],
            [0.737, 0.741, 0.133], [0.090, 0.745, 0.812],
        ], dtype="float32")
        indices = np.asarray(line_indices, dtype="int32")
        valid = indices >= 0
        rgb = np.full((len(indices), 3), 0.5, dtype="float32")
        if np.any(valid):
            rgb[valid] = palette[indices[valid] % len(palette)]
        return rgb

    @staticmethod
    def _write_coloured_ply(path, xyz, rgb_float):
        """Write a binary little-endian PLY point cloud with per-point RGB colour."""
        xyz = np.asarray(xyz, dtype="float32")
        rgb = np.clip(np.asarray(rgb_float, dtype="float32") * 255, 0, 255).astype("uint8")
        n = len(xyz)
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"),
                          ("r", "u1"), ("g", "u1"), ("b", "u1")])
        data = np.empty(n, dtype=dtype)
        data["x"] = xyz[:, 0]; data["y"] = xyz[:, 1]; data["z"] = xyz[:, 2]
        data["r"] = rgb[:, 0]; data["g"] = rgb[:, 1]; data["b"] = rgb[:, 2]
        with open(str(path), "wb") as fh:
            fh.write(header.encode("ascii"))
            fh.write(data.tobytes())

    def _format_output_value(self, value):
        if isinstance(value, (str, Path)):
            return str(Path(value))
        if isinstance(value, dict):
            return json.dumps(value, indent=2, default=str)
        if isinstance(value, (list, tuple)):
            return "\n".join(f"  - {self._format_output_value(item)}" for item in value)
        return str(value)

    @staticmethod
    def _build_local_xy_maps_mm(*, x_map_mm, y_map_mm, valid_mask):
        x_map_mm = np.asarray(x_map_mm, dtype="float32")
        y_map_mm = np.asarray(y_map_mm, dtype="float32")
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if not np.any(valid_mask):
            raise ValueError("Cannot build local reconstruction coordinates without valid samples.")
        local_x_mm = np.full_like(x_map_mm, np.nan, dtype="float32")
        local_y_mm = np.full_like(y_map_mm, np.nan, dtype="float32")
        x_origin_mm = float(np.min(x_map_mm[valid_mask]))
        y_origin_mm = float(np.min(y_map_mm[valid_mask]))
        local_x_mm[valid_mask] = x_map_mm[valid_mask] - x_origin_mm
        local_y_mm[valid_mask] = y_map_mm[valid_mask] - y_origin_mm
        return local_x_mm, local_y_mm

    @staticmethod
    def _normalize_heights(height_map_mm, valid_mask):
        valid_values = height_map_mm[valid_mask]
        normalized = np.zeros_like(height_map_mm, dtype="float32")
        if valid_values.size == 0:
            return normalized
        height_min = float(np.min(valid_values))
        height_max = float(np.max(valid_values))
        span = max(height_max - height_min, 1e-6)
        normalized[valid_mask] = (height_map_mm[valid_mask] - height_min) / span
        return normalized
