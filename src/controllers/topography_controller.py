"""Topography workflow helpers kept out of the Qt main window."""

from __future__ import annotations

import numpy as np

from src.calibration.charuco_calibration import build_robust_depth_frame_mm, compute_topography_map


class TopographyController:
    """Build calibrated topography outputs from captured depth snapshots."""

    def generate_topography_report(
        self,
        *,
        snapshots,
        calibration,
        intrinsics,
        roi_box,
        depth_scale_mm,
        topography_tools,
    ):
        """Compute, persist, and render one topography result bundle."""
        depth_stack = np.stack(
            [snapshot["frame_depth"] for snapshot in snapshots],
            axis=0,
        ).astype("float32")
        robust_depth_frame_mm, aggregation_summary = build_robust_depth_frame_mm(
            depth_stack * float(depth_scale_mm)
        )
        topography = compute_topography_map(
            frame_depth=robust_depth_frame_mm,
            depth_scale_mm=1.0,
            intrinsics=intrinsics,
            roi_box=roi_box,
            xy_homography=calibration["xy_homography"],
            plane_model=calibration["plane_model"],
            z_scale=calibration["z_scale"],
            z_bias_mm=calibration.get("z_bias_mm", 0.0),
        )
        topography["aggregation_summary"] = aggregation_summary
        report_topography = topography_tools.prepare_for_report(topography)
        output_paths = topography_tools.save_capture(report_topography, calibration)
        topography_tools.render_report(
            topography=report_topography,
            calibration=calibration,
            png_path=output_paths["png_path"],
        )
        topography_tools.show_preview(output_paths["png_path"])

        summary_payload = output_paths["summary_payload"]
        kept_fraction = float(
            summary_payload.get("aggregation_summary", {}).get("kept_valid_sample_fraction", 1.0)
        )
        message = (
            f"Topography saved to {output_paths['bundle_path']} | "
            f"mesh {output_paths['mesh_path']} | "
            f"stable peak {float(summary_payload['stable_peak_height_mm']):.3f} mm | "
            f"max {float(summary_payload['max_height_mm']):.3f} mm | "
            f"median {float(summary_payload['median_height_mm']):.3f} mm | "
            f"kept {kept_fraction * 100.0:.1f}% of valid temporal samples | "
            f"using saved calibration XY {float(calibration.get('xy_scale_mm_per_px', 0.0)):.4f} mm/px, "
            f"Z {float(calibration.get('z_scale', 1.0)):.4f}x + {float(calibration.get('z_bias_mm', 0.0)):.4f} mm"
        )
        return {
            "topography": report_topography,
            "output_paths": output_paths,
            "message": message,
        }
