# Scanner Run Contract

This document defines the scanner-side contract that downstream integrations such as FLIm should consume.

## Coordinate Conventions

- `scanner_position_mm`: position relative to the captured machine home reference.
- `machine_position_mm`: raw GRBL machine coordinates (`MPos`).
- `work_position_mm`: raw GRBL work coordinates (`WPos`) when available.
- `tray_point_mm`: physical tray-space coordinates derived from the saved machine calibration.
- `target_machine_z_mm`: commanded scanner Z target for one raster point/segment in scanner-relative coordinates.

## Scan Modes

- `fixed_z`: raster rows run at a constant Z above the tray.
- `surface_following`: raster rows/segments run at local surface height + fibre standoff + probe safety margin.

## Stable Run Identity

Every raster run has:

- a `run_id` written into metadata and CSV outputs
- a run directory named `raster_scan_<timestamp>_<runid8>[_label]`

## Core Run Outputs

Each saved raster run produces:

- `scan_metadata.json`
- `raster_motion_log.csv`
- `raster_events.csv`
- `raster_step_settled.csv`
- `scan_samples.csv`
- `scan_samples_manifest.json`
- preview overlays/images
- preview video when available

## CSV Intent

- `raster_motion_log.csv`
  - periodic motion timeline during active raster execution
  - includes `run_id`, `scan_mode`, step metadata, tray XY, scanner/machine/work XYZ
  - current `tray_x_mm` / `tray_y_mm` values are the active step target coordinates, not continuously interpolated live tray coordinates along a moving row

- `raster_events.csv`
  - high-level run events such as prepared, started, settled, completed, aborted

- `raster_step_settled.csv`
  - one record per raster step that reached its settled target
  - this is the preferred scanner-side synchronization table for future FLIm acquisition

- `scan_samples.csv`
  - depth/topography sample captures saved during scan-row traversal

## Integration Hook

The scanner publishes a controller-level `after_step_settled` payload through `RasterAcquisitionHookController`.

Payload fields include:

- `run_id`
- `scan_mode`
- `step_index`
- `step_kind`
- `line_index`
- `segment_index`
- `point_id`
- `tray_point_mm`
- `scanner_position_mm`
- `machine_position_mm`
- `work_position_mm`

## Current Limitation

The current raster executor is still row/segment-based, not true point-by-point step-and-dwell. For first FLIm integration, the cleanest synchronization point is the settled-step event and `raster_step_settled.csv`.
