#
# =====================================================
# imageprocessing.py
#
# Helper functions used by the UI and worker to define,
# detect, and track the ROI in the RealSense color stream.
#
# =====================================================

import cv2
import numpy as np


AUTO_ROI_DEBUG = False


# -----------------------------------------------------
# ROI selection
# -----------------------------------------------------


def manual_roi_from_frame(color_image, window_name="color"):
    """Let the user draw the initial ROI directly on the current color window."""
    state = {
        "start": None,
        "current": None,
        "drawing": False,
        "roi": None,
    }

    def on_mouse(event, x, y, _flags, _param):
        """Track the drag rectangle used to define the ROI."""
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = (x, y)
            state["current"] = (x, y)
            state["drawing"] = True
            state["roi"] = None
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["current"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["current"] = (x, y)
            state["drawing"] = False

            x0, y0 = state["start"]
            x1, y1 = state["current"]
            left = min(x0, x1)
            top = min(y0, y1)
            width = abs(x1 - x0)
            height = abs(y1 - y0)

            if width > 0 and height > 0:
                state["roi"] = (left, top, width, height)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        preview = color_image.copy()
        if state["start"] is not None and state["current"] is not None:
            x0, y0 = state["start"]
            x1, y1 = state["current"]
            cv2.rectangle(
                preview,
                (min(x0, x1), min(y0, y1)),
                (max(x0, x1), max(y0, y1)),
                (0, 255, 0),
                2,
            )

        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF

        # Allow the user to abort ROI selection with Escape or by closing the window.
        if key == 27:
            cv2.setMouseCallback(window_name, lambda *_args: None)
            return None

        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                return None
        except cv2.error:
            return None

        if state["roi"] is not None:
            cv2.setMouseCallback(window_name, lambda *_args: None)
            return state["roi"]



# -----------------------------------------------------
# ROI detection
# -----------------------------------------------------


def auto_roi_from_frame(color_image):
    """Detect the object ROI automatically from the current color frame."""
    stage_rect = find_dark_stage(color_image, shrink_margin=20)
    frame_h, frame_w = color_image.shape[:2]
    stage_area = stage_rect[2] * stage_rect[3]
    frame_area = frame_w * frame_h

    # Prefer a dark-background stage and then detect the brighter specimen on top of it.
    if stage_area < 0.98 * frame_area:
        detected = detect_object_on_dark_stage(
            color_image,
            stage_rect,
            fallback_to_region=False,
        )
        if detected is not None:
            return detected

    # If stage detection is unreliable, still try to find the specimen directly.
    full_frame_box, score = _find_object_bbox_in_region(
        color_image,
        allow_non_dark_fallback=True,
    )
    if full_frame_box is None or score <= 0.0:
        return None

    x_o, y_o, w_o, h_o = full_frame_box
    pad = 6
    x_o = max(0, x_o - pad)
    y_o = max(0, y_o - pad)
    w_o = min(frame_w - x_o, w_o + 2 * pad)
    h_o = min(frame_h - y_o, h_o + 2 * pad)
    return (x_o, y_o, w_o, h_o)


def find_dark_stage(color_image, shrink_margin=5):
    """Detect the dark support stage and return a slightly shrunken bounding box."""
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(color_image)

    # Black-background workflow: low value, low saturation, and dark grayscale pixels.
    mask_hsv = cv2.inRange(hsv, (0, 0, 0), (179, 115, 105))  # pyright: ignore[reportArgumentType, reportCallIssue]
    mask_gray = (gray <= 105).astype(np.uint8) * 255
    chroma_span = np.maximum.reduce([r, g, b]).astype(np.int16) - np.minimum.reduce([r, g, b]).astype(np.int16)
    mask_low_chroma = (chroma_span <= 42).astype(np.uint8) * 255
    mask_stage = cv2.bitwise_and(mask_hsv, mask_gray)
    mask_stage = cv2.bitwise_and(mask_stage, mask_low_chroma)

    if AUTO_ROI_DEBUG:
        cv2.imshow("Auto ROI - dark stage raw", mask_stage)
        cv2.waitKey(1)

    mask_stage = cv2.medianBlur(mask_stage, 5)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_stage = cv2.morphologyEx(mask_stage, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask_stage = cv2.morphologyEx(mask_stage, cv2.MORPH_OPEN, open_kernel, iterations=1)

    if AUTO_ROI_DEBUG:
        cv2.imshow("Auto ROI - dark stage cleaned", mask_stage)
        cv2.waitKey(1)

    contours, _ = cv2.findContours(mask_stage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, color_image.shape[1], color_image.shape[0])

    filtered = []
    border_touching = []   # (area, contour) pairs — kept separate, used only as fallback
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 2000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if h <= 0:
            continue
        aspect_ratio = w / float(h)
        extent = area / float(max(w * h, 1))
        touches_border = (
            x <= 2
            or y <= 2
            or (x + w) >= (color_image.shape[1] - 2)
            or (y + h) >= (color_image.shape[0] - 2)
        )
        if 0.3 < aspect_ratio < 4.0 and extent > 0.18:
            if touches_border:
                border_touching.append((area, contour))
            else:
                filtered.append(contour)

    # Prefer interior contours.  Only fall back to border-touching ones when
    # nothing else passed — and even then require the contour to cover at least
    # 25 % of the frame so that shadows, probe occlusions, and gantry edges
    # (which are typically thin slivers) cannot hijack stage detection.
    if not filtered:
        frame_pixel_count = color_image.shape[0] * color_image.shape[1]
        filtered = [
            contour
            for area, contour in border_touching
            if area > 0.25 * frame_pixel_count
        ]

    if not filtered:
        return (0, 0, color_image.shape[1], color_image.shape[0])

    # Fill the hull of the best contour to absorb holes caused by lighting on the fabric.
    stage_contour = max(filtered, key=cv2.contourArea)
    hull = cv2.convexHull(stage_contour)
    hull_mask = np.zeros_like(mask_stage)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)

    if AUTO_ROI_DEBUG:
        cv2.imshow("Auto ROI - dark stage hull", hull_mask)
        cv2.waitKey(1)

    x, y, w, h = cv2.boundingRect(hull)
    x = max(0, x + shrink_margin)
    y = max(0, y + shrink_margin)
    w = max(1, min(color_image.shape[1] - x, w - 2 * shrink_margin))
    h = max(1, min(color_image.shape[0] - y, h - 2 * shrink_margin))
    return (x, y, w, h)


def _find_object_bbox_in_region(region_image, allow_non_dark_fallback=True):
    """Find the best object bounding box inside a local image region."""
    if region_image is None or region_image.size == 0:
        return None, 0.0

    hsv = cv2.cvtColor(region_image, cv2.COLOR_BGR2HSV)

    # Primary strategy: detect warmer tissue/red object colors.
    red1 = cv2.inRange(hsv, (0, 70, 45), (16, 255, 255))  # pyright: ignore[reportArgumentType, reportCallIssue]
    red2 = cv2.inRange(hsv, (170, 70, 45), (179, 255, 255)) # pyright: ignore[reportArgumentType, reportCallIssue]
    pink = cv2.inRange(hsv, (145, 35, 55), (179, 255, 255)) # pyright: ignore[reportArgumentType, reportCallIssue]
    b, g, r = cv2.split(region_image)
    tissue_bgr = (
        (r.astype("int16") - g.astype("int16") >= 12)
        & (r.astype("int16") - b.astype("int16") >= 8)
        & (r >= 55)
    ).astype("uint8") * 255

    mask_red = cv2.bitwise_or(cv2.bitwise_or(red1, red2), pink)
    mask_red = cv2.bitwise_or(mask_red, tissue_bgr)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)

    if AUTO_ROI_DEBUG:
        cv2.imshow("Auto ROI - red mask", mask_red)
        cv2.waitKey(1)

    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 120:
            continue
        x_o, y_o, w_o, h_o = cv2.boundingRect(contour)
        if h_o <= 0:
            continue
        aspect_ratio = w_o / float(h_o)
        if 0.25 <= aspect_ratio <= 4.0:
            # Ignore blobs glued to the search-region border; those are usually nozzle/highlight artifacts.
            touches_border = (
                x_o <= 2
                or y_o <= 2
                or (x_o + w_o) >= (region_image.shape[1] - 2)
                or (y_o + h_o) >= (region_image.shape[0] - 2)
            )
            if touches_border:
                continue
            red_candidates.append((area, x_o, y_o, w_o, h_o))

    if red_candidates:
        _, x_o, y_o, w_o, h_o = max(red_candidates, key=lambda item: item[0])
        return (x_o, y_o, w_o, h_o), 1.0

    if not allow_non_dark_fallback:
        return None, 0.0

    # Fallback: detect any sizable object that is brighter or more saturated than the dark stage.
    gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
    mask_dark_hsv = cv2.inRange(hsv, (0, 0, 0), (179, 115, 105))  # pyright: ignore[reportArgumentType, reportCallIssue]
    mask_dark_gray = (gray <= 105).astype(np.uint8) * 255
    mask_dark = cv2.bitwise_and(mask_dark_hsv, mask_dark_gray)
    mask_obj = cv2.bitwise_not(mask_dark)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_CLOSE, kernel, iterations=2)

    if AUTO_ROI_DEBUG:
        cv2.imshow("Auto ROI - non-dark fallback mask", mask_obj)
        cv2.waitKey(1)
    contours, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 300:
            continue
        x_o, y_o, w_o, h_o = cv2.boundingRect(contour)
        if h_o <= 0:
            continue
        aspect_ratio = w_o / float(h_o)
        extent = area / float(max(w_o * h_o, 1))
        if 0.20 < aspect_ratio < 5.0 and extent > 0.25:
            touches_border = (
                x_o <= 2
                or y_o <= 2
                or (x_o + w_o) >= (region_image.shape[1] - 2)
                or (y_o + h_o) >= (region_image.shape[0] - 2)
            )
            if touches_border:
                continue
            candidates.append((area, x_o, y_o, w_o, h_o))

    if not candidates:
        return None, 0.0

    _, x_o, y_o, w_o, h_o = max(candidates, key=lambda item: item[0])
    return (x_o, y_o, w_o, h_o), 0.75


def detect_object_on_dark_stage(color_image, stage_rect, fallback_to_region=True):
    """Detect the specimen sitting on top of the dark stage."""
    x_m, y_m, w_m, h_m = stage_rect
    roi = color_image[y_m:y_m + h_m, x_m:x_m + w_m]
    if roi.size == 0:
        return None
    obj_box, _score = _find_object_bbox_in_region(roi, allow_non_dark_fallback=True)
    if obj_box is None:
        return stage_rect if fallback_to_region else None

    x_o, y_o, w_o, h_o = obj_box

    # Pad slightly so the ROI includes the whole silhouette.
    pad = 6
    x_o = max(0, x_o - pad)
    y_o = max(0, y_o - pad)
    w_o = min(w_m - x_o, w_o + 2 * pad)
    h_o = min(h_m - y_o, h_o + 2 * pad)
    return (x_m + x_o, y_m + y_o, w_o, h_o)


def clamp_roi_to_frame(roi_box, frame_shape):
    """Clamp a ROI box so it stays within the image boundaries."""
    x, y, w, h = roi_box 
    frame_height, frame_width = frame_shape[:2]

    x = max(0, min(int(x), frame_width -1))
    y = max(0, min(int(y), frame_height -1))
    w = max (1, min(int(w), frame_width - x))
    h = max (1, min(int(h), frame_height - y))
    return (x, y, w, h)


def extract_roi(color_image, roi_box):
    """Return the cropped image patch defined by the ROI box."""
    x, y, w, h = clamp_roi_to_frame(roi_box, color_image.shape)
    return color_image[y:y + h, x:x + w]


def expand_roi(roi_box, frame_shape, margin=60):
    """Expand an ROI into a larger local search region."""
    x, y, w, h = clamp_roi_to_frame(roi_box, frame_shape)
    frame_height, frame_width = frame_shape[:2]
    search_x = max(0, x - margin)
    search_y = max(0, y - margin)
    search_w = min(frame_width - search_x, w + (2 * margin))
    search_h = min(frame_height - search_y, h + (2 * margin))
    return (search_x, search_y, search_w, search_h)


def track_roi_in_frame(color_image, roi_box, search_margin=60):
    """Update the ROI by re-detecting the object inside a local search region."""
    if color_image is None or roi_box is None:
        return roi_box, 0.0

    search_rect = expand_roi(roi_box, color_image.shape, margin=search_margin)
    detected_roi = detect_object_on_dark_stage(
        color_image,
        search_rect,
        fallback_to_region=False,
    )
    if detected_roi is None:
        return roi_box, 0.0

    # Detection-based tracking returns a binary confidence for now:
    # 1.0 means the object was re-detected inside the search region.
    return clamp_roi_to_frame(detected_roi, color_image.shape), 1.0


# Backward-compatible aliases for older call sites or local experiments.
find_mat = find_dark_stage
detect_object_on_blue_mat = detect_object_on_dark_stage

