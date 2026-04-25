import pyrealsense2 as rs
import numpy as np
import cv2

# =========================
# RealSense settings
# =========================
COLOR_W   = 640
COLOR_H   = 480
COLOR_FPS = 30

# =========================
# 1. HSV color filter parameters (tennis ball yellow-green)
# =========================
HSV_LOWER = np.array([25,  80,  80])   # H/S/V lower bound
HSV_UPPER = np.array([85, 255, 255])   # H/S/V upper bound
MORPH_K   = 5                          # morphology kernel size (for denoising)

# =========================
# 2. Hough circle detection parameters
# =========================
HOUGH_DP        = 1.2   # accumulator resolution (smaller = finer, slower)
HOUGH_MIN_DIST  = 30    # minimum distance between circle centers (pixels)
HOUGH_PARAM1    = 80    # Canny high threshold
HOUGH_PARAM2    = 18    # accumulator threshold (smaller = more detections, more noise)
HOUGH_MIN_R     = 8     # minimum radius (pixels)
HOUGH_MAX_R     = 60    # maximum radius (pixels)

# =========================
# 3. Background subtraction parameters
# =========================
BG_HISTORY      = 200   # number of frames for background modeling
BG_VAR_THRESH   = 40    # pixel variance threshold (larger = less sensitive)
BG_DETECT_SHADE = False # whether to detect shadows

# Window names
WIN_ORIG   = "Original"
WIN_STEP1  = "1. HSV Color Filter"
WIN_STEP12 = "1+2. HSV + Circle Detection"
WIN_STEP123 = "1+2+3. HSV + Circle + Background Subtraction"


def apply_hsv_filter(color_image):
    """
    Step 1: HSV color filter.
    Returns: mask (single-channel), masked_image (color, non-ball areas blacked out)
    """
    hsv  = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    # Morphological denoising: erode to remove small noise, then dilate to restore ball shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    masked_image = cv2.bitwise_and(color_image, color_image, mask=mask)
    return mask, masked_image


def apply_hough_circle(color_image, hsv_mask):
    """
    Step 2: Hough circle detection on top of the HSV mask.
    Returns: annotated_image (circles drawn on the original image)
    """
    out = color_image.copy()

    # Detect circles only within the color region (mask -> grayscale -> blur)
    gray = cv2.GaussianBlur(hsv_mask, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_R,
        maxRadius=HOUGH_MAX_R
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for (cx, cy, r) in circles:
            cv2.circle(out, (cx, cy), r,  (0, 255, 0), 2)   # green circle outline
            cv2.circle(out, (cx, cy), 4,  (0, 0, 255), -1)  # red circle center
            cv2.putText(out, f"r={r}", (cx + r + 4, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    return out, circles


def apply_bg_subtraction(bg_subtractor, color_image, hsv_mask, circles):
    """
    Step 3: Background subtraction keep only detections inside motion regions.
    Returns: annotated_image
    """
    out = color_image.copy()

    # Background subtraction foreground mask
    fg_mask = bg_subtractor.apply(color_image)

    # Remove shadows (gray value 127 regions) and small noise
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Triple mask: color AND motion
    combined_mask = cv2.bitwise_and(hsv_mask, fg_mask)

    # Overlay foreground region on the image (semi-transparent blue)
    fg_overlay        = np.zeros_like(color_image)
    fg_overlay[combined_mask > 0] = (180, 60, 0)
    out = cv2.addWeighted(out, 0.7, fg_overlay, 0.3, 0)

    # Keep only Hough circles that fall within motion regions
    if circles is not None:
        for (cx, cy, r) in circles:
            roi = combined_mask[max(0, cy-r):cy+r, max(0, cx-r):cx+r]
            if roi.size > 0 and np.sum(roi > 0) > 0.2 * roi.size:
                # Enough pixels in the motion region -> confirmed as a valid ball
                cv2.circle(out, (cx, cy), r,  (0, 255, 0), 3)
                cv2.circle(out, (cx, cy), 4,  (0, 0, 255), -1)
                cv2.putText(out, f"BALL r={r}", (cx + r + 4, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                # Circle in static region -> mark gray (possible false positive)
                cv2.circle(out, (cx, cy), r,  (100, 100, 100), 1)

    return out, fg_mask


def draw_label(image, text):
    """Draw a stage label in the top-left corner of the image."""
    cv2.rectangle(image, (0, 0), (len(text) * 11 + 10, 28), (0, 0, 0), -1)
    cv2.putText(image, text, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main():
    # -------------------------
    # Start RealSense
    # -------------------------
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
    pipeline.start(config)

    # Background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY,
        varThreshold=BG_VAR_THRESH,
        detectShadows=BG_DETECT_SHADE
    )

    # Create and arrange windows
    for win in [WIN_ORIG, WIN_STEP1, WIN_STEP12, WIN_STEP123]:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, COLOR_W, COLOR_H)

    cv2.moveWindow(WIN_ORIG,    0,              0)
    cv2.moveWindow(WIN_STEP1,   COLOR_W + 10,   0)
    cv2.moveWindow(WIN_STEP12,  0,              COLOR_H + 50)
    cv2.moveWindow(WIN_STEP123, COLOR_W + 10,   COLOR_H + 50)

    print("Running, press Q to quit")
    print(f"HSV range: H[{HSV_LOWER[0]}-{HSV_UPPER[0]}] S[{HSV_LOWER[1]}-{HSV_UPPER[1]}] V[{HSV_LOWER[2]}-{HSV_UPPER[2]}]")

    try:
        while True:
            frames      = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # Step 1: HSV color filter
            hsv_mask, step1_img = apply_hsv_filter(color_image)

            # Step 1 + 2: Hough circle detection
            step12_img, circles = apply_hough_circle(color_image.copy(), hsv_mask)

            # Step 1 + 2 + 3: Background subtraction filter
            step123_img, fg_mask = apply_bg_subtraction(
                bg_subtractor, color_image.copy(), hsv_mask, circles
            )

            # Apply labels
            draw_label(color_image.copy(), "Original")
            draw_label(step1_img,   "1. HSV Color Filter")
            draw_label(step12_img,  "1+2. HSV + Circle Detection")
            draw_label(step123_img, "1+2+3. HSV + Circle + Background Subtraction")

            # Print current detection results to terminal
            n_circles = len(circles) if circles is not None else 0
            print(f"\rCircles detected: {n_circles}  (press Q to quit)", end="", flush=True)

            cv2.imshow(WIN_ORIG,    color_image)
            cv2.imshow(WIN_STEP1,   step1_img)
            cv2.imshow(WIN_STEP12,  step12_img)
            cv2.imshow(WIN_STEP123, step123_img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\nExited")


if __name__ == "__main__":
    main()
