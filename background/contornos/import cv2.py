import cv2
import numpy as np
import window_names
import track_bars

#vid = 'blackpool_tram_result.mp4'
vid = 'videos/vidrio20.mp4'

cap = cv2.VideoCapture(vid)

frame_counter = 0

while (True):
    ret, frame = cap.read()

    frame_counter += 1

    if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    blank = np.zeros(frame.shape[:2], dtype='uint8')

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    output = np.empty(grey.shape, dtype=np.uint8)

    cv2.normalize(
        grey,
        output,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX)

    hist = cv2.equalizeHist(output)

    track_bars.lower_threshold = cv2.getTrackbarPos("lower", window_names.window_canny)
    track_bars.upper_threshold = cv2.getTrackbarPos("upper", window_names.window_canny)
    track_bars.smoothing_neighbourhood = cv2.getTrackbarPos("smoothing", window_names.window_canny)
    track_bars.sobel_size = cv2.getTrackbarPos("sobel size", window_names.window_canny)

    track_bars.smoothing_neighbourhood = max(3, track_bars.smoothing_neighbourhood)
    if not (track_bars.smoothing_neighbourhood % 2):
        track_bars.smoothing_neighbourhood = track_bars.smoothing_neighbourhood + 1

    track_bars.sobel_size = max(3, track_bars.sobel_size)
    if not (track_bars.sobel_size % 2):
        track_bars.sobel_size = track_bars.sobel_size + 1

    smoothed = cv2.GaussianBlur(
        hist, (track_bars.smoothing_neighbourhood, track_bars.smoothing_neighbourhood), 0)

    edges = cv2.Canny(
        smoothed,
        track_bars.lower_threshold,
        track_bars.upper_threshold,
        apertureSize=track_bars.sobel_size)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    minLineLength = 50  # minimum number of pixels making up a line
    maxLineGap = 20  
    line_image = np.copy(frame) * 0

    mask = cv2.rectangle(blank, (edges.shape[1]//2 + 150, edges.shape[0]//2 - 150), (edges.shape[1]//2 - 150, edges.shape[0]//2 - 300), 255, -1)

    masked = cv2.bitwise_and(edges,edges,mask=mask)

    lines = cv2.HoughLinesP(masked, rho, theta, threshold, np.array([]), minLineLength, maxLineGap)

    if lines is not None:
        for x1, y1, x2, y2 in lines[0]:
            cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),5)

    lines_edges = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    cv2.imshow(window_names.window_hough, frame)
    cv2.imshow(window_names.window_canny, edges)
    cv2.imshow(window_names.window_mask, mask)
    cv2.imshow(window_names.window_masked_image, masked)

    key = cv2.waitKey(27)
    if (key == ord('x')) & 0xFF:
        break

cv2.destroyAllWindows()