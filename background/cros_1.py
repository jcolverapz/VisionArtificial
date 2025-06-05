import cv2
import numpy as np
import torchvision.ops.boxes as bops

def contour_intersect(cnt_ref, cnt_query):
    ## Contours are both an np array of points
    ## Check for bbox intersection, then check pixel intersection if bboxes intersect

    # first check if it is possible that any of the contours intersect
    x1, y1, w1, h1 = cv2.boundingRect(cnt_ref)
    x2, y2, w2, h2 = cv2.boundingRect(cnt_query)
    # get contour areas
    area_ref = cv2.contourArea(cnt_ref)
    area_query = cv2.contourArea(cnt_query)
    # get coordinates as tensors
    box1 = torch.tensor([[x1, y1, x1 + w1, y1 + h1]], dtype=torch.float)
    box2 = torch.tensor([[x2, y2, x2 + w2, y2 + h2]], dtype=torch.float)
    # get bbox iou
    iou = bops.box_iou(box1, box2)

    if iou == 0:
        # bboxes dont intersect, so contours dont either
        return False
    else:
        # bboxes intersect, now check pixels
        # get the height, width, x, and y of the smaller contour
        if area_ref >= area_query:
            h = h2
            w = w2
            x = x2
            y = y2
        else:
            h = h1
            w = w1
            x = x1
            y = y1

        # get a canvas to draw the small contour and subspace of the large contour
        contour_canvas_ref = np.zeros((h, w), dtype='uint8')
        contour_canvas_query = np.zeros((h, w), dtype='uint8')
        # draw the pixels areas, filled (can also be outline)
        cv2.drawContours(contour_canvas_ref, [cnt_ref], -1, 255, thickness=cv2.FILLED,
                         offset=(-x, -y))
        cv2.drawContours(contour_canvas_query, [cnt_query], -1, 255, thickness=cv2.FILLED,
                         offset=(-x, -y))

        # check for any pixel overlap
        return np.any(np.bitwise_and(contour_canvas_ref, contour_canvas_query))