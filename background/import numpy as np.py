import numpy as np
from PIL import ImageGrab
import cv2


def process_image4(original_image):  # Douglas-peucker approximation
    # Convert to black and white threshold map
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    (thresh, bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert bw image back to colored so that red, green and blue contour lines are visible, draw contours
    modified_image = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(modified_image, contours, -1, (255, 0, 0), 3)

    # Contour approximation
    try:  # Just to be sure it doesn't crash while testing!
        for cnt in contours:
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            # cv2.drawContours(modified_image, [approx], -1, (0, 0, 255), 3)
    except:
        pass
    return modified_image


def screen_record():
    while(True):
        screen = np.array(ImageGrab.grab(bbox=(100, 240, 750, 600)))
        image = process_image4(screen)
        cv2.imshow('window', image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

screen_record()