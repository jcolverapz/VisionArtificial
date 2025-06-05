import cv2
#import numpy as np
#import imutils

#cap = cv2.VideoCapture('autos.mp4')
cap = cv2.VideoCapture(0)

#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
car_counter = 0

while True:
    ret, frame = cap.read()
    if ret == False: break
    #frame = imutils.resize(frame, width=640)
    # Especificamos los puntos extremos del área a analizar
    #area_pts = np.array([[330, 216], [frame.shape[1]-80, 216], [frame.shape[1]-80, 271], [330, 271]])
    cv2.imshow('Frame', frame)
    k=cv2.waitKey(70) & 0xFF
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
    