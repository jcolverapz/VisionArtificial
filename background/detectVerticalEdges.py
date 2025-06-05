import cv2
import numpy as np
#import serial
import time
import threading

#ser = serial.Serial('COM4',bytesize=8, baudrate=9600, timeout=1)  # Replace 'COM3' with the appropriate serial port
time.sleep(3)

cap = cv2.VideoCapture('videos/vidrio20.mp4')
circle_detected = False

flag = False

def callback():
    global flag
    flag = True

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20,
                            param1=60, param2=34, minRadius=10, maxRadius=50)

    # Draw circles around detected centers and set flag if circle is detected
    if circles is not None:
        circle_detected = True
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 0, 255), 2)

    if circle_detected:
        if threading.active_count() <= 1:
            timer = threading.Timer(7.0, callback)
            timer.start()
        if circle_detected and flag:
            flag = False
           # ser.write(b'00000001')
            print("Sent data to serial port")
        circle_detected = False


    # Display the resulting frame
    cv2.imshow('Sugarcane Buds Detection', frame)

    # Exit program when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

    # Flag signal if circle is detected
    if circle_detected:
        print("Circle detected!")
    else:
        print("No circle detected.")