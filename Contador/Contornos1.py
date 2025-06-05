import cv2
import numpy as np
import imutils
#from matplotlib import pyplot as plt
import time

# Path to video  
#video_path="videos/bicycle1.mp4" 
video_path="videos/video30.mp4" 
#video = cv2.VideoCapture(video_path)
#video = cv2.VideoCapture(0)
############################ Algorithm ####################################

# Read video
cap = cv2.VideoCapture('videos/vidrio30.mp4')

# Take first frame and find corners in it
ret, old_frame = cap.read()

width = old_frame.shape[1]
height = old_frame.shape[0]

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

frame_count = 0
start_time = time.time()
first_gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)

old_gray = first_gray

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(7, 7),  # Window size
                 maxLevel=2,  # Number of pyramid levels
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

