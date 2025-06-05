import cv2
from moviepy.editor import VideoFileClip
from rembg import remove

video =  VideoFileClip('videos/vidrio51.mp4')
BG_COLOR = (0,255, 0, 255)

output_filename  = "videos/result.mp4"
output_fps = 30

def remove_background(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = remove(frame_rgb, bgcolor=BG_COLOR)
    
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

final =  video.fl_image(remove_background)

final.write_videofile(output_filename, fps=output_fps)
    
