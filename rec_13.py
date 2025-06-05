import face_recognition
import cv2
import numpy as np
import imutils
from datetime import datetime
import time

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.


# Load a sample picture and learn how to recognize it.
image_1 = face_recognition.load_image_file("known_faces/Julio.jpg")
face_encoding_1 = face_recognition.face_encodings(image_1)[0]

# Load a second sample picture and learn how to recognize it.
image_2 = face_recognition.load_image_file("known_faces/Joel.jpg")
face_encoding_2 = face_recognition.face_encodings(image_2)[0]

image_3 = face_recognition.load_image_file("known_faces/Joel_1.jpg")
face_encoding_3 = face_recognition.face_encodings(image_3)[0]


image_4 = face_recognition.load_image_file("known_faces/Jessie Pinkman.png")
face_encoding_4 = face_recognition.face_encodings(image_4)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    face_encoding_1,
    face_encoding_2,
    face_encoding_3,
    face_encoding_4
]
known_face_names = [
    "Cesar",
    "Joel",
    "Joel",
    "Jessie Pinkman"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []


def CheckStatus():
    
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    scale = 0.25
    process_this_frame = True
    
    TopLeft = [200, 100]
    TopRight = [400, 100]
    BotRight = [400, 350]
    BotLeft = [200, 350]
    area_pts = np.array([TopLeft, TopRight, BotRight , BotLeft])
    
    Datos = 'objects'
    band = False
    
    global face_locations
    global face_encodings
    global face_names
    
    # Configura el tiempo de captura (en segundos)
    tiempo_captura = 10

    # Obtiene el tiempo de inicio
    tiempo_inicio = time.time()

    # Bucle principal para capturar y mostrar los frames
    while (time.time() - tiempo_inicio) < tiempo_captura:
    #while band==False:
        # Grab a single frame of video
        ret, cap = video_capture.read()

        imAux = np.zeros(shape=(cap.shape[:2]), dtype=np.uint8)
        imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
        frame = cv2.bitwise_and(cap, cap, mask=imAux)
        
        cv2.drawContours(cap, [area_pts], -1, (0,255,0), 2)
        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    objeto = frame[40:480,150:600]
                    objeto = imutils.resize(objeto,width=400)
                    
                    band = True
                    #timeON = format(datetime.now(),"dd_MMM_yyyy_hh_mm")
                    now = datetime.now()
                    #OFF_formatted = now.strftime("%H:%M:%S")
                    timeON = now.strftime("%Y_%m_%d_%H_%M_%S")
                    print(timeON)
                    
                    
			        #cv2.imwrite(Datos+'/{}.jpg'.format(OFF_formattedImage ),objeto)
                    cv2.imwrite(Datos+'/{}.jpg'.format(name + "_" + str(timeON)),objeto)
                    
                face_names.append(name)

        process_this_frame = not process_this_frame

        if band == True:
        # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (0, 0, 0), 1)
               
        cv2.imshow('Video', frame)
                # Calculate the start time
                # start = time.time()
                # running = True
                # seconds = 0
                # end = 2

                # while (running):
                #     #time.sleep(1)
                #     seconds +=1
                # if seconds >= end:
                #     running = False
                #     break
                    # Release handle to the webcam

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("Sale de Video")
    video_capture.release()
    cv2.destroyAllWindows()

   
#from Guardar import *
def timer1():
    # Calculate the start time
    start = time.time()

    running = True
    seconds = 0
    end = 5
    print("Inicio")
    CheckStatus()

    while (running):
        time.sleep(1)
        seconds +=1
        print(seconds)
        
        if seconds >= end:
            running = False
            print("Timer_Tick")
            
            CheckStatus()
            
            running = True
            
timer1()