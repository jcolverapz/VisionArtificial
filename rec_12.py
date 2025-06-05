import face_recognition
import cv2
import numpy as np

from PIL import Image, ImageDraw

video_capture = cv2.VideoCapture(0)

# Find all the faces and face encodings in the unknown image
image = face_recognition.load_image_file("known_faces/Julio.jpg")
image_face_encoding = face_recognition.face_encodings(image)[0]

known_face_encodings = [
    image_face_encoding
]
known_face_names = [
    "Julio",
]

while True:
    
    ret, frame =  video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
   #face_recognition.api.face_encodings(face_image, known_face_locations=None, num_jitters=1, model='small')[source]
    #face_locations = face_recognition.face_locations(frame, model="cnn")
    
    if face_locations != []:
        for face_location in face_locations:
               #face_encondings = face_recognition.face_encodings(rgb_frame, face_locations)       
            face_frame_encodings = face_recognition.face_encodings(frame, known_face_locations=[face_location])[0]
           
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_frame_encodings):  
            #face_frame_encodings = face_recognition.face_encodings(frame, known_face_locations =[face_location])[0]

                matches = face_recognition.compare_faces([known_face_encodings], face_frame_encodings)

                name="Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), cv2.FILLED)

           
         

            
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
    
video_capture.release()

cv2.destroyAllWindows()
