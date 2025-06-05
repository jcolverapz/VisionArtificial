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
            result = face_recognition.compare_faces([image_face_encoding], face_frame_encodings)
            print("Result:", result)

        if result[0] == True:
          text = "Julio"
          color = (125, 220, 0)
        else:
          text = "Desconocido"
          color = (50, 50, 255)

    cv2.rectangle(frame, (face_location[3], face_location[2]), (face_location[1], face_location[2] + 30), color, -1)
    cv2.rectangle(frame, (face_location[3], face_location[0]), (face_location[1], face_location[2]), color, 2)
    cv2.putText(frame, text, (face_location[3], face_location[2] + 20), 2, 0.7, (255, 255, 255), 1)

            
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
    
video_capture.release()

cv2.destroyAllWindows()
