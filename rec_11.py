import face_recognition
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
           
image = face_recognition.load_image_file('known_faces/Joel.jpg')
#image = face_recognition.load_image_file("Reconocimiento_facial/Gaby.jpg")
image_face_encoding = face_recognition.face_encodings(image)[0]  
            
video_capture = cv2.VideoCapture(0)
# Find all the faces and face encodings in the unknown image

image1 = face_recognition.load_image_file("known_faces/Julio.jpg")
image_face_encoding1 = face_recognition.face_encodings(image)[0]

image2 = face_recognition.load_image_file("known_faces/Joel.jpg")
image_face_encoding2 = face_recognition.face_encodings(image)[0]

known_face_encodings = [
    image_face_encoding1,
    image_face_encoding2
]
known_face_names = [
    "Julio",
    "Joel"
]
 

while True:
    
    ret, frame =  video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    #face_frame_encodings = face_recognition.face_encodings(frame, face_frame_locations)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    #print(len(face_locations))
    #if face_locations != []:
        
        #for face_location in face_locations:
        
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):  
            
        matches = face_recognition.compare_faces(known_face_encodings, face_encodings)
            
        name="Unknown"
            
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), cv2.FILLED)
                
        # if result[0] == True:
        #   text = "Julio"
        #   color = (125, 220, 0)
        # else:
        #   text = "Desconocido"
        #   color = (50, 50, 255)

        # cv2.rectangle(frame, (face_location[3], face_location[0]), (face_location[1], face_location[2]), color, 2)
        # cv2.putText(frame, text, (face_location[3], face_location[2] + 20), 2, 0.7, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
    
video_capture.release()

cv2.destroyAllWindows()
  
# # Returns (R, G, B) from name
# def name_to_color(name):
#     # Take 3 first letters, tolower()
#     # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
#     color = [(ord(c.lower())-97)*8 for c in name[:3]]
#     return color


# print('Loading known faces...')
# known_faces = []
# known_names = []


def Busca(obj):

    KNOWN_FACES_DIR = 'known_faces'
    UNKNOWN_FACES_DIR = 'unknown_faces'
    TOLERANCE = 0.6
    FRAME_THICKNESS = 3
    FONT_THICKNESS = 2
    MODEL = 'cnn'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model
    # We oranize known faces as subfolders of KNOWN_FACES_DIR
    # Each subfolder's name becomes our label (name)
    for name in os.listdir(KNOWN_FACES_DIR):

        # Next we load every file of faces of known person
        for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):

            # Load an image
            image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
            #image = face_recognition.load_image_file("Reconocimiento_facial/Gaby.jpg")
            image_face_encoding = face_recognition.face_encodings(image)[0]

            # known_face_encodings = [
            # image_face_encoding
            # ]
            # known_face_names = [
            # "Julio",
            # ]
            result = face_recognition.compare_faces([image_face_encoding], face_frame_encodings)
            
            print("Result:", result)
           # if result==True:
