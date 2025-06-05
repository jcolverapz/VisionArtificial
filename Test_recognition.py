import cv2
import face_recognition

# Imagen a comparar
image = cv2.imread("Images/Gaby.jpg")
face_loc = face_recognition.face_locations(image)[0]
#print("face_loc:", face_loc)
face_image_encodings = face_recognition.face_encodings(image, known_face_locations=[face_loc])[0]
print("face_image_encodings:", face_image_encodings)

cv2.rectangle(image, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (0, 255, 0))
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
