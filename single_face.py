from mtcnn import MTCNN
import cv2
detector = MTCNN()

image = cv2.imread('images/single_face_man.jpg')

#image = cv2.imread('images/rabin.jpeg')

output = detector.detect_faces(image)

print(output)


x,y,w,h = output[0]['box']

left_eyeX, left_eyeY = output[0]['keypoints']['left_eye']
cv2.circle(image, center = (left_eyeX, left_eyeY), radius = 5, color = (255, 0, 0), thickness = 2)

right_eyeX, right_eyeY = output[0]['keypoints']['right_eye']
cv2.circle(image, center = (right_eyeX, right_eyeY), radius = 5, color = (255, 0, 0), thickness = 2)


noseX, noseY = output[0]['keypoints']['nose']
cv2.circle(image, center = (noseX, noseY), radius = 5, color = (255, 0, 0), thickness = 2)

mouth_leftX, mouth_leftY = output[0]['keypoints']['mouth_left']
cv2.circle(image, center = (mouth_leftX, mouth_leftY), radius = 5, color = (255, 0, 0), thickness = 2)

mouth_rightX, mouth_rightY = output[0]['keypoints']['mouth_right']
cv2.circle(image, center = (mouth_rightX, mouth_rightY), radius = 5, color = (255, 0, 0), thickness = 2)

cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
cv2.imshow('window', image)


cv2.waitKey(0)

print(output)