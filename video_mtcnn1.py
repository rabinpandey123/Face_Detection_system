import cv2
from mtcnn import MTCNN

# Open webcam
cap = cv2.VideoCapture(0)

detector = MTCNN()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output = detector.detect_faces(frame)

    for single_output in output:
        x, y, w, h = single_output['box']
        cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)

        # Draw landmarks (use frame, not cap/image)
        left_eyeX, left_eyeY = single_output['keypoints']['left_eye']
        cv2.circle(frame, center=(left_eyeX, left_eyeY), radius=5, color=(0, 255, 0), thickness=2)

        right_eyeX, right_eyeY = single_output['keypoints']['right_eye']
        cv2.circle(frame, center=(right_eyeX, right_eyeY), radius=5, color=(0, 255, 0), thickness=2)

        noseX, noseY = single_output['keypoints']['nose']
        cv2.circle(frame, center=(noseX, noseY), radius=5, color=(0, 0, 255), thickness=2)

        mouth_leftX, mouth_leftY = single_output['keypoints']['mouth_left']
        cv2.circle(frame, center=(mouth_leftX, mouth_leftY), radius=5, color=(255, 255, 0), thickness=2)

        mouth_rightX, mouth_rightY = single_output['keypoints']['mouth_right']
        cv2.circle(frame, center=(mouth_rightX, mouth_rightY), radius=5, color=(255, 255, 0), thickness=2)

    cv2.imshow('Face Detection', frame)

    # Press 'x' to quit
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
