import os
import cv2
from mtcnn import MTCNN

detector = MTCNN()

# Get absolute path based on script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(BASE_DIR,"images", "rabin.jpeg")

image = cv2.imread(img_path)

if image is None:
    raise FileNotFoundError(f"⚠️ Could not load image at {img_path}")

results = detector.detect_faces(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
print(results)
