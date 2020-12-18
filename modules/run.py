import cv2
import rough
image_path = "../data/input/1.jpg"
image = cv2.imread(image_path)

print(rough.generateCaption(image))
