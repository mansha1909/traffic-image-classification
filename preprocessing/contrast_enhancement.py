import cv2

def enhance_contrast(image):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    enhanced = clahe.apply(image)

    print("Contrast enhanced")

    return enhanced