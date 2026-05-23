import cv2

def remove_noise(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised = cv2.GaussianBlur(gray, (5,5), 0)

    print("Noise removed successfully")

    return denoised