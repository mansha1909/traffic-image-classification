import cv2
import os

def resize_sample_image(dataset_path):

    image_folder = os.path.join(dataset_path, "train", "accident")
    image_file = os.listdir(image_folder)[0]
    image_path = os.path.join(image_folder, image_file)

    image = cv2.imread(image_path)

    resized = cv2.resize(image, (224, 224))

    print("Original Shape:", image.shape)
    print("Resized Shape:", resized.shape)

    return resized