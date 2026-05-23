import cv2
import matplotlib.pyplot as plt


def visualize_smoothing(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Error loading image")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ---------------------------------------
    # Apply Smoothing Techniques
    # ---------------------------------------

    gaussian = cv2.GaussianBlur(image, (5,5), 0)
    median = cv2.medianBlur(image, 5)

    gaussian_rgb = cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)
    median_rgb = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)

    # ---------------------------------------
    # Visualization
    # ---------------------------------------

    plt.figure(figsize=(12,6))

    plt.subplot(1,3,1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(gaussian_rgb)
    plt.title("Gaussian Smoothing")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(median_rgb)
    plt.title("Median Smoothing")
    plt.axis("off")

    plt.suptitle("Before vs After Smoothing")

    plt.show()