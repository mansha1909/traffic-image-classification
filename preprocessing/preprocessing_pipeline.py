import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocessing_pipeline(image_path):

    print("Selected Image:", image_path)

    image = cv2.imread(image_path)

    if image is None:
        print("Error loading image")
        return

    # Step 1 - Original
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 2 - Resize
    resized = cv2.resize(original, (224,224))

    # Step 3 - Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    # Step 4 - Noise removal
    denoised = cv2.GaussianBlur(gray,(5,5),0)

    # Step 5 - Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)

    # Step 6 - Edge detection
    edges = cv2.Canny(enhanced,100,200)

    # Step 7 - Sharpening
    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])

    sharpened = cv2.filter2D(enhanced,-1,kernel)

    # Visualization
    plt.figure(figsize=(12,6))

    plt.subplot(2,4,1)
    plt.imshow(original)
    plt.title("1 Original")
    plt.axis("off")

    plt.subplot(2,4,2)
    plt.imshow(resized)
    plt.title("2 Resize")
    plt.axis("off")

    plt.subplot(2,4,3)
    plt.imshow(gray,cmap="gray")
    plt.title("3 Grayscale")
    plt.axis("off")

    plt.subplot(2,4,4)
    plt.imshow(denoised,cmap="gray")
    plt.title("4 Noise Removal")
    plt.axis("off")

    plt.subplot(2,4,5)
    plt.imshow(enhanced,cmap="gray")
    plt.title("5 Contrast")
    plt.axis("off")

    plt.subplot(2,4,6)
    plt.imshow(edges,cmap="gray")
    plt.title("6 Edge Detection")
    plt.axis("off")

    plt.subplot(2,4,7)
    plt.imshow(sharpened,cmap="gray")
    plt.title("7 Sharpened")
    plt.axis("off")

    plt.tight_layout()
    plt.show()