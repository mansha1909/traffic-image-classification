import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
import numpy as np


def step5_vs_step6_visualization(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Error loading image")
        return

    image = cv2.resize(image, (224,224))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------
    # STEP 5: HOG (VISIBLE FEATURE)
    # ---------------------------------------
    hog_features, hog_image = hog(
        gray,
        orientations=8,
        pixels_per_cell=(16,16),
        cells_per_block=(2,2),
        visualize=True
    )

    # ---------------------------------------
    # STEP 6: CNN (SIMULATION - VECTOR)
    # ---------------------------------------
    # Just simulate deep feature vector (like ResNet output)
    deep_features = np.random.rand(2048)

    # ---------------------------------------
    # VISUALIZATION
    # ---------------------------------------
    plt.figure(figsize=(8,4))

    # Original Image
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Input Image")
    plt.axis("off")


    # Step 6 Output (Vector representation)
    plt.subplot(1,2,2)
    plt.plot(deep_features[:100])  # show only first 50
    plt.title("CNN Features (Vector)")
    plt.xlabel("Feature Index")
    plt.ylabel("Value")

    plt.suptitle("FEATURE EXTRACTION")

    plt.show()