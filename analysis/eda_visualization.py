import os
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew


# --------------------------------------------------
# 1. SAMPLE IMAGE VISUALIZATION
# --------------------------------------------------

def visualize_sample_images(dataset_path):

    train_path = os.path.join(dataset_path, "train")

    plt.figure(figsize=(10,8))

    i = 1

    for label in os.listdir(train_path):

        class_path = os.path.join(train_path, label)

        img_name = random.choice(os.listdir(class_path))

        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(2,2,i)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")

        i += 1

    plt.suptitle("Sample Images From Each Class")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 2. IMAGE SIZE DISTRIBUTION
# --------------------------------------------------

def visualize_image_sizes(dataset_path):

    train_path = os.path.join(dataset_path, "train")

    widths = []
    heights = []

    for label in os.listdir(train_path):

        class_path = os.path.join(train_path, label)

        for file in os.listdir(class_path)[:50]:

            img_path = os.path.join(class_path, file)

            img = cv2.imread(img_path)

            if img is None:
                continue

            h, w = img.shape[:2]

            heights.append(h)
            widths.append(w)

    plt.figure(figsize=(7,5))

    plt.scatter(widths, heights, alpha=0.6)

    plt.title("Image Size Distribution")
    plt.xlabel("Width")
    plt.ylabel("Height")

    plt.grid(True)

    plt.show()


# --------------------------------------------------
# 3. PIXEL INTENSITY DISTRIBUTION
# --------------------------------------------------

def visualize_pixel_distribution(dataset_path):

    train_path = os.path.join(dataset_path, "train")

    pixels = []

    for label in os.listdir(train_path):

        class_path = os.path.join(train_path, label)

        for file in os.listdir(class_path)[:20]:

            img_path = os.path.join(class_path, file)

            img = cv2.imread(img_path)

            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            pixels.extend(gray.flatten())

    plt.figure(figsize=(7,5))

    plt.hist(pixels, bins=50)

    plt.title("Pixel Intensity Distribution")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()


# --------------------------------------------------
# 4. SKEWNESS VISUALIZATION (MAIN FOCUS)
# --------------------------------------------------

def visualize_skewness(dataset_path):

    train_path = os.path.join(dataset_path, "train")

    pixels = []

    for label in os.listdir(train_path):

        class_path = os.path.join(train_path, label)

        for file in os.listdir(class_path)[:20]:

            img_path = os.path.join(class_path, file)

            img = cv2.imread(img_path)

            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            pixels.extend(gray.flatten())

    pixels = np.array(pixels)

    skew_value = skew(pixels)

    plt.figure(figsize=(7,5))

    plt.hist(pixels, bins=50, color="orange")

    # mean line
    plt.axvline(np.mean(pixels), linestyle='--', label="Mean")

    plt.title(f"Pixel Distribution (Skewness = {round(skew_value,3)})")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()


# --------------------------------------------------
# MAIN EDA FUNCTION
# --------------------------------------------------

def run_eda(dataset_path):

    print("\n===== EDA VISUALIZATION =====")

    visualize_sample_images(dataset_path)

    visualize_image_sizes(dataset_path)

    visualize_pixel_distribution(dataset_path)

    visualize_skewness(dataset_path)