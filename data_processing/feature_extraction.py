import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, hog


# -------------------------------------------------
# FEATURE VISUALIZATION (for random image)
# -------------------------------------------------
def visualize_features(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Error loading image for visualization")
        return

    image = cv2.resize(image, (224,224))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray,100,200)

    # HOG feature visualization
    hog_features, hog_image = hog(
        gray,
        orientations=8,
        pixels_per_cell=(16,16),
        cells_per_block=(2,2),
        visualize=True
    )

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(edges,cmap="gray")
    plt.title("Edge Feature")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(hog_image,cmap="gray")
    plt.title("HOG Feature")
    plt.axis("off")

    plt.tight_layout()
    plt.show()



# -------------------------------------------------
# FEATURE EXTRACTION (for dataset training)
# -------------------------------------------------
def extract_features(dataset_path):

    X = []
    y = []

    train_path = os.path.join(dataset_path, "train")

    count = 0

    for label in os.listdir(train_path):

        class_path = os.path.join(train_path, label)

        for file in os.listdir(class_path):

            count += 1

            if count % 100 == 0:
                print("Processed images:", count)

            image_path = os.path.join(class_path, file)

            image = cv2.imread(image_path)

            if image is None:
                continue

            # Resize image
            image = cv2.resize(image, (224,224))

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # ---------------------------------
            # Histogram Feature
            # ---------------------------------
            hist = cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()

            # ---------------------------------
            # Edge Feature
            # ---------------------------------
            edges = cv2.Canny(gray,100,200)
            edge_density = np.sum(edges > 0) / edges.size

            # ---------------------------------
            # Texture Feature (LBP)
            # ---------------------------------
            lbp = local_binary_pattern(gray, 24, 3, method="uniform")
            lbp_hist,_ = np.histogram(lbp.ravel(), bins=26, range=(0,26))

            # ---------------------------------
            # Color Histogram
            # ---------------------------------
            color_hist = cv2.calcHist(
                [image],
                [0,1,2],
                None,
                [16,16,16],
                [0,256,0,256,0,256]
            ).flatten()

            # ---------------------------------
            # HOG Feature
            # ---------------------------------
            hog_features = hog(
                gray,
                orientations=8,
                pixels_per_cell=(16,16),
                cells_per_block=(2,2),
                visualize=False
            )

            # ---------------------------------
            # Statistical Features
            # ---------------------------------
            mean_val = np.mean(gray)
            std_val = np.std(gray)

            # ---------------------------------
            # Combine All Features
            # ---------------------------------
            features = np.concatenate([
                hist,
                lbp_hist,
                color_hist,
                hog_features,
                [edge_density],
                [mean_val],
                [std_val]
            ])

            X.append(features)
            y.append(label)


    X = np.array(X)
    y = np.array(y)
    print("Training feature length:", len(features))
    print("\nFeature Matrix Shape:", X.shape)
    print("Labels Shape:", y.shape)

    return X, y

def extract_features_single(image):

    import cv2
    import numpy as np
    from skimage.feature import local_binary_pattern, hog

    image = cv2.resize(image, (224,224))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()

    edges = cv2.Canny(gray,100,200)
    edge_density = np.sum(edges > 0) / edges.size

    lbp = local_binary_pattern(gray, 24, 3, method="uniform")
    lbp_hist,_ = np.histogram(lbp.ravel(), bins=26, range=(0,26))

    color_hist = cv2.calcHist(
        [image],[0,1,2],None,
        [16,16,16],
        [0,256,0,256,0,256]
    ).flatten()

    hog_features = hog(
        gray,
        orientations=8,
        pixels_per_cell=(16,16),
        cells_per_block=(2,2),
        visualize=False
    )

    mean_val = np.mean(gray)
    std_val = np.std(gray)

    features = np.concatenate([
        hist,
        lbp_hist,
        color_hist,
        hog_features,
        [edge_density],
        [mean_val],
        [std_val]
    ])
    print("Extracted Features:", features.shape)


    return features.reshape(1,-1)
