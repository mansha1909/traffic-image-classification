import cv2
import numpy as np
import os

def extract_features(dataset_path):

    X = []
    y = []

    train_path = os.path.join(dataset_path, "train")

    for label in os.listdir(train_path):

        class_path = os.path.join(train_path, label)

        for file in os.listdir(class_path):

            image_path = os.path.join(class_path, file)

            image = cv2.imread(image_path)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Histogram feature
            hist = cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()

            # Texture features
            mean_val = np.mean(gray)
            std_val = np.std(gray)

            features = np.concatenate([hist,[mean_val,std_val]])

            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    print("\nFeature Matrix Shape:", X.shape)
    print("Labels Shape:", y.shape)

    return X, y