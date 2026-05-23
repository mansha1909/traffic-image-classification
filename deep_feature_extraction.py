import numpy as np
import cv2
import os
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model

base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_deep_features(dataset_path):

    # If features already exist, load them instantly
    if os.path.exists("deep_features.npz"):
        print("Loading precomputed deep features...")
        data = np.load("deep_features.npz")
        return data["X"], data["y"]

    print("Extracting deep features (first time only)...")

    X = []
    y = []

    train_path = os.path.join(dataset_path, "train")

    for label in os.listdir(train_path):

        class_path = os.path.join(train_path, label)

        for file in os.listdir(class_path):

            image_path = os.path.join(class_path, file)

            image = cv2.imread(image_path)

            if image is None:
                continue

            image = cv2.resize(image, (224,224))
            image = preprocess_input(image)
            image = np.expand_dims(image, axis=0)
            image = cv2.GaussianBlur(image, (5,5), 0)
            features = model.predict(image, verbose=0)

            X.append(features.flatten())
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Save features
    np.savez("deep_features.npz", X=X, y=y)

    print("Features saved to deep_features.npz")

    return X, y