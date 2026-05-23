import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input
from deep_feature_extraction import model


def predict_image(image_path, model_rf, pca, scaler, encoder):

    print("\n===== IMAGE PREDICTION =====")

    image = cv2.imread(image_path)

    if image is None:
        print("Error loading image")
        return

    image_resized = cv2.resize(image,(224,224))

    image_processed = preprocess_input(image_resized)

    image_processed = np.expand_dims(image_processed, axis=0)

    # Extract deep features (2048)
    features = model.predict(image_processed, verbose=0)

    print("Extracted Deep Features:", features.shape)
   
    # Flatten
    features = features.reshape(1,-1)

    features_scaled = scaler.transform(features)

    features_pca = pca.transform(features_scaled)

    prediction = model_rf.predict(features_pca)

    label = encoder.inverse_transform(prediction)

    print("Predicted Class:", label[0])

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Prediction: " + label[0])
    plt.axis("off")
    plt.show()