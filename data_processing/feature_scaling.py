import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def scale_features(X):

    print("\n===== FEATURE SCALING =====")

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    print("Original Feature Range:", np.min(X), "to", np.max(X))
    print("Scaled Feature Range:", np.min(X_scaled), "to", np.max(X_scaled))


    # Visualization of scaling effect
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(X[0])
    plt.title("Before Scaling")
    plt.xlabel("Feature Index")

    plt.subplot(1,2,2)
    plt.plot(X_scaled[0])
    plt.title("After Scaling")
    plt.xlabel("Feature Index")

    plt.tight_layout()
    plt.show()


    return X_scaled, scaler