import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def detect_outliers(X, y):

    print("\n===== OUTLIER DETECTION =====")

    # ---------------------------------------
    # Compute Z-scores
    # ---------------------------------------
    z_scores = np.abs(stats.zscore(X))

    threshold = 3

    # remove samples having too many outlier features
    mask = (z_scores < threshold).mean(axis=1) > 0.95
    X_clean = X[mask]
    y_clean = y[mask]

    print("Original Samples:", X.shape[0])
    print("Samples After Removing Outliers:", X_clean.shape[0])
    print("Outliers Removed:", X.shape[0] - X_clean.shape[0])

    # ---------------------------------------
    # Visualization 1: Scatter Plot with Outliers
    # ---------------------------------------

    feature_index = 0  # visualize first feature
    feature = X[:, feature_index]

    outliers = z_scores[:, feature_index] > threshold
    normal = z_scores[:, feature_index] <= threshold

    plt.figure(figsize=(8,5))

    # normal points
    plt.scatter(np.where(normal)[0], feature[normal],
                label="Normal Data", alpha=0.6)

    # outlier points (red circles)
    plt.scatter(np.where(outliers)[0], feature[outliers],
                facecolors='none',
                edgecolors='red',
                s=120,
                linewidths=2,
                label="Outliers")

    # regression line
    z = np.polyfit(np.arange(len(feature)), feature, 1)
    p = np.poly1d(z)

    plt.plot(np.arange(len(feature)),
             p(np.arange(len(feature))),
             color="black",
             label="Trend Line")

    plt.title("Scatter Plot Showing Outliers")
    plt.xlabel("Sample Index")
    plt.ylabel("Feature Value")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()

    # ---------------------------------------
    # Visualization 2: Z-score Histogram
    # ---------------------------------------

    plt.figure(figsize=(8,5))

    plt.hist(z_scores.flatten(), bins=50)

    plt.axvline(3, color='red', linestyle='--', label='Outlier Threshold')
    plt.axvline(-3, color='red', linestyle='--')

    plt.title("Z-Score Distribution for Outlier Detection")
    plt.xlabel("Z-Score")
    plt.ylabel("Frequency")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()

    return X_clean, y_clean