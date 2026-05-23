import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_labels(y):

    print("\n===== LABEL ENCODING STEP =====")

    # ---------- Visualization BEFORE encoding ----------
    label_counts = pd.Series(y).value_counts()

    plt.figure(figsize=(6,4))
    label_counts.plot(kind="bar", color="skyblue")

    plt.title("Class Distribution (Before Encoding)")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")

    plt.tight_layout()
    plt.show()


    # ---------- Encoding ----------
    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)


    # ---------- Label Mapping ----------
    print("\nLabel Mapping:")
    for i, label in enumerate(encoder.classes_):
        print(label, "→", i)


    # ---------- Visualization AFTER encoding ----------
    encoded_counts = pd.Series(y_encoded).value_counts()

    plt.figure(figsize=(6,4))
    encoded_counts.plot(kind="bar", color="orange")

    plt.title("Encoded Label Distribution")
    plt.xlabel("Encoded Class")
    plt.ylabel("Number of Images")

    plt.tight_layout()
    plt.show()


    print("\nFirst 10 encoded labels:", y_encoded[:10])

    return y_encoded, encoder