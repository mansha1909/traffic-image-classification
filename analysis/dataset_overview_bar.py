import os
import matplotlib.pyplot as plt


def dataset_overview_bar(dataset_path):

    train_path = os.path.join(dataset_path, "train")

    labels = []
    counts = []

    for label in os.listdir(train_path):
        class_path = os.path.join(train_path, label)

        labels.append(label)
        counts.append(len(os.listdir(class_path)))

    # 🎨 subtle professional colors
    colors = ["#6BAED6", "#FD8D3C", "#74C476", "#FB6A4A"]

    plt.figure(figsize=(8,5))

    bars = plt.bar(labels, counts, color=colors)

    plt.title("Dataset Distribution (TrafficNet)", fontsize=14)
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")

    # add values on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 height,
                 str(height),
                 ha='center',
                 va='bottom')

    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.show()