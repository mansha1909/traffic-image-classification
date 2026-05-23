import matplotlib.pyplot as plt


def fe_comparison_final():

    # ---------------------------------------
    # Methods
    # ---------------------------------------
    methods = [
        "HOG + LBP + Histogram\n(Feature Engineering)",
        "CNN Features (ResNet50)\n(Enhanced Feature Engineering)"
    ]

    accuracy = [0.71, 0.86]

    # ---------------------------------------
    # BAR GRAPH
    # ---------------------------------------
    plt.figure(figsize=(9,5))

    bars = plt.bar(methods, accuracy)

    plt.title("Feature Engineering vs Enhanced Feature Engineering")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)

    # Value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 height,
                 f"{height:.2f}",
                 ha='center',
                 va='bottom',
                 fontsize=12)

    plt.show()


  