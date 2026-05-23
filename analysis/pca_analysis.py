import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def apply_pca(X_scaled, y_encoded):

    print("\n===== PCA ANALYSIS =====")

    pca = PCA(n_components=350)

    X_pca = pca.fit_transform(X_scaled)

    print("Original Feature Size:", X_scaled.shape)
    print("Reduced Feature Size:", X_pca.shape)

    print("Explained Variance Ratio:", pca.explained_variance_ratio_.sum())

    plt.figure(figsize=(6,5))

    scatter = plt.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=y_encoded,
        cmap="viridis",
        alpha=0.7
    )

    plt.title("PCA Feature Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.colorbar(scatter)

    plt.show()

    return X_pca, pca