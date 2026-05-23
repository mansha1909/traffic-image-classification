from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_model(X_pca, y_encoded):

    print("\n===== MODEL TRAINING =====")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca,
        y_encoded,
        test_size=0.2,
        random_state=42
    )

    print("Training Samples:", X_train.shape[0])
    print("Testing Samples:", X_test.shape[0])

    # Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    print("Model training completed")

    return model, X_test, y_test