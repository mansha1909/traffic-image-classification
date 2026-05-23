from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_model(X_pca, y_encoded):

    print("\n===== MODEL TRAINING =====")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca,
        y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42
    )

    print("Training Samples:", X_train.shape[0])
    print("Testing Samples:", X_test.shape[0])

    # Improved Random Forest model
    model = RandomForestClassifier(
        n_estimators=1200,
        max_depth=50,
        random_state=42,
        n_jobs=-1
    )

    # Train model
    model.fit(X_train, y_train)

    print("Model training completed")

    return model, X_test, y_test