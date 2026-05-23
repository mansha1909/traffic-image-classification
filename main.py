import os
import random

from etl.dataset_overview import dataset_overview
from analysis.dataset_overview_bar import dataset_overview_bar
from analysis.eda_visualization import run_eda
from etl.extract_data import extract_sample
from preprocessing.preprocessing_pipeline import preprocessing_pipeline
from analysis.smoothing_visualization import visualize_smoothing
from deep_feature_extraction import extract_deep_features
from data_processing.feature_extraction import visualize_features
from analysis.step5_vs_step6_visualization import step5_vs_step6_visualization
from data_processing.encoding_labels import encode_labels
from data_processing.feature_scaling import scale_features
from data_processing.outlier_detection import detect_outliers
from analysis.pca_analysis import apply_pca
from model.train_model import train_model
from model.model_evaluation import evaluate_model
from model.predict_image import predict_image
from analysis.fe_comparison_final import fe_comparison_final


dataset_path = "dataset/trafficnet_dataset_v1"


# ---------------------------------
# STEP 0 : SELECT RANDOM IMAGE
# ---------------------------------

test_root = os.path.join(dataset_path, "test")

random_class = random.choice(os.listdir(test_root))

class_path = os.path.join(test_root, random_class)

test_image = os.path.join(class_path, random.choice(os.listdir(class_path)))

print("\nSelected Image For Pipeline:", test_image)


print("\n==============================")
print("STEP 1 : DATASET OVERVIEW")
print("==============================")
dataset_overview(dataset_path)
dataset_overview_bar(dataset_path)


print("\n==============================")
print("STEP 2 : EDA VISUALIZATION")
print("==============================")

run_eda(dataset_path)
# pass selected image
preprocessing_pipeline(test_image)
print("\n==============================")
print("STEP 3 : SMOOTHING VISUALIZATION")
print("==============================")

visualize_smoothing(test_image)

print("\n==============================")
print("STEP 4 : PREPROCESSING COMPARISON")
print("==============================")


print("\n✅ PREPROCESSING PIPELINE COMPLETED")
print("\n==============================")
print("STEP 5 : FEATURE VISUALIZATION")
print("==============================")

print("Visualization skipped (using deep features)")
print("\n==============================")
print("STEP 6 : FEATURE EXTRACTION")
print("==============================")
X, y = extract_deep_features(dataset_path)
step5_vs_step6_visualization(test_image)

print("\n==============================")
print("STEP 7 : LABEL ENCODING")
print("==============================")
y_encoded, encoder = encode_labels(y)

print("\n==============================")
print("STEP 8 : FEATURE SCALING")
print("==============================")
X_scaled, scaler = scale_features(X)


print("\n==============================")
print("STEP 9 : OUTLIER DETECTION")
print("==============================")
X_clean, y_clean = detect_outliers(X_scaled, y_encoded)

print("\n==============================")
print("STEP 10 : PCA ANALYSIS")
print("==============================")
X_pca, pca = apply_pca(X_clean, y_clean)


print("\n==============================")
print("STEP 11 : MODEL TRAINING")
print("==============================")
model, X_test, y_test = train_model(X_pca, y_clean)


print("\n==============================")
print("STEP 12 : MODEL EVALUATION")
print("==============================")
accuracy = evaluate_model(model, X_test, y_test)


print("\n==============================")
print("STEP 13 : IMAGE PREDICTION")
print("==============================")

print("Testing Image:", test_image)

predict_image(test_image, model, pca, scaler, encoder)

print("\n==============================")
print("FINAL STEP : FEATURE ENGINEERING COMPARISON")
print("==============================")

fe_comparison_final()
