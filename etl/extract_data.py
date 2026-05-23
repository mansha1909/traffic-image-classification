import os
import random

def extract_sample(dataset_path):

    print("===== SAMPLE FILES =====")

    train_path = os.path.join(dataset_path, "train")

    class_folder = random.choice(os.listdir(train_path))

    class_path = os.path.join(train_path, class_folder)

    files = os.listdir(class_path)

    sample = random.sample(files,5)

    for f in sample:
        print(f)