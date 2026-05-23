import os

def dataset_overview(dataset_path = "dataset/trafficnet_dataset_v1"):

    train_path = os.path.join(dataset_path,"train")
    test_path = os.path.join(dataset_path,"test")

    print("===== DATASET OVERVIEW =====")

    print("Train folders:", os.listdir(train_path))
    print("Test folders:", os.listdir(test_path))

    total_images = 0

    for folder in os.listdir(train_path):
        path = os.path.join(train_path,folder)
        count=len(os.listdir(path))
        print(f"Train {folder}: {count}")
        total_images += count

    for folder in os.listdir(test_path):
        path = os.path.join(test_path,folder)
        count=len(os.listdir(path))
        print(f"Test {folder}: {count}")
        total_images += count

    print("Total Images:", total_images)