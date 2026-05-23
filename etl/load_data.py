import cv2
import os
import matplotlib.pyplot as plt

def load_sample_image(dataset_path):

    image_path=os.path.join(dataset_path,"train","accident")

    file=os.listdir(image_path)[0]

    img_path=os.path.join(image_path,file)

    image=cv2.imread(img_path)

    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.title("Sample Accident Image")
    plt.axis("off")
    plt.show()