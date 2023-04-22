import os
from PIL import Image
import matplotlib.pyplot as plt

def get_image_sizes(folder_path):
    image_sizes = []

    for file in os.listdir(folder_path):
        if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image = Image.open(os.path.join(folder_path, file))
            image_sizes.append(image.size)

    return image_sizes

def plot_image_height_width_distribution(folder_path):
    image_sizes = get_image_sizes(folder_path)

    if not image_sizes:
        print("No images found in the folder.")
        return

    widths, heights = zip(*image_sizes)

    plt.scatter(widths, heights)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Image Height and Width Distribution")
    plt.show()

folder_path = 'out\myhive_midbee\dataset\crop'
plot_image_height_width_distribution(folder_path)