
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_size_distribution(sizes, title):
    plt.figure(figsize=(8, 4))
    plt.hist(sizes, bins=50)
    plt.title(f'{title} Distribution')
    plt.xlabel('Pixels')
    plt.ylabel('Frequency')
    plt.show()

def plot_aspect_ratio_distribution(aspects):
    plt.figure(figsize=(8, 4))
    plt.hist(aspects, bins=50)
    plt.title('Aspect Ratio Distribution')
    plt.xlabel('Aspect Ratio (width/height)')
    plt.ylabel('Frequency')
    plt.show()

def display_sample_images(directory, num_samples=5):
    all_images = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                all_images.append(os.path.join(subdir, file))

    sample_images = np.random.choice(all_images, num_samples, replace=False)

    plt.figure(figsize=(15, 3*num_samples))
    for i, img_path in enumerate(sample_images):
        img = Image.open(img_path)
        plt.subplot(num_samples, 1, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f'Image {i+1}: {img.size[0]}x{img.size[1]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()