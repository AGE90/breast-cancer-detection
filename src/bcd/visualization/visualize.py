
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from bcd.utils.paths import data_raw_dir

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

def display_sample_images(images_paths, num_samples=5):
    
    sample_images = np.random.choice(images_paths, num_samples, replace=False)

    plt.figure(figsize=(15, 3*num_samples))
    for i, img_path in enumerate(sample_images):
        img = Image.open(data_raw_dir(img_path))
        plt.subplot(num_samples, 1, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f'Image: {img_path}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()