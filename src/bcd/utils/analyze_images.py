
import os
from pathlib import Path
from typing import Tuple

from PIL import Image


def analyze_images(directory: Path) -> Tuple[list, list, list]:
    """
    Analyzes the images in a directory and returns their dimensions 
    and aspect ratios.

    Parameters
    ----------
    directory : Path
        The directory path containing the image files. The function 
        will recursively traverse the directory to find image files 
        with the extensions: .png, .jpg, .jpeg, .tiff, .bmp, and .gif.

    Returns
    -------
    Tuple[list, list, list]
        A tuple containing three lists:
        - widths: A list of the widths of the images found.
        - heights: A list of the heights of the images found.
        - aspects: A list of the aspect ratios (width / height) of the images found.
    """
    widths = []
    heights = []
    aspects = []

    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                img_path = os.path.join(subdir, file)
                with Image.open(img_path) as img:
                    width, height = img.size
                    widths.append(width)
                    heights.append(height)
                    aspects.append(width / height)

    return widths, heights, aspects
