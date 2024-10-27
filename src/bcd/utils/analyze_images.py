
import os
from pathlib import Path
from typing import Tuple, List

from PIL import Image

from bcd.utils.paths import data_raw_dir


def analyze_images(img_paths: List[Path]) -> Tuple[list, list, list]:
    """
    Analyzes the images in a directory and returns their dimensions 
    and aspect ratios.

    Parameters
    ----------
    img_paths : Path
        Paths to the image files.

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

    for img_path in img_paths:
        with Image.open(data_raw_dir(img_path)) as img:
            width, height = img.size
            widths.append(width)
            heights.append(height)
            aspects.append(width / height)

    return widths, heights, aspects
