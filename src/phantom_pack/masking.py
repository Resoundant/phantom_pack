import cv2
import numpy as np

import cv2
import numpy as np

def create_mask_using_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Create a mask using CLAHE.

    Args:
        image (numpy array): Input image.
        clip_limit (float, optional): Clip limit for CLAHE. Defaults to 2.0.
        tile_grid_size (tuple, optional): Tile grid size for CLAHE. Defaults to (8, 8).

    Returns:
        numpy array: Mask image.
    """
    # Apply CLAHE to the image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_image = clahe.apply(image)

    # Create a mask by thresholding the CLAHE image
    # auto_thresh, mask = cv2.threshold(clahe_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    auto_thresh, mask = cv2.threshold(clahe_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    norm_mask = (mask - mask.min()) / (mask.max() - mask.min())
    norm_mask = norm_mask.astype(image.dtype)
    return norm_mask

import cv2
import numpy as np

def fill_voids(image:np.ndarray, kernel=3, iterations=10):
    """
    Fill in voids inside an image using morphological operations.

    Args:
        image (numpy array): Input image.

    Returns:
        numpy array: Image with voids filled.
    """
    # Create a kernel for morphological operations
    kernel = np.ones((kernel, kernel), image.dtype)

    # Apply morphological closing to fill in voids
    filled_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    return filled_image
