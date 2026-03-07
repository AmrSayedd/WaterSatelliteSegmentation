import rasterio
from rasterio.io import MemoryFile
import numpy as np
import json

json_file = "outputs/preprocessing2.json"

def preprocess_image_for_model(file_bytes, preprocessing_file=json_file):
    """
    Preprocess an in-memory TIF for inference.

    - img      : normalized using global min/max (ready for model)
    - img_rgb  : normalized per-channel for viewing purposes only

    Returns:
        img      : np.array of shape (1, H, W, C) ready for model input
        img_rgb  : np.array of shape (H, W, 3) for visualization
    """

    # ---------------- Load preprocessing parameters ----------------
    with open(preprocessing_file, "r") as f:
        prep = json.load(f)

    selected_channels = prep["selected_channels"]
    global_mins = np.array(prep["global_mins"], dtype=np.float32)
    global_maxs = np.array(prep["global_maxs"], dtype=np.float32)

    # ---------------- Load image from memory ----------------
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            # ---------------- RGB for visualization ----------------
            img_rgb = src.read()[[3, 2, 1]].astype(np.float32)  # BGR -> RGB
            for c in range(3):
                band = img_rgb[c]
                min_val = band.min()
                max_val = band.max()
                if max_val > min_val:
                    img_rgb[c] = (band - min_val) / (max_val - min_val)
                else:
                    img_rgb[c] = 0
            img_rgb = np.transpose(img_rgb, (1, 2, 0))  # (H, W, 3)

            # ---------------- Model input ----------------
            img = src.read()[selected_channels, :, :].astype(np.float32)  # (C, H, W)

    # ---------------- Normalize channels using global min/max ----------------
    for c in range(len(selected_channels)):
        min_val = global_mins[c]
        max_val = global_maxs[c]
        if max_val > min_val:
            img[c] = (img[c] - min_val) / (max_val - min_val)
        else:
            img[c] = 0

    # ---------------- Final formatting ----------------
    img = np.transpose(img, (1, 2, 0))               # (H, W, C)
    img = np.expand_dims(img, axis=0).astype(np.float32)  # (1, H, W, C)

    return img, img_rgb