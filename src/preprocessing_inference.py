import rasterio
import numpy as np
import json

json_file = r"outputs\preprocessing2.json"

def preprocess_image_for_model(tif_path, preprocessing_file=json_file):
    """
    Preprocess a single image for inference using the stored preprocessing parameters.
    WaterProb channel is included as the last channel.
    
    Returns:
        img : np.array of shape (1, H, W, C) ready for model input
        img_rgb : np.array of shape (H, W, 3) for visualization (RGB)
    """
    
    # ---------------- Load preprocessing parameters ----------------
    with open(preprocessing_file, "r") as f:
        prep = json.load(f)

    selected_channels = prep["selected_channels"]
    global_mins = np.array(prep["global_mins"], dtype=np.float32)
    global_maxs = np.array(prep["global_maxs"], dtype=np.float32)

    # ---------------- Load image ----------------
    with rasterio.open(tif_path) as src:
        # RGB visualization (BGR -> RGB)
        img_rgb = src.read()[[3, 2, 1]].astype(np.float32)
        for c in range(3):
            band = img_rgb[c]
            img_rgb[c] = (band - band.min()) / (band.max() - band.min())
        img_rgb = np.transpose(img_rgb, (1, 2, 0))  # (H, W, 3)

        # Load selected channels for model input
        img = src.read()[selected_channels, :, :].astype(np.float32)  # (C, H, W)

    # ---------------- Normalize channels ----------------
    for c in range(len(selected_channels)):
        min_val = global_mins[c]
        max_val = global_maxs[c]
        if max_val > min_val:
            img[c] = (img[c] - min_val) / (max_val - min_val)
        else:
            img[c] = 0  # fallback if min=max

    # ---------------- Final formatting ----------------
    img = np.transpose(img, (1, 2, 0))        # (H, W, C)
    img = np.expand_dims(img, axis=0).astype(np.float32)  # add batch dim: (1, H, W, C)

    return img, img_rgb