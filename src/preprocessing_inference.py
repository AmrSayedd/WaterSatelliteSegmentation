import rasterio
import numpy as np
import json
json_file = "D:\Amr\VScode\Computer Vision\SatelliteSegmentation\outputs\preprocessing.json"
def preprocess_image_for_model(tif_path, preprocessing_file=json_file):
    # Load preprocessing parameters
    with open(preprocessing_file, "r") as f:
        prep = json.load(f)

    selected_channels = prep["selected_channels"]
    global_mins = np.array(prep["global_mins"], dtype=np.float32)
    global_maxs = np.array(prep["global_maxs"], dtype=np.float32)

    # Load image
    with rasterio.open(tif_path) as src:
        img_rgb = src.read()[[3,2,1]].astype(np.float32)
        for c in range(3):
            band = img_rgb[c]
            img_rgb[c] = (band - band.min()) / (band.max() - band.min())
        img_rgb = np.transpose(img_rgb,(1,2,0))

        img = src.read()[selected_channels, :, :].astype(np.float32)  # (C,H,W)

    # Normalize each channel
    for c in range(len(selected_channels)):
        min_val = global_mins[c]
        max_val = global_maxs[c]
        if max_val > min_val:
            img[c] = (img[c] - min_val) / (max_val - min_val)
        else:
            img[c] = 0

    # Compute NDWI
    green = img[1]
    nir = img[3]
    ndwi = (green - nir) / (green + nir + 1e-6)
    ndwi = (ndwi + 1) / 2  # scale 0-1
    ndwi = np.expand_dims(ndwi, axis=0)

    # Append NDWI
    img = np.concatenate([img, ndwi], axis=0)

    # (H,W,C)
    img = np.transpose(img, (1,2,0))
    # Add batch dimension
    img = np.expand_dims(img, axis=0).astype(np.float32)

    return img , img_rgb