from src.flask_preprocessing import preprocess_image_for_model
from src.model import build_unet_fusion
import numpy as np

# ---------------- Load Model Once ----------------

model = build_unet_fusion(input_shape=(128,128,7))

model.load_weights(
    r"outputs/fusion3_unet_model.h5"
)

print("Model loaded successfully.")


# ---------------- Inference Function ----------------

def predict_image(file_bytes):
    """
    Runs segmentation inference on an uploaded .tif image.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of the uploaded TIFF image.

    Returns
    -------
    rgb_image : np.ndarray
        RGB visualization of the satellite image (H,W,3)

    pred_mask : np.ndarray
        Binary segmentation mask (H,W)
    """

    # ---------- Preprocess ----------
    X_new, rgb_image = preprocess_image_for_model(file_bytes)

    # ---------- Predict ----------
    pred_mask = model.predict(X_new)

    # ---------- Threshold ----------
    pred_mask = (pred_mask > 0.5).astype(np.float32)

    # remove batch + channel
    pred_mask = pred_mask[0, :, :, 0]

    return rgb_image, pred_mask