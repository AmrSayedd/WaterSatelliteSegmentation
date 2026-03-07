from tensorflow.keras.models import load_model
from src.preprocessing_inference import preprocess_image_for_model
from src.model import build_unet_fusion
import matplotlib.pyplot as plt
import numpy as np


# ---------------- Model ----------------
model = build_unet_fusion(input_shape=(128,128,7))
model.load_weights(r"outputs/fusion3_unet_model.h5")

# ---------------- Preprocess new image ----------------
X_new , rgb_image = preprocess_image_for_model(r"data\images\149.tif")

# ---------------- Predict ----------------
pred_mask = model.predict(X_new)
pred_mask = (pred_mask > 0.5).astype(np.float32)

# ---------------- Visualize ----------------
fig, axes = plt.subplots(1, 2, figsize=(10,5))
# RGB image
axes[0].imshow(rgb_image)
axes[0].set_title("RGB Image")
axes[0].axis("off")

# Predicted mask
axes[1].imshow(pred_mask[0,:,:,0], cmap='Blues')
axes[1].set_title("Predicted Mask")
axes[1].axis("off")

plt.tight_layout()
plt.show()