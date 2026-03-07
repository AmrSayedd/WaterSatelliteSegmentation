from flask import Flask, request, render_template_string
import numpy as np
import cv2
import io
import base64

from src.flask_inference import model
from src.flask_preprocessing import preprocess_image_for_model
from PIL import Image
app = Flask(__name__)

# ---------------- Home page ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    html = """
    <!doctype html>
    <html lang="en">
    <head>
        <title>Satellite Segmentation</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { padding: 30px; background-color: #f8f9fa; }
            .img-container { text-align: center; margin-top: 10px; }
            .img-container img { width: 100%; max-height: 600px; object-fit: contain; border: 1px solid #ddd; }
            h5 { margin-top: 15px; margin-bottom: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center mb-4">Satellite Segmentation</h1>
            <form method="post" enctype="multipart/form-data" class="text-center mb-4">
                <input type="file" name="image" accept=".tif" required>
                <button type="submit" class="btn btn-primary">Predict</button>
            </form>

            {% if rgb and mask %}
            <div class="row">
                <div class="col-md-6 img-container">
                    <h5>Input Image</h5>
                    <img src="data:image/png;base64,{{ rgb }}" alt="RGB">
                </div>
                <div class="col-md-6 img-container">
                    <h5>Prediction Mask</h5>
                    <img src="data:image/png;base64,{{ mask }}" alt="Mask">
                </div>
            </div>
            {% endif %}
        </div>
    </body>
    </html>
    """

    rgb_b64 = None
    mask_b64 = None

    if request.method == "POST":
        file = request.files["image"]
        file_bytes = file.read()

        # ---------------- Preprocess ----------------
        X, rgb = preprocess_image_for_model(file_bytes)  # rgb: H,W,3 float [0-1]

        # ---------------- Predict ----------------
        pred_mask = model.predict(X)
        mask = (pred_mask[0,:,:,0] > 0.5).astype(np.uint8)  # 0 or 1

        # ---------------- RGB visualization ----------------
        rgb_display = (rgb * 255).astype(np.uint8)  # scale to 0-255
        rgb_img = Image.fromarray(rgb_display)
        rgb_bytes = io.BytesIO()
        rgb_img.save(rgb_bytes, format="PNG")
        rgb_b64 = base64.b64encode(rgb_bytes.getvalue()).decode('utf-8')

        # ---------------- Mask visualization ----------------
        # Invert mask: 1->0 (black), 0->1 (white)
        mask_inv = 1 - mask  # now water is white, background black
        mask_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        mask_color[mask_inv==1] = [255,255,255]  # white background
        mask_color[mask_inv==0] = [0,0,200]      # blue for water

        mask_img = Image.fromarray(mask_color)
        mask_bytes = io.BytesIO()
        mask_img.save(mask_bytes, format="PNG")
        mask_b64 = base64.b64encode(mask_bytes.getvalue()).decode('utf-8')

    return render_template_string(html, rgb=rgb_b64, mask=mask_b64)

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=5000)