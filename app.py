# # app.py
# from flask import Flask, request, render_template, redirect
# import os
# from resnet import predict_caption

# app = Flask(__name__)

# @app.route("/", methods=["GET", "POST"])
# def upload_image():
#     if request.method == "POST":
#         if 'file' not in request.files or request.files['file'].filename == '':
#             return redirect(request.url)

#         file = request.files['file']
#         img_path = 'input/' + file.filename
#         file.save(img_path)

#         caption = predict_caption(img_path)
#         os.remove(img_path)

#         return render_template('result.html', caption=caption)

#     return render_template('upload.html')

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=3000)

from flask import Flask, request, render_template, jsonify
import base64
import cv2
import numpy as np
from resnet import predict_caption

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "image_data" not in data:
        return jsonify({"error": "No image data"}), 400

    # Decode base64 image from JS (strip header)
    image_data = data["image_data"].split(",")[1]
    img_bytes = base64.b64decode(image_data)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, (1, 224, 224, 3))

    caption = predict_caption(img)
    return jsonify({"caption": caption})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)

