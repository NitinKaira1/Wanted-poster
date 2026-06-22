import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, Response, jsonify, request, send_from_directory

from roast import get_roast
from poster import build_poster

# CONFIG

OUTPUT_DIR      = "output"
FACE_PHOTO_PATH = os.path.join(OUTPUT_DIR, "captured_face.jpg")
POSTER_PATH     = os.path.join(OUTPUT_DIR, "wanted_poster.jpg")

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# HELPERS

def get_largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def safe_crop(frame, x, y, w, h):
    fh, fw = frame.shape[:2]

    mx = int(w * 0.45)
    my = int(h * 0.65)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(fw, x + w + mx)
    y2 = min(fh, y + h + my)

    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = x, y, x + w, y + h

    x1 = max(0, min(x1, fw - 1))
    x2 = max(x1 + 1, min(x2, fw))
    y1 = max(0, min(y1, fh - 1))
    y2 = max(y1 + 1, min(y2, fh))

    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return None
    return cropped


def decode_image(data_url):
    """Decode a base64 data URL (from browser canvas) into an OpenCV frame."""
    # Strip the data:image/...;base64, prefix
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(data_url)
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


# ROUTES

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/capture", methods=["POST"])
def capture():
    try:
        data      = request.get_json()
        use_api   = data.get("use_api", False)
        api_key   = data.get("api_key", "")
        image_b64 = data.get("image", "")

        if not image_b64:
            return jsonify({"error": "No image received from browser"}), 400

        frame = decode_image(image_b64)
        if frame is None:
            return jsonify({"error": "Could not decode image from browser"}), 400

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        face = get_largest_face(faces)
        if face is None:
            return jsonify({"error": "No face detected — move closer or improve lighting!"}), 400

        x, y, w, h = face
        cropped = safe_crop(frame, x, y, w, h)

        if cropped is None:
            return jsonify({"error": "Face crop failed — try moving slightly and capturing again"}), 400

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cv2.imwrite(FACE_PHOTO_PATH, cropped)

        roast = get_roast(use_api=use_api, api_key=api_key if use_api else None)
        build_poster(FACE_PHOTO_PATH, roast, POSTER_PATH)

        return jsonify(roast)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/output/<path:filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)


# RUN

if __name__ == "__main__":
    print("=" * 50)
    print("  WANTED POSTER CAM  —  Flask GUI")
    print("  Open browser:  http://127.0.0.1:5000")
    print("  Ctrl+C to stop")
    print("=" * 50)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
