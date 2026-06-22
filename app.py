import os
import cv2
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
 
from roast import get_roast
from poster import build_poster
 
# CONFIG 
 
OUTPUT_DIR      = "output"
FACE_PHOTO_PATH = os.path.join(OUTPUT_DIR, "captured_face.jpg")
POSTER_PATH     = os.path.join(OUTPUT_DIR, "wanted_poster.jpg")
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
app = Flask(__name__)
 
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
 
# HELPERS
 
def get_largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])
 
 
def grab_frame():
    return cap.read()
 
 
def generate_feed():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
 
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Suspect", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
 
        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
 
 
def safe_crop(frame, x, y, w, h):
    """
    Crop face with padding, but ALWAYS clamp to frame bounds
    and verify the resulting crop has valid dimensions.
    Returns cropped image or None if something is wrong.
    """
    fh, fw = frame.shape[:2]   # frame height, width
 
    mx = int(w * 0.45)
    my = int(h * 0.65)
 
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(fw, x + w + mx)
    y2 = min(fh, y + h + my)
 
    # Guard: crop must have positive width AND height
    if x2 <= x1 or y2 <= y1:
        # Fallback: use the raw detection box with no margin
        x1, y1, x2, y2 = x, y, x + w, y + h
 
    # Final clamp just in case
    x1 = max(0, min(x1, fw - 1))
    x2 = max(x1 + 1, min(x2, fw))
    y1 = max(0, min(y1, fh - 1))
    y2 = max(y1 + 1, min(y2, fh))
 
    cropped = frame[y1:y2, x1:x2]
 
    if cropped.size == 0:
        return None
 
    return cropped
 
#ROUTES 
 
@app.route("/")
def index():
    return render_template("index.html")
 
 
@app.route("/video_feed")
def video_feed():
    return Response(
        generate_feed(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )
 
 
@app.route("/capture", methods=["POST"])
def capture():
    try:
        data    = request.get_json()
        use_api = data.get("use_api", False)
        api_key = data.get("api_key", "")
 
        ret, frame = grab_frame()
        if not ret:
            return jsonify({"error": "Could not read from webcam"}), 500
 
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
    app.run(host="127.0.0.1", port=5000, debug=False)
 