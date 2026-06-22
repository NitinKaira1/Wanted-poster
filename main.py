import cv2
import os
from roast import get_roast
from poster import build_poster

# ─── CONFIG ───────────────────────────────────────────────────────────────────

USE_API  = False          # Set True to use Claude API for roasts
API_KEY  = ""             # Paste your Anthropic API key here if USE_API = True

OUTPUT_DIR       = "output"
FACE_PHOTO_PATH  = os.path.join(OUTPUT_DIR, "captured_face.jpg")
POSTER_PATH      = os.path.join(OUTPUT_DIR, "wanted_poster.jpg")

# SETUP 

os.makedirs(OUTPUT_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# HELPERS 

def draw_ui(frame, faces, captured):
    """Draw detection boxes + on-screen instructions onto the frame."""
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Suspect identified", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    status = "CAPTURED — check output/ folder" if captured else "SPACE = capture  |  Q = quit"
    color  = (0, 200, 0) if captured else (0, 220, 255)
    cv2.putText(frame, status, (12, frame.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame


def get_largest_face(faces):
    """Return the (x,y,w,h) of the biggest detected face, or None."""
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def crop_face(frame, x, y, w, h):
    """Crop face with generous padding so the poster photo isn't too tight."""
    mx = int(w * 0.45)
    my = int(h * 0.65)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(frame.shape[1], x + w + mx)
    y2 = min(frame.shape[0], y + h + my)
    return frame[y1:y2, x1:x2]

# MAIN LOOP

print("=" * 50)
print("  WANTED POSTER CAM")
print("  Look at camera → press SPACE to capture")
print("  Press Q to quit without capturing")
print("=" * 50)

captured = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    frame = draw_ui(frame, faces, captured)
    cv2.imshow("WANTED POSTER CAM — Press SPACE to capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord(' ') and not captured:
        face = get_largest_face(faces)
        if face is None:
            print("No face detected — move closer or improve lighting.")
            continue

        x, y, w, h = face
        cropped = crop_face(frame, x, y, w, h)
        cv2.imwrite(FACE_PHOTO_PATH, cropped)
        print(f"[main] Face saved → {FACE_PHOTO_PATH}")

        print("[main] Generating roast...")
        roast = get_roast(use_api=USE_API, api_key=API_KEY if USE_API else None)
        print(f"[main] Roast: {roast}")

        print("[main] Building poster...")
        build_poster(FACE_PHOTO_PATH, roast, POSTER_PATH)

        print(f"\n✅ YOUR WANTED POSTER IS READY: {POSTER_PATH}\n")
        captured = True   # update UI overlay but keep window open so user sees confirmation

cap.release()
cv2.destroyAllWindows()

if captured:
    # Auto-open the poster on Windows
    os.startfile(os.path.abspath(POSTER_PATH))
