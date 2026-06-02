import cv2
import time
import os
from datetime import datetime
import face_recognition

# =========================================
# LOGIN SYSTEM
# =========================================

username = "admin"
password = "1234"

if username != "admin" or password != "1234":
    print("Access Denied")
    exit()

print("Login Successful")

# =========================================
# LOAD FACE DETECTOR
# =========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

# =========================================
# LOAD KNOWN FACE
# =========================================

known_image = face_recognition.load_image_file(
    "images/sujatha.jpg"
)

known_encoding = face_recognition.face_encodings(
    known_image
)[0]

known_faces = [known_encoding]

known_names = ["Sujatha"]

# =========================================
# START WEBCAM
# =========================================

cap = cv2.VideoCapture(0)

# =========================================
# CREATE SCREENSHOT FOLDER
# =========================================

if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# =========================================
# FPS VARIABLES
# =========================================

prev_time = 0

print("Press:")
print("Q -> Quit")
print("S -> Save Screenshot")

# =========================================
# MAIN LOOP
# =========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to gray
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # =========================================
    # FACE DETECTION
    # =========================================

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # =========================================
    # FACE RECOGNITION
    # =========================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    face_locations = face_recognition.face_locations(
        rgb_frame
    )

    face_encodings = face_recognition.face_encodings(
        rgb_frame,
        face_locations
    )

    for (top, right, bottom, left), face_encoding in zip(
        face_locations,
        face_encodings
    ):

        matches = face_recognition.compare_faces(
            known_faces,
            face_encoding
        )

        name = "Unknown"

        if True in matches:
            matched_index = matches.index(True)
            name = known_names[matched_index]

        # =========================================
        # DRAW RECTANGLE
        # =========================================

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            3
        )

        # Corner lines
        line_length = 25

        # Top Left
        cv2.line(
            frame,
            (left, top),
            (left + line_length, top),
            (255, 0, 255),
            3
        )

        cv2.line(
            frame,
            (left, top),
            (left, top + line_length),
            (255, 0, 255),
            3
        )

        # Top Right
        cv2.line(
            frame,
            (right, top),
            (right - line_length, top),
            (255, 0, 255),
            3
        )

        cv2.line(
            frame,
            (right, top),
            (right, top + line_length),
            (255, 0, 255),
            3
        )

        # Bottom Left
        cv2.line(
            frame,
            (left, bottom),
            (left + line_length, bottom),
            (255, 0, 255),
            3
        )

        cv2.line(
            frame,
            (left, bottom),
            (left, bottom - line_length),
            (255, 0, 255),
            3
        )

        # Bottom Right
        cv2.line(
            frame,
            (right, bottom),
            (right - line_length, bottom),
            (255, 0, 255),
            3
        )

        cv2.line(
            frame,
            (right, bottom),
            (right, bottom - line_length),
            (255, 0, 255),
            3
        )

        # =========================================
        # NAME DISPLAY
        # =========================================

        cv2.putText(
            frame,
            name,
            (left, top - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        # =========================================
        # SCANNING LINE
        # =========================================

        scan_y = top + int(
            (time.time() * 200) %
            (bottom - top)
        )

        cv2.line(
            frame,
            (left, scan_y),
            (right, scan_y),
            (0, 255, 255),
            2
        )

    # =========================================
    # FACE COUNT
    # =========================================

    face_count = len(face_locations)

    cv2.putText(
        frame,
        f"Faces Detected: {face_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        3
    )

    # =========================================
    # FPS COUNTER
    # =========================================

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3
    )

    # =========================================
    # DATE & TIME
    # =========================================

    current_datetime = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    cv2.putText(
        frame,
        current_datetime,
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 255),
        2
    )

    # =========================================
    # TITLE
    # =========================================

    cv2.putText(
        frame,
        "AI FACE RECOGNITION SECURITY SYSTEM",
        (60, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    # =========================================
    # SHOW WINDOW
    # =========================================

    cv2.imshow(
        "AI Face Recognition System",
        frame
    )

    # =========================================
    # KEYBOARD CONTROLS
    # =========================================

    key = cv2.waitKey(1)

    # Quit
    if key == ord('q'):
        break

    # Save Screenshot
    elif key == ord('s'):

        filename = datetime.now().strftime(
            "screenshots/face_%Y%m%d_%H%M%S.jpg"
        )

        cv2.imwrite(filename, frame)

        print(f"Screenshot Saved: {filename}")

# =========================================
# RELEASE EVERYTHING
# =========================================

cap.release()
cv2.destroyAllWindows()