# -*- coding: utf-8 -*-
"""
Created on Sat May 16 10:33:10 2026

@author: M.Ravi prasath
"""

import cv2
import os
import time

# ==============================
# LOAD CASCADE CLASSIFIERS
# ==============================

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Eye detector
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# ==============================
# CREATE FOLDER TO SAVE FACES
# ==============================

save_folder = "saved_faces"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# ==============================
# OPEN WEBCAM
# ==============================

cap = cv2.VideoCapture(0)

# Check webcam access
if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

print("Webcam started successfully...")
print("Press 'q' to quit")

# ==============================
# SAVE TIMER
# ==============================

last_saved = time.time()

# ==============================
# MAIN LOOP
# ==============================

while True:

    # Read webcam frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ==============================
    # FACE DETECTION
    # ==============================

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Count total faces
    face_count = len(faces)

    # ==============================
    # PROCESS EACH FACE
    # ==============================

    for (x, y, w, h) in faces:

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Label face
        cv2.putText(
            frame,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # ==============================
        # ROI FOR EYE DETECTION
        # ==============================

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        # Draw eye rectangles
        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (255, 0, 0),
                2
            )

        # ==============================
        # SAVE FACE EVERY 5 SECONDS
        # ==============================

        current_time = time.time()

        if current_time - last_saved > 5:

            # Crop face
            face_crop = frame[y:y+h, x:x+w]

            # File name using timestamp
            filename = os.path.join(
                save_folder,
                f"face_{int(current_time)}.jpg"
            )

            # Save image
            cv2.imwrite(filename, face_crop)

            print(f"Saved: {filename}")

            last_saved = current_time

    # ==============================
    # DISPLAY TOTAL FACE COUNT
    # ==============================

    cv2.putText(
        frame,
        f'Total Faces: {face_count}',
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # ==============================
    # DISPLAY INSTRUCTIONS
    # ==============================

    cv2.putText(
        frame,
        "Press Q to Quit",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ==============================
    # SHOW OUTPUT WINDOW
    # ==============================

    cv2.imshow("Advanced Face Detection System", frame)

    # ==============================
    # EXIT KEY
    # ==============================

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# RELEASE RESOURCES
# ==============================

cap.release()
cv2.destroyAllWindows()

print("Program closed successfully.")