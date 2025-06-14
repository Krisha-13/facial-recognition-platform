import cv2
import face_recognition
import numpy as np
import os
import base64
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
KNOWN_FACE_PATH = os.path.join(BASE_DIR, "known_faces.npy")

def register_face(name: str, image_base64: str) -> str:
    # Decode base64 image
    image_data = base64.b64decode(image_base64.split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    face_locations = face_recognition.face_locations(image)
    if len(face_locations) != 1:
        return "Please make sure exactly one face is visible"

    face_encoding = face_recognition.face_encodings(image, face_locations)[0]

    # Load or create known_faces
    if os.path.exists(KNOWN_FACE_PATH):
        known_faces = np.load(KNOWN_FACE_PATH, allow_pickle=True).item()
    else:
        known_faces = {"encodings": [], "names": []}

    known_faces["encodings"].append(face_encoding)
    known_faces["names"].append(f"{name} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

    np.save(KNOWN_FACE_PATH, known_faces)
    return f"Face registered for {name}"
