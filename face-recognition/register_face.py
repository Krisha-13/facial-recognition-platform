import os
import cv2
import base64
import face_recognition
import numpy as np
from datetime import datetime

# === Paths ===
BASE_DIR = os.path.dirname(__file__)
KNOWN_FACE_PATH = os.path.join(BASE_DIR, "known_faces.npy")

def register_face(name: str, image_base64: str) -> str:
    try:
        # Step 1: Decode base64 image
        image_data = base64.b64decode(image_base64.split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Step 2: Detect face
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) != 1:
            return "❌ Please show exactly one face."

        # Step 3: Encode face
        encoding = face_recognition.face_encodings(image, face_locations)[0]

        # Step 4: Load known faces
        if os.path.exists(KNOWN_FACE_PATH):
            faces = np.load(KNOWN_FACE_PATH, allow_pickle=True).tolist()
            if not isinstance(faces, list):
                faces = []
        else:
            faces = []

        # Step 5: Update existing face if match found
        updated = False
        for i, face in enumerate(faces):
            existing_encoding = np.array(face["encoding"])
            distance = face_recognition.face_distance([existing_encoding], encoding)[0]
            if distance < 0.45:
                # Match found – update name and timestamp
                faces[i] = {
                    "name": name,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "encoding": encoding.tolist()
                }
                updated = True
                break

        # Step 6: If no match found, add new entry
        if not updated:
            faces.append({
                "name": name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "encoding": encoding.tolist()
            })

        # Step 7: Save back to .npy
        np.save(KNOWN_FACE_PATH, np.array(faces, dtype=object))
        return f"✅ Face registered for {name}"

    except Exception as e:
        return f"❌ Error: {str(e)}"
