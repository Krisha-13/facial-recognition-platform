import os
import io  # ✅ REQUIRED
import base64
import numpy as np
from PIL import Image
import face_recognition

BASE_DIR = os.path.dirname(__file__)
KNOWN_FACE_PATH = os.path.join(BASE_DIR, "known_faces.npy")

def recognize_face(image_base64):
    try:
        # Decode image
        header, encoded = image_base64.split(",", 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data))  # ✅ uses io.BytesIO
        rgb_image = np.array(image.convert("RGB"))

        # Detect and encode face
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            return {"error": "No face detected."}

        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        # Load known faces
        if not os.path.exists(KNOWN_FACE_PATH):
            return {"error": "No registered faces available."}

        known_faces = np.load(KNOWN_FACE_PATH, allow_pickle=True).tolist()
        known_names = [name for name, _ in known_faces]
        known_encodings = [encoding for _, encoding in known_faces]

        # Compare faces
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
            if True in matches:
                match_index = matches.index(True)
                return {"match": known_names[match_index]}

        return {"match": None}
    except Exception as e:
        return {"error": f"Recognition failed: {str(e)}"}
