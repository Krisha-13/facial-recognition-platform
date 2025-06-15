from fastapi import APIRouter, UploadFile, Form, HTTPException
import numpy as np
import cv2
import face_recognition
import os

router = APIRouter()
ENCODING_PATH = os.path.join(os.path.dirname(__file__), "known_faces.npy")

@router.post("/recognize-face")
async def recognize_face(file: UploadFile = Form(...)):
    try:
        # Read uploaded image
        content = await file.read()
        np_arr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Detect and encode face(s)
        locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, locations)

        if not encodings:
            return {"message": "❌ No face detected", "results": []}

        # Load known face data
        if not os.path.exists(ENCODING_PATH):
            return {"message": "❌ No registered faces", "results": []}

        known_data = np.load(ENCODING_PATH, allow_pickle=True).tolist()
        known_encodings = [np.array(item["encoding"]) for item in known_data]
        known_names = [item["name"] for item in known_data]

        recognized_names = []

        for face_encoding in encodings:
            # Calculate distances
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]

            # Match only if distance is below strict threshold
            if best_distance < 0.45:
                recognized_names.append(known_names[best_match_index])
            else:
                recognized_names.append("Unknown")

            # Debug log
            print(f"Distances: {face_distances}")
            print(f"Best match: {known_names[best_match_index]}, Distance: {best_distance}")

        return {"message": "✅ Recognized faces", "results": recognized_names}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
