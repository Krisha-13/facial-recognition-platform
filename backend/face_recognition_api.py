# backend/face_recognition_api.py

from fastapi import APIRouter, UploadFile, Form, HTTPException
import os
import cv2
import numpy as np
import face_recognition

router = APIRouter()

REGISTERED_DIR = os.path.join(os.path.dirname(__file__), "registered")

@router.post("/recognize-face")
async def recognize_face(file: UploadFile = Form(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image uploaded.")

        # Encode incoming face
        unknown_encodings = face_recognition.face_encodings(frame)
        if not unknown_encodings:
            return {"message": "No face found in the image."}

        unknown_encoding = unknown_encodings[0]

        # Loop through registered images
        for filename in os.listdir(REGISTERED_DIR):
            filepath = os.path.join(REGISTERED_DIR, filename)
            known_image = face_recognition.load_image_file(filepath)
            known_encodings = face_recognition.face_encodings(known_image)

            if not known_encodings:
                continue

            if face_recognition.compare_faces([known_encodings[0]], unknown_encoding)[0]:
                name = filename.split("_")[0]
                return {"match": True, "name": name}

        return {"match": False, "message": "No match found."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
