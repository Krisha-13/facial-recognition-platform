from fastapi import APIRouter, UploadFile, Form, HTTPException
from datetime import datetime
import os
import shutil
import subprocess
import cv2
import face_recognition
import numpy as np

router = APIRouter()

# === Paths ===
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "registered")
ENCODING_PATH = os.path.join(BASE_DIR, "../face-recognition/known_faces.npy")
METADATA_PATH = os.path.join(BASE_DIR, "face_metadata.txt")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# === Vectorstore Trigger ===
def call_build_vectorstore():
    ragengine_folder = os.path.abspath(os.path.join(BASE_DIR, "../ragengine"))
    print("✅ Rebuilding vectorstore from:", ragengine_folder)
    subprocess.run(["python", "build_vectorstore.py"], check=True, cwd=ragengine_folder)

# === Register Face Endpoint ===
@router.post("/register-face")
async def register_face(name: str = Form(...), file: UploadFile = Form(...)):
    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{name}_{timestamp}.jpg"
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load and process image
        image = cv2.imread(file_path)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) != 1:
            raise HTTPException(status_code=400, detail="❌ Please make sure exactly one face is visible.")

        face_encoding = face_recognition.face_encodings(image, face_locations)[0]

        # Load or create known_faces list
        if os.path.exists(ENCODING_PATH):
            data = np.load(ENCODING_PATH, allow_pickle=True).tolist()
            if not isinstance(data, list):
                data = []
        else:
            data = []

        # ✅ Check for existing match and update if found
        updated = False
        for i, face in enumerate(data):
            existing_encoding = np.array(face["encoding"])
            distance = face_recognition.face_distance([existing_encoding], face_encoding)[0]
            if distance < 0.45:
                data[i] = {
                    "name": name,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "encoding": face_encoding.tolist()
                }
                updated = True
                break

        # ✅ If no match found, append new entry
        if not updated:
            data.append({
                "name": name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "encoding": face_encoding.tolist()
            })

        # Save to known_faces.npy
        np.save(ENCODING_PATH, np.array(data, dtype=object))

        # Save readable metadata
        with open(METADATA_PATH, "a") as f:
            f.write(f"{name} registered at {timestamp}\n")

        # Rebuild RAG vectorstore
        call_build_vectorstore()

        return {"message": f"✅ Face registered for {name}"}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Vectorstore build failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
