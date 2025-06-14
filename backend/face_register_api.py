from fastapi import APIRouter, UploadFile, Form, HTTPException
from datetime import datetime
import os
import shutil
import subprocess

router = APIRouter()
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "registered")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def call_build_vectorstore():
    backend_folder = os.path.dirname(os.path.abspath(__file__))
    ragengine_folder = os.path.abspath(os.path.join(backend_folder, "../ragengine"))
    print("Running build_vectorstore.py from folder:", ragengine_folder)
    
    # Run build_vectorstore.py inside ragengine folder
    subprocess.run(["python", "build_vectorstore.py"], check=True, cwd=ragengine_folder)

@router.post("/register-face")
async def register_face(name: str = Form(...), file: UploadFile = Form(...)):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{name}_{timestamp}.jpg"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Save metadata (make sure path is correct)
        metadata_path = os.path.join(os.path.dirname(__file__), "face_metadata.txt")
        with open(metadata_path, "a") as f:
            f.write(f"{name} registered at {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

        # Call vectorstore build script with fixed working directory
        call_build_vectorstore()

        return {"message": "Face registered successfully"}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Vectorstore build failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
