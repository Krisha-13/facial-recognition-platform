from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "🎉 Python Backend is Running!"

@app.route('/verify-face', methods=['POST'])
def verify_face():
    try:
        file = request.files['image']
        unknown_image = face_recognition.load_image_file(file)
        unknown_encoding = face_recognition.face_encodings(unknown_image)

        if not unknown_encoding:
            return jsonify({"success": False, "message": "No face detected."})

        # Placeholder known encoding (replace with real one)
        known_encoding = np.load("known_face.npy")

        results = face_recognition.compare_faces([known_encoding], unknown_encoding[0])
        return jsonify({"success": True, "match": results[0]})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
