# face_database.py

import sqlite3
import numpy as np
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "faces.db")

def create_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding BLOB NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def insert_face(name, encoding, timestamp):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    encoding_blob = encoding.tobytes()
    cursor.execute(
        "INSERT INTO faces (name, encoding, timestamp) VALUES (?, ?, ?)",
        (name, encoding_blob, timestamp)
    )
    conn.commit()
    conn.close()

def get_all_faces():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM faces")
    rows = cursor.fetchall()
    conn.close()

    known_faces = []
    for name, encoding_blob in rows:
        encoding = np.frombuffer(encoding_blob, dtype=np.float64)
        known_faces.append((name, encoding))
    return known_faces
