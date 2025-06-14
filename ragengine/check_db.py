import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "../database/faces.db")

def check_faces():
    if not os.path.exists(DB_PATH):
        print("❌ Database not found at:", DB_PATH)
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT name, timestamp FROM faces")
        rows = cursor.fetchall()
        if rows:
            print("✅ Found face records:")
            for row in rows:
                print(" -", row)
        else:
            print("⚠️ No records found in the database.")
    except sqlite3.Error as e:
        print("❌ Error querying database:", e)

    conn.close()

if __name__ == "__main__":
    check_faces()
