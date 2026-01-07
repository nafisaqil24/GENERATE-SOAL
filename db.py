# db.py
import sqlite3

DB_PATH = "database.db"

def get_db():
    """Membuka koneksi ke SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # agar hasil query bisa diakses seperti dict
    return conn
