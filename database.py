import sqlite3
import numpy as np
import io

class DatabaseManager:
    def __init__(self, db_name="palm_pay.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                upi_id TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        """)
        self.conn.commit()

    def add_user(self, name, upi_id, embedding):
        """
        Saves a new user.
        embedding: A numpy array representing the palm features.
        """
        cursor = self.conn.cursor()
        binary_embedding = embedding.tobytes()
        cursor.execute("INSERT INTO users (name, upi_id, embedding) VALUES (?, ?, ?)",
                       (name, upi_id, binary_embedding))
        self.conn.commit()
        print(f"User {name} added successfully.")

    def get_all_users(self):
        """
        Retrieves all users to compare against during payment.
        Returns a list of dictionaries.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, upi_id, embedding FROM users")
        rows = cursor.fetchall()
        
        users = []
        for row in rows:
            name, upi, blob = row
            embedding = np.frombuffer(blob, dtype=np.float32)
            users.append({
                "name": name,
                "upi_id": upi,
                "embedding": embedding
            })
        return users

    def close(self):
        self.conn.close()