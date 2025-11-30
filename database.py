# import sqlite3
# import numpy as np
# import io

# class DatabaseManager:
#     def __init__(self, db_name="palm_pay.db"):
#         self.conn = sqlite3.connect(db_name, check_same_thread=False)
#         self.create_tables()

#     def create_tables(self):
#         cursor = self.conn.cursor()
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS users (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 name TEXT NOT NULL,
#                 upi_id TEXT NOT NULL,
#                 embedding BLOB NOT NULL
#             )
#         """)
#         self.conn.commit()

#     def add_user(self, name, upi_id, embedding):
#         """
#         Saves a new user.
#         embedding: A numpy array representing the palm features.
#         """
#         cursor = self.conn.cursor()
#         binary_embedding = embedding.tobytes()
#         cursor.execute("INSERT INTO users (name, upi_id, embedding) VALUES (?, ?, ?)",
#                        (name, upi_id, binary_embedding))
#         self.conn.commit()
#         print(f"User {name} added successfully.")

#     def get_all_users(self):
#         """
#         Retrieves all users to compare against during payment.
#         Returns a list of dictionaries.
#         """
#         cursor = self.conn.cursor()
#         cursor.execute("SELECT name, upi_id, embedding FROM users")
#         rows = cursor.fetchall()
        
#         users = []
#         for row in rows:
#             name, upi, blob = row
#             embedding = np.frombuffer(blob, dtype=np.float32)
#             users.append({
#                 "name": name,
#                 "upi_id": upi,
#                 "embedding": embedding
#             })
#             # print(name)
        
#         return users

#     def close(self):
#         self.conn.close()

import sqlite3
import numpy as np
import os
from cryptography.fernet import Fernet

class DatabaseManager:
    def __init__(self, db_name="palm_pay_secure.db"):
        # 1. Industrial Security: Key Management
        self.key_file = "secret.key"
        self.key = self.load_or_create_key()
        self.cipher = Fernet(self.key)

        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.create_tables()

    def load_or_create_key(self):
        """
        Loads the existing encryption key or creates a new one.
        CRITICAL: If you lose 'secret.key', the database becomes unreadable junk.
        """
        if os.path.exists(self.key_file):
            with open(self.key_file, "rb") as file:
                return file.read()
        else:
            print("Generating new AES Encryption Key...")
            key = Fernet.generate_key()
            with open(self.key_file, "wb") as file:
                file.write(key)
            return key

    def create_tables(self):
        cursor = self.conn.cursor()
        # Note: We store 'encrypted_embedding' instead of raw embedding
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                upi_id TEXT NOT NULL,
                balance REAL DEFAULT 500.0,
                encrypted_embedding BLOB NOT NULL
            )
        """)
        self.conn.commit()

    def add_user(self, name, upi_id, embedding):
        cursor = self.conn.cursor()
        
        # 1. Convert to Bytes
        binary_data = embedding.tobytes()
        
        # 2. ENCRYPT the data (AES-256)
        # This turns your biometric data into random noise
        encrypted_data = self.cipher.encrypt(binary_data)
        
        cursor.execute("INSERT INTO users (name, upi_id, encrypted_embedding) VALUES (?, ?, ?)",
                       (name, upi_id, encrypted_data))
        self.conn.commit()
        print(f"User {name} added with AES-256 Encryption.")

    def get_all_users(self):
        """
        Decrypts data on-the-fly when needed for the AI.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, upi_id, encrypted_embedding FROM users")
        rows = cursor.fetchall()
        
        users = []
        for row in rows:
            uid, name, upi, enc_blob = row
            
            try:
                # 3. DECRYPT data
                # Only this program (with the key) can turn the noise back into a palm print
                decrypted_blob = self.cipher.decrypt(enc_blob)
                embedding = np.frombuffer(decrypted_blob, dtype=np.float32)
                
                users.append({
                    "id": uid,
                    "name": name,
                    "upi_id": upi,
                    "embedding": embedding
                })
            except Exception as e:
                print(f"Security Alert: Could not decrypt user {uid}. Data may be tampered.")
                
        return users

    def close(self):
        self.conn.close()