# import tkinter as tk
# from tkinter import messagebox, simpledialog
# from PIL import Image, ImageTk
# import cv2
# import time
# import numpy as np

# # Import our modules
# from database import DatabaseManager
# from hand_tracker import HandTracker
# from recognition import PalmRecognizer

# class PalmPayApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Palm Pay - Exam Version")
#         self.root.geometry("1100x700")
#         self.root.configure(bg="#1e1e1e")

#         self.db = DatabaseManager()
#         self.tracker = HandTracker()
#         self.recognizer = PalmRecognizer()
        
#         self.cap = cv2.VideoCapture(0)
#         self.is_running = True
#         self.mode = "IDLE" 
        
#         # Registration Variables
#         self.reg_samples = []
#         self.REQUIRED_SAMPLES = 5 # Takes 5 pics to ensure quality
        
#         self.last_scan_time = 0

#         self.setup_ui()
#         self.update_frame()

#     def setup_ui(self):
#         # Header
#         tk.Label(self.root, text="Palm Pay System", font=("Arial", 24, "bold"), bg="#1e1e1e", fg="white").pack(pady=20)

#         # Video
#         self.lbl_video = tk.Label(self.root, bg="black")
#         self.lbl_video.pack(pady=10)

#         # Status
#         self.lbl_status = tk.Label(self.root, text="System Ready", font=("Arial", 14), bg="#1e1e1e", fg="#00ff00")
#         self.lbl_status.pack(pady=10)

#         # Buttons
#         btn_frame = tk.Frame(self.root, bg="#1e1e1e")
#         btn_frame.pack(pady=20)

#         tk.Button(btn_frame, text="Register User", bg="#007bff", fg="white", font=("Arial", 12),
#                   width=20, command=self.start_registration).grid(row=0, column=0, padx=20)

#         tk.Button(btn_frame, text="Scan to Pay", bg="#28a745", fg="white", font=("Arial", 12),
#                   width=20, command=self.start_payment).grid(row=0, column=1, padx=20)
        
#         # Small debug window to show the rotation in action
#         self.lbl_debug = tk.Label(self.root, text="AI View", bg="black", fg="white")
#         self.lbl_debug.place(x=900, y=50)

#     def start_registration(self):
#         self.mode = "REGISTER"
#         self.reg_samples = []
#         self.lbl_status.config(text="Keep hand steady... Capturing 0/5", fg="cyan")

#     def start_payment(self):
#         self.mode = "PAY"
#         self.lbl_status.config(text="Scan your hand to pay...", fg="yellow")

#     def handle_registration(self, roi):
#         current_time = time.time()
#         # Take a sample every 0.3 seconds
#         if current_time - self.last_scan_time < 0.3: return
        
#         self.last_scan_time = current_time
        
#         # 1. Get embedding
#         embedding = self.recognizer.get_embedding(roi)
#         self.reg_samples.append(embedding)
        
#         count = len(self.reg_samples)
#         self.lbl_status.config(text=f"Capturing... {count}/{self.REQUIRED_SAMPLES}", fg="cyan")
        
#         # 2. If we have 5 samples, finish
#         if count >= self.REQUIRED_SAMPLES:
#             # Average the embeddings for high accuracy
#             avg_embedding = np.mean(self.reg_samples, axis=0)
#             # Normalize again
#             avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            
#             self.mode = "IDLE"
#             self.save_user(avg_embedding)

#     def save_user(self, embedding):
#         name = simpledialog.askstring("Register", "Enter Name:")
#         if name:
#             upi = simpledialog.askstring("Register", "Enter UPI ID:")
#             self.db.add_user(name, upi, embedding)
#             messagebox.showinfo("Success", "User Registered Successfully!")
#             self.lbl_status.config(text="System Ready", fg="#00ff00")
#         else:
#             self.lbl_status.config(text="Registration Cancelled", fg="red")

#     def handle_payment(self, roi):
#         current_time = time.time()
#         if current_time - self.last_scan_time < 2: return # 2 sec cooldown
        
#         self.last_scan_time = current_time
#         self.lbl_status.config(text="Processing...", fg="yellow")
#         self.root.update()

#         embedding = self.recognizer.get_embedding(roi)
#         match = self.recognizer.find_match(embedding, self.db.get_all_users())
        
#         if match:
#             msg = f"User: {match['name']}\nUPI: {match['upi_id']}\nStatus: Verified"
#             messagebox.showinfo("Payment Success", msg)
#             self.mode = "IDLE"
#             self.lbl_status.config(text="System Ready", fg="#00ff00")
#         else:
#             self.lbl_status.config(text="Hand Not Recognized", fg="red")

#     def update_frame(self):
#         if self.is_running:
#             ret, frame = self.cap.read()
#             if ret:
#                 frame = cv2.flip(frame, 1)
#                 results = self.tracker.process_frame(frame)
                
#                 # Draw standard overlay
#                 self.tracker.draw_blueprint(frame)

#                 if results.multi_hand_landmarks:
#                     for hand_landmarks in results.multi_hand_landmarks:
#                         self.tracker.draw_advanced_visuals(frame, hand_landmarks)
                        
#                         # --- THE CORE UPGRADE ---
#                         # Use the new Rotation-Correction Logic
#                         roi = self.tracker.get_palm_warp(frame, hand_landmarks)
                        
#                         if roi is not None:
#                             # Show the AI's view in the corner (Debug)
#                             # You will see this stays upright even if you tilt your hand!
#                             debug_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
#                             debug_img = debug_img.resize((100, 100))
#                             debug_tk = ImageTk.PhotoImage(image=debug_img)
#                             self.lbl_debug.imgtk = debug_tk
#                             self.lbl_debug.configure(image=debug_tk)

#                             if self.tracker.is_hand_stable(hand_landmarks, frame.shape):
#                                 if self.mode == "REGISTER":
#                                     self.handle_registration(roi)
#                                 elif self.mode == "PAY":
#                                     self.handle_payment(roi)

#                 img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#                 imgtk = ImageTk.PhotoImage(image=img)
#                 self.lbl_video.imgtk = imgtk
#                 self.lbl_video.configure(image=imgtk)

#         self.root.after(10, self.update_frame)

#     def on_close(self):
#         self.is_running = False
#         self.cap.release()
#         self.root.destroy()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = PalmPayApp(root)
#     root.protocol("WM_DELETE_WINDOW", app.on_close)
#     root.mainloop()

# import sqlite3
# import numpy as np
# import os
# from cryptography.fernet import Fernet

# class DatabaseManager:
#     def __init__(self, db_name="palm_pay_secure.db"):
#         # 1. Industrial Security: Key Management
#         self.key_file = "secret.key"
#         self.key = self.load_or_create_key()
#         self.cipher = Fernet(self.key)

#         self.conn = sqlite3.connect(db_name, check_same_thread=False)
#         self.create_tables()

#     def load_or_create_key(self):
#         """
#         Loads the existing encryption key or creates a new one.
#         CRITICAL: If you lose 'secret.key', the database becomes unreadable junk.
#         """
#         if os.path.exists(self.key_file):
#             with open(self.key_file, "rb") as file:
#                 return file.read()
#         else:
#             print("Generating new AES Encryption Key...")
#             key = Fernet.generate_key()
#             with open(self.key_file, "wb") as file:
#                 file.write(key)
#             return key

#     def create_tables(self):
#         cursor = self.conn.cursor()
#         # Note: We store 'encrypted_embedding' instead of raw embedding
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS users (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 name TEXT NOT NULL,
#                 upi_id TEXT NOT NULL,
#                 balance REAL DEFAULT 500.0,
#                 encrypted_embedding BLOB NOT NULL
#             )
#         """)
#         self.conn.commit()

#     def add_user(self, name, upi_id, embedding):
#         cursor = self.conn.cursor()
        
#         # 1. Convert to Bytes
#         binary_data = embedding.tobytes()
        
#         # 2. ENCRYPT the data (AES-256)
#         # This turns your biometric data into random noise
#         encrypted_data = self.cipher.encrypt(binary_data)
        
#         cursor.execute("INSERT INTO users (name, upi_id, encrypted_embedding) VALUES (?, ?, ?)",
#                        (name, upi_id, encrypted_data))
#         self.conn.commit()
#         print(f"User {name} added with AES-256 Encryption.")

#     def get_all_users(self):
#         """
#         Decrypts data on-the-fly when needed for the AI.
#         """
#         cursor = self.conn.cursor()
#         cursor.execute("SELECT id, name, upi_id, encrypted_embedding FROM users")
#         rows = cursor.fetchall()
        
#         users = []
#         for row in rows:
#             uid, name, upi, enc_blob = row
            
#             try:
#                 # 3. DECRYPT data
#                 # Only this program (with the key) can turn the noise back into a palm print
#                 decrypted_blob = self.cipher.decrypt(enc_blob)
#                 embedding = np.frombuffer(decrypted_blob, dtype=np.float32)
                
#                 users.append({
#                     "id": uid,
#                     "name": name,
#                     "upi_id": upi,
#                     "embedding": embedding
#                 })
#             except Exception as e:
#                 print(f"Security Alert: Could not decrypt user {uid}. Data may be tampered.")
                
#         return users

#     def close(self):
#         self.conn.close()

import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import time
import numpy as np

# Import our modules
from database import DatabaseManager
from hand_tracker import HandTracker
from recognition import PalmRecognizer

class PalmPayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Palm Pay - Industrial Encrypted")
        self.root.geometry("1100x700")
        self.root.configure(bg="#1e1e1e")

        self.db = DatabaseManager()
        self.tracker = HandTracker()
        self.recognizer = PalmRecognizer()
        
        # --- INDUSTRIAL UPGRADE: Build Search Index ---
        print("Initializing FAISS Search Engine...")
        self.reload_faiss_index()
        
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        self.mode = "IDLE" 
        
        # Registration Variables (5-shot averaging)
        self.reg_samples = []
        self.REQUIRED_SAMPLES = 5 
        
        self.last_scan_time = 0

        self.setup_ui()
        self.update_frame()

    def reload_faiss_index(self):
        """
        Fetches all decrypted users and rebuilds the fast search index.
        """
        all_users = self.db.get_all_users()
        self.recognizer.rebuild_index(all_users)

    def setup_ui(self):
        # Header
        tk.Label(self.root, text="Palm Pay | Secure Industrial", font=("Arial", 24, "bold"), bg="#1e1e1e", fg="white").pack(pady=20)

        # Video
        self.lbl_video = tk.Label(self.root, bg="black")
        self.lbl_video.pack(pady=10)

        # Status
        self.lbl_status = tk.Label(self.root, text="System Ready (AES-256 Active)", font=("Arial", 14), bg="#1e1e1e", fg="#00ff00")
        self.lbl_status.pack(pady=10)

        # Buttons
        btn_frame = tk.Frame(self.root, bg="#1e1e1e")
        btn_frame.pack(pady=20)

        tk.Button(btn_frame, text="Register User", bg="#007bff", fg="white", font=("Arial", 12),
                  width=20, command=self.start_registration).grid(row=0, column=0, padx=20)

        tk.Button(btn_frame, text="Scan to Pay", bg="#28a745", fg="white", font=("Arial", 12),
                  width=20, command=self.start_payment).grid(row=0, column=1, padx=20)
        
        # Debug window
        self.lbl_debug = tk.Label(self.root, text="AI View", bg="black", fg="white")
        self.lbl_debug.place(x=900, y=50)

    def start_registration(self):
        self.mode = "REGISTER"
        self.reg_samples = []
        self.lbl_status.config(text="Keep hand steady... Capturing 0/5", fg="cyan")

    def start_payment(self):
        self.mode = "PAY"
        self.lbl_status.config(text="Scan your hand to pay...", fg="yellow")

    def handle_registration(self, roi):
        current_time = time.time()
        if current_time - self.last_scan_time < 0.3: return
        
        self.last_scan_time = current_time
        
        # 1. Get embedding
        embedding = self.recognizer.get_embedding(roi)
        self.reg_samples.append(embedding)
        
        count = len(self.reg_samples)
        self.lbl_status.config(text=f"Capturing... {count}/{self.REQUIRED_SAMPLES}", fg="cyan")
        
        # 2. If we have 5 samples, finish
        if count >= self.REQUIRED_SAMPLES:
            # Average the embeddings for high accuracy
            avg_embedding = np.mean(self.reg_samples, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            
            self.mode = "IDLE"
            self.save_user(avg_embedding)

    def save_user(self, embedding):
        name = simpledialog.askstring("Register", "Enter Name:")
        if name:
            upi = simpledialog.askstring("Register", "Enter UPI ID:")
            # Save to Secure DB
            self.db.add_user(name, upi, embedding)
            # Update Fast Search Index
            self.reload_faiss_index()
            
            messagebox.showinfo("Success", "User Registered Encrypted & Indexed!")
            self.lbl_status.config(text="System Ready", fg="#00ff00")
        else:
            self.lbl_status.config(text="Registration Cancelled", fg="red")

    def handle_payment(self, roi):
        current_time = time.time()
        if current_time - self.last_scan_time < 2: return 
        
        self.last_scan_time = current_time
        self.lbl_status.config(text="Searching Secure Index...", fg="yellow")
        self.root.update()

        # 1. Get Vector
        embedding = self.recognizer.get_embedding(roi)
        
        # 2. FAISS SEARCH (Instant Match)
        match = self.recognizer.find_match_faiss(embedding)
        
        if match:
            msg = f"User: {match['name']}\nUPI: {match['upi_id']}\nStatus: Authenticated"
            messagebox.showinfo("Payment Success", msg)
            self.mode = "IDLE"
            self.lbl_status.config(text="System Ready", fg="#00ff00")
        else:
            self.lbl_status.config(text="Hand Not Recognized", fg="red")

    def update_frame(self):
        if self.is_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                results = self.tracker.process_frame(frame)
                
                self.tracker.draw_blueprint(frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.tracker.draw_advanced_visuals(frame, hand_landmarks)
                        
                        roi = self.tracker.get_palm_warp(frame, hand_landmarks)
                        
                        if roi is not None:
                            # Debug View
                            debug_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                            debug_img = debug_img.resize((100, 100))
                            debug_tk = ImageTk.PhotoImage(image=debug_img)
                            self.lbl_debug.imgtk = debug_tk
                            self.lbl_debug.configure(image=debug_tk)

                            if self.tracker.is_hand_stable(hand_landmarks, frame.shape):
                                if self.mode == "REGISTER":
                                    self.handle_registration(roi)
                                elif self.mode == "PAY":
                                    self.handle_payment(roi)

                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.lbl_video.imgtk = imgtk
                self.lbl_video.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def on_close(self):
        self.is_running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PalmPayApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()