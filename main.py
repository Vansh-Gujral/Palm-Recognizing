# import tkinter as tk
# from tkinter import messagebox, simpledialog
# from PIL import Image, ImageTk
# import cv2
# import time

# from database import DatabaseManager
# from hand_tracker import HandTracker
# from recognition import PalmRecognizer

# class PalmPayApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Palm Pay - Biometric Payment System")
#         self.root.geometry("1100x700")
#         self.root.configure(bg="#1e1e1e") 

#         self.db = DatabaseManager()
#         self.tracker = HandTracker()
#         self.recognizer = PalmRecognizer()
        
#         self.cap = cv2.VideoCapture(0) 
#         self.is_running = True
#         self.mode = "IDLE" 
#         self.scanned_embedding = None
#         self.last_scan_time = 0

#         self.setup_ui()
        
#         self.update_frame()

#     def setup_ui(self):
#         self.lbl_title = tk.Label(self.root, text="Palm Pay System", font=("Arial", 24, "bold"), bg="#1e1e1e", fg="#00ffcc")
#         self.lbl_title.pack(pady=20)
#         self.lbl_video = tk.Label(self.root, bg="black", borderwidth=2, relief="solid")
#         self.lbl_video.pack()
#         self.lbl_status = tk.Label(self.root, text="Ready. Select a mode below.", font=("Arial", 14), bg="#1e1e1e", fg="white")
#         self.lbl_status.pack(pady=10)
#         btn_frame = tk.Frame(self.root, bg="#1e1e1e")
#         btn_frame.pack(pady=20)
#         btn_reg = tk.Button(btn_frame, text="Register New Hand", font=("Arial", 14), bg="#007bff", fg="white", 
#                             width=20, command=self.start_registration)
#         btn_reg.grid(row=0, column=0, padx=20)
#         btn_pay = tk.Button(btn_frame, text="Scan to Pay", font=("Arial", 14), bg="#28a745", fg="white", 
#                             width=20, command=self.start_payment)
#         btn_pay.grid(row=0, column=1, padx=20)

#     def start_registration(self):
#         self.mode = "REGISTER"
#         self.lbl_status.config(text="Mode: REGISTER. Place hand in the box.")
#         self.scanned_embedding = None

#     def start_payment(self):
#         self.mode = "PAY"
#         self.lbl_status.config(text="Mode: PAYMENT. Place hand in the box.")
#         self.scanned_embedding = None

#     def handle_auto_capture(self, roi, frame_display):
#         """
#         Called when hand is perfectly aligned.
#         """
#         current_time = time.time()
#         if current_time - self.last_scan_time < 2:
#             return

#         cv2.rectangle(frame_display, (0, 0), (frame_display.shape[1], frame_display.shape[0]), (0, 255, 0), 20)
#         embedding = self.recognizer.get_embedding(roi)
#         self.last_scan_time = current_time

#         if self.mode == "REGISTER":
#             self.register_logic(embedding)
#         elif self.mode == "PAY":
#             self.payment_logic(embedding)
#         self.mode = "IDLE"
#         self.lbl_status.config(text="Scan Complete. Select mode to continue.")

#     def register_logic(self, embedding):
#         name = simpledialog.askstring("Input", "Enter User Name:")
#         if not name: return
        
#         upi = simpledialog.askstring("Input", "Enter UPI ID (e.g., name@okicici):")
#         if not upi: return
#         self.db.add_user(name, upi, embedding)
#         messagebox.showinfo("Success", f"User {name} linked to {upi} successfully!")

#     def payment_logic(self, embedding):
#         users = self.db.get_all_users()
#         if not users:
#             messagebox.showwarning("Error", "No users in database!")
#             return
#         match = self.recognizer.find_match(embedding, users)

#         if match:
#             msg = f"Hand Verified!\nUser: {match['name']}\nUPI: {match['upi_id']}\n\nProceeding to Payment..."
#             messagebox.showinfo("Payment Successful", msg)
#         else:
#             messagebox.showerror("Failed", "Hand not recognized. Please try again.")

#     def update_frame(self):
#         if self.is_running:
#             ret, frame = self.cap.read()
#             if ret:
#                 frame = cv2.flip(frame, 1) 
                
#                 # 1. Process Hand
#                 results = self.tracker.process_frame(frame)
                
#                 # 2. Draw Blueprint
#                 box_coords = self.tracker.draw_blueprint(frame)
#                 x1, y1, x2, y2 = box_coords

#                 # 3. Check Alignment if hand detected
#                 if results.multi_hand_landmarks:
#                     for hand_landmarks in results.multi_hand_landmarks:
#                         self.tracker.mp_draw.draw_landmarks(frame, hand_landmarks, self.tracker.mp_hands.HAND_CONNECTIONS)
                        
#                         if self.tracker.is_hand_aligned(hand_landmarks, frame.shape, box_coords):
#                             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
#                             cv2.putText(frame, "ALIGNED - HOLD STILL", (x1, y2 + 30), 
#                                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
#                             if self.mode in ["REGISTER", "PAY"]:
#                                 roi = self.tracker.extract_roi(frame, box_coords)
#                                 self.handle_auto_capture(roi, frame)
#                         else:
#                             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

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

import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import time

# Import our modules
from database import DatabaseManager
from hand_tracker import HandTracker
from recognition import PalmRecognizer

class PalmPayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Palm Pay - Industrial Grade Biometrics")
        self.root.geometry("1100x750")
        self.root.configure(bg="#121212")

        # --- Initialize System Components ---
        self.db = DatabaseManager()
        self.tracker = HandTracker()
        self.recognizer = PalmRecognizer()
        
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        self.mode = "IDLE"
        self.last_scan_time = 0

        self.setup_ui()
        self.update_frame()

    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#1e1e1e")
        header_frame.pack(fill="x", pady=0)
        
        self.lbl_title = tk.Label(header_frame, text="PALM PAY SECURE", font=("Segoe UI", 20, "bold"), bg="#1e1e1e", fg="#00e5ff")
        self.lbl_title.pack(pady=10)

        # Video Area
        self.lbl_video = tk.Label(self.root, bg="black", borderwidth=2, relief="sunken")
        self.lbl_video.pack(pady=10)

        # Status Bar
        self.lbl_status = tk.Label(self.root, text="SYSTEM READY. SELECT OPTION.", font=("Consolas", 14), bg="#121212", fg="#00ff00")
        self.lbl_status.pack(pady=10)

        # Controls
        btn_frame = tk.Frame(self.root, bg="#121212")
        btn_frame.pack(pady=20)

        # Register Button
        btn_reg = tk.Button(btn_frame, text="[ REGISTER HAND ]", font=("Consolas", 12, "bold"), bg="#0069d9", fg="white", 
                            width=20, height=2, borderwidth=0, command=self.start_registration)
        btn_reg.grid(row=0, column=0, padx=20)

        # Pay Button
        btn_pay = tk.Button(btn_frame, text="[ SCAN TO PAY ]", font=("Consolas", 12, "bold"), bg="#218838", fg="white", 
                            width=20, height=2, borderwidth=0, command=self.start_payment)
        btn_pay.grid(row=0, column=1, padx=20)

    def start_registration(self):
        self.mode = "REGISTER"
        self.lbl_status.config(text="MODE: REGISTRATION - ALIGN HAND IN BLUEPRINT", fg="#00bfff")

    def start_payment(self):
        self.mode = "PAY"
        self.lbl_status.config(text="MODE: PAYMENT - ALIGN HAND TO SCAN", fg="#76ff03")

    def handle_auto_capture(self, roi, frame_display):
        current_time = time.time()
        if current_time - self.last_scan_time < 3: # 3 second cooldown
            return

        # Flash Effect
        cv2.rectangle(frame_display, (0, 0), (frame_display.shape[1], frame_display.shape[0]), (255, 255, 255), 30)
        self.root.update()
        
        self.lbl_status.config(text="PROCESSING BIOMETRICS...", fg="yellow")
        self.root.update()

        # Get Embedding
        embedding = self.recognizer.get_embedding(roi)
        self.last_scan_time = current_time

        if self.mode == "REGISTER":
            self.register_logic(embedding)
        elif self.mode == "PAY":
            self.payment_logic(embedding)
            
        self.mode = "IDLE"
        self.lbl_status.config(text="TRANSACTION COMPLETE. SYSTEM IDLE.", fg="white")

    def register_logic(self, embedding):
        name = simpledialog.askstring("Registration", "Enter User Name:")
        if not name: return
        upi = simpledialog.askstring("Registration", "Enter UPI ID:")
        if not upi: return

        self.db.add_user(name, upi, embedding)
        messagebox.showinfo("Success", f"Identity Registered: {name}")

    def payment_logic(self, embedding):
        users = self.db.get_all_users()
        if not users:
            messagebox.showwarning("Database Empty", "No users registered yet.")
            return

        # MATCHING LOGIC
        match = self.recognizer.find_match(embedding, users)

        if match:
            # SUCCESS
            msg = f"IDENTITY VERIFIED\n\nUser: {match['name']}\nLinked UPI: {match['upi_id']}"
            messagebox.showinfo("Payment Approved", msg)
        else:
            # FAILED - UNKNOWN USER
            response = messagebox.askyesno("Access Denied", "UNKNOWN HAND DETECTED.\n\nWould you like to register this hand now?")
            if response:
                self.register_logic(embedding)

    def update_frame(self):
        if self.is_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                
                # 1. Process Hand
                results = self.tracker.process_frame(frame)
                box_coords = self.tracker.draw_blueprint(frame)
                x1, y1, x2, y2 = box_coords

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # 2. Draw Advanced Visuals (Outline + Bones)
                        self.tracker.draw_advanced_visuals(frame, hand_landmarks)
                        
                        # 3. Check Alignment
                        if self.tracker.is_hand_aligned(hand_landmarks, frame.shape, box_coords):
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, "LOCKED - SCANNING", (x1, y2 + 25), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
                            if self.mode in ["REGISTER", "PAY"]:
                                roi = self.tracker.extract_roi(frame, box_coords)
                                self.handle_auto_capture(roi, frame)
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2) # Orange for misalignment

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