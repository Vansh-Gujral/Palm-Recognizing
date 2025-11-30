# Palm Recognition System

A Python-based palm recognition system using FAISS (vector search), Mediapipe (hand landmarks), OpenCV, and PyTorch.

🚀 Features

  Palm embedding extraction

  Palm matching using FAISS index

  Real-time camera input (OpenCV)

  Data storage for multiple users

  GUI support using Tkinter + PIL

📦 Requirements

  This project is confirmed to work on:

  Windows 10/11 (64-bit)

  Python 3.11

  FAISS 1.9 (CPU)

  NumPy 1.26 (⚠ required — FAISS does NOT support NumPy 2.x)

🔧 Installation
  1️⃣ Create Conda Environment
  conda create -n faissenv python=3.11
  conda activate faissenv
  
  2️⃣ Install FAISS (Windows-compatible)
  conda install -c conda-forge faiss-cpu
  
  3️⃣ Install All Python Dependencies
  pip install -r requirements.txt

▶️ Running the Project

  Run the main program:

    python main.py

📁 Project Structure
  Palm Recog/
    │ main.py
    │ recognition.py
    │ requirements.txt
    │ README.md
    │ models/
     │  data/
    └─ gui_assets/

❗ Important Notes

    Do not install NumPy 2.x, it breaks FAISS on Windows.
    
    OpenCV 4.9.0.80 is required (latest version forces NumPy 2).
    
    Mediapipe 0.10.8 avoids JAX dependency.
    
    FAISS must be installed via Conda on Windows (pip wheels are discontinued).

🧩 Troubleshooting

  FAISS import error?
    → Ensure NumPy is exactly:
  
    pip install numpy==1.26.4
  
  
  Mediapipe asks for JAX?
    → Use:
  
      pip install mediapipe==0.10.8
  
  
  OpenCV crashes?
    → Use:
  
      pip install opencv-python==4.9.0.80

Requirements

| Package       | Version you will have | Works with NumPy 1.26 |
| ------------- | --------------------- | --------------------- |
| numpy         | 1.26.4                | ✔                     |
| faiss-cpu     | 1.9.0                 | ✔                     |
| opencv-python | 4.9.0.80              | ✔                     |
| mediapipe     | 0.10.8                | ✔                     |
| torch         | 2.9.1                 | ✔                     |
| pillow        | 12.0.0                | ✔                     |





📝 License

MIT License (you can change this if needed).
