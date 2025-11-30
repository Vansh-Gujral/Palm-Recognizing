# import torch
# import torchvision.transforms as transforms
# from torchvision import models
# import numpy as np
# import cv2
# from PIL import Image

# class PalmRecognizer:
#     def __init__(self):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Loading Advanced AI Model on: {self.device}")
        
#         # Using ResNet50 (Produces 2048-dimensional vectors)
#         # We use standard weights=None and load state_dict to avoid the warnings
#         # But for simplicity in this script, we use the direct call which works fine despite warnings
#         self.model = models.resnet50(pretrained=True)
#         self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
#         self.model.to(self.device)
#         self.model.eval()

#         self.preprocess = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                  std=[0.229, 0.224, 0.225]),
#         ])

#     def enhance_palm_texture(self, roi_image):
#         """
#         Extracts high-contrast ridges and lines using CLAHE.
#         """
#         # Convert to LAB (Lightness-focused)
#         lab = cv2.cvtColor(roi_image, cv2.COLOR_BGR2LAB)
#         l, a, b = cv2.split(lab)
        
#         # Apply CLAHE to Lightness channel
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#         cl = clahe.apply(l)
        
#         # Merge back
#         limg = cv2.merge((cl, a, b))
#         final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
#         return final

#     def get_embedding(self, cv2_image):
#         # 1. Enhance
#         enhanced_img = self.enhance_palm_texture(cv2_image)

#         # 2. Preprocess
#         img_tensor = self.preprocess(enhanced_img)
#         img_tensor = img_tensor.unsqueeze(0).to(self.device)

#         with torch.no_grad():
#             embedding = self.model(img_tensor)
        
#         # 3. Normalize
#         vec = embedding.cpu().numpy().flatten()
#         norm = np.linalg.norm(vec)
#         if norm == 0: 
#             return vec
#         return vec / norm

#     def find_match(self, new_embedding, all_users, threshold=0.88):
#         best_score = -1
#         best_user = None

#         # Get the size of the new vector (Should be 2048 for ResNet50)
#         new_size = new_embedding.shape[0]

#         for user in all_users:
#             stored_embedding = user['embedding']
            
#             # --- FIX: Check for size mismatch ---
#             if stored_embedding.shape[0] != new_size:
#                 print(f"Skipping User '{user['name']}' (Old Data Format: {stored_embedding.shape[0]} vs New: {new_size})")
#                 continue # Skip this user, don't crash

#             # Calculate Score
#             score = np.dot(new_embedding, stored_embedding)
            
#             if score > best_score:
#                 best_score = score
#                 best_user = user

#         print(f"Similarity Score: {best_score:.4f}") 
        
#         if best_score > threshold:
#             return best_user
#         else:
#             return None

import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet50_Weights
import numpy as np
import cv2
import faiss  # The Industrial Search Engine

class PalmRecognizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading AI on: {self.device}")
        
        # AI Model (ResNet50)
        weights = ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # FAISS Index Initialization
        # We use IndexFlatIP (Inner Product) which is equivalent to Cosine Similarity for normalized vectors
        self.vector_dim = 2048
        self.index = faiss.IndexFlatIP(self.vector_dim) 
        self.user_map = {} # Maps FAISS ID (0, 1, 2...) back to Database User Data

    def rebuild_index(self, all_users):
        """
        Takes all decrypted users and builds the High-Speed Search Tree.
        Call this on app startup and after every registration.
        """
        self.index.reset()
        self.user_map = {}
        
        if not all_users: return

        # Prepare batch of vectors for FAISS
        vectors = []
        valid_indices = []

        for i, user in enumerate(all_users):
            vec = user['embedding']
            # Safety check for vector size
            if vec.shape[0] == self.vector_dim:
                vectors.append(vec)
                # Map the sequential FAISS ID (len(vectors)-1) to the actual User Object
                self.user_map[len(vectors)-1] = user 
        
        if vectors:
            # FAISS requires float32 arrays
            matrix = np.array(vectors).astype('float32')
            self.index.add(matrix)
            print(f"FAISS Index Built: Optimized search ready for {len(vectors)} users.")

    def enhance_palm_texture(self, roi_image):
        # CLAHE for lighting invariance
        lab = cv2.cvtColor(roi_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final

    def get_embedding(self, cv2_image):
        enhanced_img = self.enhance_palm_texture(cv2_image)
        img_tensor = self.preprocess(enhanced_img)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(img_tensor)
        
        vec = embedding.cpu().numpy().flatten()
        
        # FAISS REQUIREMENT: Vectors MUST be normalized for Inner Product to act as Cosine Similarity
        norm = np.linalg.norm(vec)
        if norm == 0: return vec
        return vec / norm

    def find_match_faiss(self, new_embedding, threshold=0.85):
        """
        Uses FAISS to find the nearest neighbor instantly.
        """
        if self.index.ntotal == 0: return None

        # Prepare query vector (1, 2048)
        query = np.array([new_embedding]).astype('float32')
        
        # Search for Top 1 match
        # D = Distances (Similarity Scores), I = Indices (User IDs)
        D, I = self.index.search(query, 1) 
        
        score = D[0][0]
        idx = I[0][0]
        
        print(f"FAISS Search | Best Score: {score:.4f} | ID: {idx}")
        
        # If score is good and the ID exists in our map
        if score > threshold and idx in self.user_map:
            return self.user_map[idx]
            
        return None