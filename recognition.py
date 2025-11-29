# import torch
# import torchvision.transforms as transforms
# from torchvision import models
# import numpy as np
# from PIL import Image

# class PalmRecognizer:
#     def __init__(self):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Loading AI Model on: {self.device}")
#         self.model = models.resnet18(pretrained=True)
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

#     def get_embedding(self, cv2_image):
#         """
#         Takes an OpenCV image, runs it through the AI, and returns a 512-D vector.
#         """
#         img_tensor = self.preprocess(cv2_image)
#         img_tensor = img_tensor.unsqueeze(0).to(self.device) 
#         with torch.no_grad():
#             embedding = self.model(img_tensor)
        
#         return embedding.cpu().numpy().flatten()

#     def find_match(self, new_embedding, all_users, threshold=0.8):
#         """
#         Compares the new palm with all stored palms.
#         Returns the User object if a match is found, else None.
#         Uses Cosine Similarity.
#         """
#         best_score = -1
#         best_user = None

#         for user in all_users:
#             stored_embedding = user['embedding']
#             dot_product = np.dot(new_embedding, stored_embedding)
#             norm_a = np.linalg.norm(new_embedding)
#             norm_b = np.linalg.norm(stored_embedding)
            
#             score = dot_product / (norm_a * norm_b)

#             if score > best_score:
#                 best_score = score
#                 best_user = user

#         print(f"Best Match Score: {best_score}")
#         if best_score > threshold:
#             return best_user
#         return None

import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
from PIL import Image

class PalmRecognizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Advanced AI Model on: {self.device}")
        
        # Using ResNet50 (Produces 2048-dimensional vectors)
        # We use standard weights=None and load state_dict to avoid the warnings
        # But for simplicity in this script, we use the direct call which works fine despite warnings
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])

    def enhance_palm_texture(self, roi_image):
        """
        Extracts high-contrast ridges and lines using CLAHE.
        """
        # Convert to LAB (Lightness-focused)
        lab = cv2.cvtColor(roi_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to Lightness channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge back
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final

    def get_embedding(self, cv2_image):
        # 1. Enhance
        enhanced_img = self.enhance_palm_texture(cv2_image)

        # 2. Preprocess
        img_tensor = self.preprocess(enhanced_img)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(img_tensor)
        
        # 3. Normalize
        vec = embedding.cpu().numpy().flatten()
        norm = np.linalg.norm(vec)
        if norm == 0: 
            return vec
        return vec / norm

    def find_match(self, new_embedding, all_users, threshold=0.88):
        best_score = -1
        best_user = None

        # Get the size of the new vector (Should be 2048 for ResNet50)
        new_size = new_embedding.shape[0]

        for user in all_users:
            stored_embedding = user['embedding']
            
            # --- FIX: Check for size mismatch ---
            if stored_embedding.shape[0] != new_size:
                print(f"Skipping User '{user['name']}' (Old Data Format: {stored_embedding.shape[0]} vs New: {new_size})")
                continue # Skip this user, don't crash

            # Calculate Score
            score = np.dot(new_embedding, stored_embedding)
            
            if score > best_score:
                best_score = score
                best_user = user

        print(f"Similarity Score: {best_score:.4f}") 
        
        if best_score > threshold:
            return best_user
        else:
            return None