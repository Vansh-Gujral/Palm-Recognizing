# import cv2
# import mediapipe as mp
# import numpy as np

# class HandTracker:
#     def __init__(self):
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=1,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.7
#         )
#         self.mp_draw = mp.solutions.drawing_utils

#     def process_frame(self, frame):
#         """
#         Detects hand landmarks in the frame.
#         """
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.hands.process(frame_rgb)
#         return results

#     def draw_blueprint(self, frame):
#         """
#         Draws a guide overlay (Blueprint) on the screen.
#         Users must place their hand inside this box.
#         """
#         h, w, c = frame.shape
#         box_size = 300
#         x1 = (w - box_size) // 2
#         y1 = (h - box_size) // 2
#         x2 = x1 + box_size
#         y2 = y1 + box_size
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
#         cv2.putText(frame, "Place Hand Here", (x1 + 60, y1 - 10), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
#         return (x1, y1, x2, y2)

#     def is_hand_aligned(self, landmarks, frame_shape, box_coords):
#         """
#         Checks if the hand is perfectly inside the blueprint box.
#         Returns: True if aligned, False otherwise.
#         """
#         h, w, _ = frame_shape
#         x1, y1, x2, y2 = box_coords
#         wrist = landmarks.landmark[0]
#         middle_tip = landmarks.landmark[12]
#         wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
#         mid_x, mid_y = int(middle_tip.x * w), int(middle_tip.y * h)
#         margin = 40 

#         in_box_x = x1 < wrist_x < x2 and x1 < mid_x < x2
#         in_box_y = y1 < mid_y < y2 and y1 < wrist_y < y2
        
#         is_vertical = abs(wrist_x - mid_x) < 80

#         if in_box_x and in_box_y and is_vertical:
#             return True
#         return False

#     def extract_roi(self, frame, box_coords):
#         """
#         Crops the image to the blueprint box for the AI model.
#         """
#         x1, y1, x2, y2 = box_coords
#         roi = frame[y1:y2, x1:x2]
#         return roi

# import cv2
# import mediapipe as mp
# import numpy as np

# class HandTracker:
#     def __init__(self):
#         self.mp_hands = mp.solutions.hands
#         # Increased confidence to ensure the outline doesn't jitter
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=1,
#             min_detection_confidence=0.75,
#             min_tracking_confidence=0.75
#         )
#         self.mp_draw = mp.solutions.drawing_utils
#         self.mp_drawing_styles = mp.solutions.drawing_styles

#     def process_frame(self, frame):
#         """
#         Detects hand landmarks in the frame.
#         """
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.hands.process(frame_rgb)
#         return results

#     def draw_advanced_visuals(self, frame, landmarks):
#         """
#         Draws a sci-fi style outline and structure around the hand.
#         """
#         h, w, c = frame.shape
        
#         # 1. Collect all landmark points
#         points = []
#         for lm in landmarks.landmark:
#             cx, cy = int(lm.x * w), int(lm.y * h)
#             points.append([cx, cy])
        
#         points = np.array(points, dtype=np.int32)

#         # 2. Draw the Convex Hull (The Outline wrapping the hand)
#         hull = cv2.convexHull(points)
        
#         # Draw filled semi-transparent hand backing
#         overlay = frame.copy()
#         cv2.fillConvexPoly(overlay, hull, (0, 255, 255)) # Yellow tint
#         alpha = 0.2 # Transparency
#         cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

#         # Draw the solid outline
#         cv2.polylines(frame, [hull], True, (0, 255, 255), 2)

#         # 3. Draw connections (Bones) with thicker, distinct lines
#         self.mp_draw.draw_landmarks(
#             frame, 
#             landmarks, 
#             self.mp_hands.HAND_CONNECTIONS,
#             self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), # Connections
#             self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4)  # Joints
#         )

#     def draw_blueprint(self, frame):
#         """
#         Draws a guide overlay (Blueprint) on the screen.
#         Users must place their hand inside this box.
#         """
#         h, w, c = frame.shape
#         box_size = 320 # Slightly larger to fit full hand
#         x1 = (w - box_size) // 2
#         y1 = (h - box_size) // 2
#         x2 = x1 + box_size
#         y2 = y1 + box_size

#         # Draw the rectangle corners (Bracket style)
#         color = (255, 255, 255)
#         thickness = 2
#         d = 30 # line length
        
#         # Top-Left
#         cv2.line(frame, (x1, y1), (x1 + d, y1), color, thickness)
#         cv2.line(frame, (x1, y1), (x1, y1 + d), color, thickness)
#         # Top-Right
#         cv2.line(frame, (x2, y1), (x2 - d, y1), color, thickness)
#         cv2.line(frame, (x2, y1), (x2, y1 + d), color, thickness)
#         # Bottom-Left
#         cv2.line(frame, (x1, y2), (x1 + d, y2), color, thickness)
#         cv2.line(frame, (x1, y2), (x1, y2 - d), color, thickness)
#         # Bottom-Right
#         cv2.line(frame, (x2, y2), (x2 - d, y2), color, thickness)
#         cv2.line(frame, (x2, y2), (x2, y2 - d), color, thickness)

#         cv2.putText(frame, "Align Hand Here", (x1 + 50, y1 - 15), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
        
#         return (x1, y1, x2, y2)

#     def is_hand_aligned(self, landmarks, frame_shape, box_coords):
#         h, w, _ = frame_shape
#         x1, y1, x2, y2 = box_coords

#         wrist = landmarks.landmark[0]
#         middle_tip = landmarks.landmark[12]
        
#         wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
#         mid_x, mid_y = int(middle_tip.x * w), int(middle_tip.y * h)

#         # Stricter alignment logic
#         margin = 20
#         in_box_x = x1 < wrist_x < x2 and x1 < mid_x < x2
#         in_box_y = y1 < mid_y < y2 and y1 < wrist_y < y2
#         is_vertical = abs(wrist_x - mid_x) < 60 

#         return in_box_x and in_box_y and is_vertical

#     def extract_roi(self, frame, box_coords):
#         x1, y1, x2, y2 = box_coords
#         # Ensure coordinates are within frame bounds
#         h, w, _ = frame.shape
#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(w, x2), min(h, y2)
        
#         roi = frame[y1:y2, x1:x2]
#         return roi


# import cv2
# import mediapipe as mp
# import numpy as np

# class HandTracker:
#     def __init__(self):
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=1,
#             min_detection_confidence=0.75,
#             min_tracking_confidence=0.75
#         )
#         self.mp_draw = mp.solutions.drawing_utils

#     def process_frame(self, frame):
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.hands.process(frame_rgb)
#         return results

#     def draw_advanced_visuals(self, frame, landmarks):
#         """
#         Draws a clean, anatomical structure of the hand.
#         """
#         h, w, c = frame.shape
        
#         self.mp_draw.draw_landmarks(
#             frame, 
#             landmarks, 
#             self.mp_hands.HAND_CONNECTIONS,
#             self.mp_draw.DrawingSpec(color=(240, 240, 240), thickness=2, circle_radius=1), # Connections: Off-White
#             self.mp_draw.DrawingSpec(color=(0, 120, 255), thickness=2, circle_radius=3)    # Joints: Professional Blue
#         )

#         # 2. Draw the Hand Outline (Tracing the skin)
#         # We connect specific landmarks to create a contour around the hand
#         # Thumb(1-4), Index(5-8), Middle(9-12), Ring(13-16), Pinky(17-20), Base(0)
        
#         outline_indices = [
#             0, 1, 2, 3, 4,      # Thumb Outer
#             8,                  # Index Tip 
#             5, 6, 7, 8,         # Index
#             9, 10, 11, 12,      # Middle
#             13, 14, 15, 16,     # Ring
#             17, 18, 19, 20,     # Pinky
#             0                   # Back to Wrist
#         ]
#         self.draw_line(frame, landmarks, 0, 1, w, h)
#         self.draw_line(frame, landmarks, 1, 2, w, h)
#         self.draw_line(frame, landmarks, 2, 3, w, h)
#         self.draw_line(frame, landmarks, 3, 4, w, h)
        
#         tips = [4, 8, 12, 16, 20]
#         for i in range(len(tips) - 1):
#             self.draw_line(frame, landmarks, tips[i], tips[i+1], w, h, (255, 255, 255), 1)

#     def draw_line(self, frame, landmarks, idx1, idx2, w, h, color=(255, 255, 255), thickness=1):
#         p1 = landmarks.landmark[idx1]
#         p2 = landmarks.landmark[idx2]
#         pt1 = (int(p1.x * w), int(p1.y * h))
#         pt2 = (int(p2.x * w), int(p2.y * h))
#         cv2.line(frame, pt1, pt2, color, thickness)

#     def draw_blueprint(self, frame):
#         h, w, c = frame.shape
#         box_size = 320 
#         x1 = (w - box_size) // 2
#         y1 = (h - box_size) // 2
#         x2 = x1 + box_size
#         y2 = y1 + box_size

#         color = (200, 200, 200) 
#         thickness = 2
        
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        
#         d = 20
#         cv2.line(frame, (x1, y1), (x1 + d, y1), (255, 255, 255), 2)
#         cv2.line(frame, (x1, y1), (x1, y1 + d), (255, 255, 255), 2)
        
#         cv2.line(frame, (x2, y1), (x2 - d, y1), (255, 255, 255), 2)
#         cv2.line(frame, (x2, y1), (x2, y1 + d), (255, 255, 255), 2)
        
#         cv2.line(frame, (x1, y2), (x1 + d, y2), (255, 255, 255), 2)
#         cv2.line(frame, (x1, y2), (x1, y2 - d), (255, 255, 255), 2)
        
#         cv2.line(frame, (x2, y2), (x2 - d, y2), (255, 255, 255), 2)
#         cv2.line(frame, (x2, y2), (x2, y2 - d), (255, 255, 255), 2)

#         cv2.putText(frame, "Scan Zone", (x1 + 110, y1 - 10), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
#         return (x1, y1, x2, y2)

#     def is_hand_aligned(self, landmarks, frame_shape, box_coords):
#         h, w, _ = frame_shape
#         x1, y1, x2, y2 = box_coords

#         wrist = landmarks.landmark[0]
#         middle_tip = landmarks.landmark[12]
        
#         wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
#         mid_x, mid_y = int(middle_tip.x * w), int(middle_tip.y * h)

#         in_box_x = x1 < wrist_x < x2 and x1 < mid_x < x2
#         in_box_y = y1 < mid_y < y2 and y1 < wrist_y < y2
#         is_vertical = abs(wrist_x - mid_x) < 60 

#         return in_box_x and in_box_y and is_vertical

#     def extract_roi(self, frame, box_coords):
#         x1, y1, x2, y2 = box_coords
#         h, w, _ = frame.shape
#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(w, x2), min(h, y2)
        
#         roi = frame[y1:y2, x1:x2]
#         return roi

import cv2
import mediapipe as mp
import numpy as np
import math

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        return results

    def get_palm_warp(self, frame, landmarks):
        """
        INDUSTRIAL GRADE CROPPER
        1. Calculates hand angle.
        2. Rotates image to make hand upright.
        3. Crops ONLY the palm center (ignoring fingers).
        """
        h, w, c = frame.shape
        lm = landmarks.landmark

        # 1. Get Key Coordinates
        # Wrist (0) and Middle Finger Knuckle (9) define the hand's central axis
        wrist = np.array([lm[0].x * w, lm[0].y * h])
        middle_mcp = np.array([lm[9].x * w, lm[9].y * h])
        
        # 2. Calculate Angle
        delta = middle_mcp - wrist
        angle_rad = math.atan2(delta[1], delta[0])
        angle_deg = math.degrees(angle_rad)
        
        # We want the hand pointing UP (-90 degrees). 
        # Calculate how much we need to rotate to get there.
        rotation_angle = angle_deg + 90

        # 3. Calculate Center of Rotation (The Palm Center)
        center_x = (wrist[0] + middle_mcp[0]) / 2
        center_y = (wrist[1] + middle_mcp[1]) / 2

        # 4. Determine Crop Size based on hand width
        # Distance between Index Knuckle (5) and Pinky Knuckle (17)
        index_mcp = np.array([lm[5].x * w, lm[5].y * h])
        pinky_mcp = np.array([lm[17].x * w, lm[17].y * h])
        hand_width = np.linalg.norm(index_mcp - pinky_mcp)
        
        # We crop a square roughly 2.2x the width of the palm
        # This covers the lifeline but cuts off the fingers
        crop_size = int(hand_width * 2.2)

        # 5. Rotate the Image
        M = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
        rotated_frame = cv2.warpAffine(frame, M, (w, h))

        # 6. Crop the Square from the Rotated Image
        x1 = int(center_x - crop_size // 2)
        y1 = int(center_y - crop_size // 2)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        # Boundary checks
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return None

        roi = rotated_frame[y1:y2, x1:x2]
        
        try:
            roi = cv2.resize(roi, (224, 224))
            return roi
        except:
            return None

    def draw_advanced_visuals(self, frame, landmarks):
        """
        Clean, Anatomical Visuals. No Flashing.
        """
        h, w, c = frame.shape
        
        # Draw Skeleton (White)
        self.mp_draw.draw_landmarks(
            frame, 
            landmarks, 
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1),
            self.mp_draw.DrawingSpec(color=(0, 121, 255), thickness=2, circle_radius=2)
        )
        
        # Draw Crop Zone Preview (Green Circle in center)
        lm = landmarks.landmark
        wrist = np.array([lm[0].x * w, lm[0].y * h])
        middle_mcp = np.array([lm[9].x * w, lm[9].y * h])
        cx, cy = int((wrist[0] + middle_mcp[0]) / 2), int((wrist[1] + middle_mcp[1]) / 2)
        
        # This shows the user where the AI is looking
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.circle(frame, (cx, cy), 60, (0, 255, 0), 1)

    def draw_blueprint(self, frame):
        h, w, c = frame.shape
        box_size = 300
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        # Simple Guide Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
        cv2.putText(frame, "Place Hand Here", (x1 + 80, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return (x1, y1, x2, y2)

    def is_hand_stable(self, landmarks, frame_shape):
        # Relaxed check: Just ensure hand is roughly in frame
        h, w, _ = frame_shape
        wrist = landmarks.landmark[0]
        return 0.2 < wrist.x < 0.8 and 0.2 < wrist.y < 0.8