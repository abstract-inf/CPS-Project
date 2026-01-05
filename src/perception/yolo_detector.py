import cv2
import numpy as np
import torch
import os
from ultralytics import YOLOWorld

class OpenVocabularyDetector: 
    def __init__(self, model_name="yolov8s-worldv2.pt", confidence=0.1):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Robust model path handling
        local_model_path = os.path.join(os.getcwd(), 'models', model_name)
        
        print(f"Loading YOLO-World on {self.device}...")
        
        if os.path.exists(local_model_path):
            self.model = YOLOWorld(local_model_path)
        else:
            print(f"Downloading model to {local_model_path}...")
            self.model = YOLOWorld(model_name)

        self.conf_threshold = confidence
        self.classes_set = False

    def set_classes(self, text_queries):
        self.model.set_classes(text_queries)
        self.classes_set = True
        print(f"YOLO-World classes set to: {text_queries}")

    def detect_objects(self, image_bgr, text_queries=None): 
        if not self.classes_set and text_queries:
            self.set_classes(text_queries)

        results = self.model.predict(image_bgr, conf=self.conf_threshold, verbose=False)
        detections = []
        
        if results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = results[0].names[cls_id]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                detections.append({
                    "bbox": [x1, y1, x2-x1, y2-y1], # x,y,w,h
                    "label": label,
                    "score": conf,
                    "center": (cx, cy),
                    "segmentation": None 
                })
        return detections