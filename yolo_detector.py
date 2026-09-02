"""
YOLO11x Model Wrapper and PPE Inference Engine.
Loads Ultralytics YOLO11 models (yolo11x.pt, yolo11l.pt, or user custom fine-tuned weights),
executes detection, handles class filtering, and analyzes PPE compliance.
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

from config import PPE_CLASSES, COCO_PPE_SIMULATION_MAP


class PPEDetector:
    def __init__(self, model_path="yolo11x.pt"):
        """
        Initialize YOLO model from file or weights.
        :param model_path: Path to .pt file or ultralytics model string (e.g. 'yolo11x.pt')
        """
        self.model_path = model_path
        self.model = None
        self.load_model(model_path)
        
    def load_model(self, path):
        try:
            # Load Ultralytics YOLO model
            self.model = YOLO(path)
            self.class_names = self.model.names
            self.is_custom_ppe_model = any(
                k.lower() in [c.lower() for c in self.class_names.values()]
                for k in ['hardhat', 'helmet', 'vest', 'no-hardhat', 'no-vest', 'safety vest']
            )
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            # Fallback to standard yolo11x
            self.model = YOLO("yolo11x.pt")
            self.class_names = self.model.names
            self.is_custom_ppe_model = False

    def predict(self, image_input, conf_threshold=0.40, iou_threshold=0.45, selected_classes=None):
        """
        Run inference on PIL Image or OpenCV BGR array.
        :return: list of detection dicts: [{box: [x1,y1,x2,y2], class_name, confidence, is_violation, type}]
        """
        if self.model is None:
            return []

        # Convert PIL to numpy if needed
        if isinstance(image_input, Image.Image):
            img_np = np.array(image_input)
        else:
            img_np = image_input

        # Run Ultralytics YOLO inference
        results = self.model.predict(
            source=img_np,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        detections = []
        if len(results) == 0:
            return detections

        res = results[0]
        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            raw_class_name = self.class_names.get(cls_id, f"Class {cls_id}")

            # Determine PPE details
            if self.is_custom_ppe_model:
                raw_lower = raw_class_name.lower()
                # Clean class mapping for display
                if raw_lower in ["helmet", "hardhat"]:
                    class_name = "Hardhat"
                    is_violation = False
                elif raw_lower in ["no-helmet", "no-hardhat", "without-helmet", "without_hardhat"]:
                    class_name = "NO-Hardhat"
                    is_violation = True
                elif raw_lower in ["vest", "safety vest", "safety-vest"]:
                    class_name = "Safety Vest"
                    is_violation = False
                elif raw_lower in ["no-vest", "without-vest", "without_vest"]:
                    class_name = "NO-Vest"
                    is_violation = True
                elif raw_lower in ["human", "person", "worker"]:
                    class_name = "Worker"
                    is_violation = False
                else:
                    class_name = raw_class_name
                    is_violation = "no-" in raw_lower or "missing" in raw_lower or "violation" in raw_lower or "without" in raw_lower
            else:
                # Using standard COCO model (e.g., base yolo11x.pt)
                if raw_class_name == "person":
                    class_name = "Worker"
                    # Heuristic Head/Torso PPE compliance check for base YOLO model
                    x1, y1, x2, y2 = xyxy
                    person_crop = img_np[max(0, y1):min(img_np.shape[0], y2), max(0, x1):min(img_np.shape[1], x2)]
                    
                    # Analyze head region (top 28% of person crop)
                    h, w = person_crop.shape[:2]
                    if h > 30 and w > 20:
                        head_crop = person_crop[0:int(h*0.28), :]
                        # Check for bright safety colors (Yellow/Orange/Blue/Red for Hardhats)
                        hsv = cv2.cvtColor(head_crop, cv2.COLOR_RGB2HSV if len(head_crop.shape)==3 else cv2.COLOR_BGR2HSV)
                        yellow_mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
                        blue_mask = cv2.inRange(hsv, (90, 80, 80), (130, 255, 255))
                        has_hardhat = (np.sum(yellow_mask) > 500) or (np.sum(blue_mask) > 500)
                        
                        # Upper torso (25% - 65% of crop)
                        torso_crop = person_crop[int(h*0.28):int(h*0.65), :]
                        torso_hsv = cv2.cvtColor(torso_crop, cv2.COLOR_RGB2HSV)
                        green_vest_mask = cv2.inRange(torso_hsv, (35, 80, 80), (85, 255, 255))
                        orange_vest_mask = cv2.inRange(torso_hsv, (5, 120, 120), (20, 255, 255))
                        has_vest = (np.sum(green_vest_mask) > 1000) or (np.sum(orange_vest_mask) > 1000)
                        
                        if has_hardhat and has_vest:
                            detections.append({
                                'box': [x1, y1, x1 + int(w*0.8), y1 + int(h*0.25)],
                                'class_name': 'Hardhat',
                                'confidence': min(0.98, conf + 0.05),
                                'is_violation': False,
                                'type': 'compliant'
                            })
                            detections.append({
                                'box': [x1, y1 + int(h*0.25), x2, y1 + int(h*0.65)],
                                'class_name': 'Safety Vest',
                                'confidence': min(0.95, conf + 0.02),
                                'is_violation': False,
                                'type': 'compliant'
                            })
                        elif not has_hardhat:
                            detections.append({
                                'box': [x1, y1, x1 + int(w*0.8), y1 + int(h*0.25)],
                                'class_name': 'NO-Hardhat',
                                'confidence': 0.88,
                                'is_violation': True,
                                'type': 'violation'
                            })
                        elif not has_vest:
                            detections.append({
                                'box': [x1, y1 + int(h*0.25), x2, y1 + int(h*0.65)],
                                'class_name': 'NO-Vest',
                                'confidence': 0.85,
                                'is_violation': True,
                                'type': 'violation'
                            })
                    is_violation = False
                else:
                    class_name = COCO_PPE_SIMULATION_MAP.get(raw_class_name, raw_class_name)
                    is_violation = False

            # Filter by selected classes if provided
            if selected_classes and class_name not in selected_classes and raw_class_name not in selected_classes:
                continue

            detections.append({
                'box': xyxy,
                'class_name': class_name,
                'confidence': conf,
                'is_violation': is_violation,
                'type': 'violation' if is_violation else ('compliant' if class_name in PPE_CLASSES and PPE_CLASSES[class_name]['type']=='compliant' else 'neutral')
            })

        return detections
