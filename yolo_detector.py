"""
YOLO11x Model Wrapper and PPE Inference Engine.
Loads Ultralytics YOLO11 models (yolo11x.pt, yolo11l.pt, or user custom fine-tuned weights),
executes detection, handles class filtering, and analyzes PPE compliance.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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

    def predict(self, image_input, conf_threshold=0.25, iou_threshold=0.45, selected_classes=None):
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
                elif raw_lower in ["no-helmet", "no-hardhat", "without-helmet", "without_hardhat", "missing-hardhat", "missing-helmet"]:
                    class_name = "NO-Hardhat"
                    is_violation = True
                elif raw_lower in ["vest", "safety vest", "safety-vest"]:
                    class_name = "Safety Vest"
                    is_violation = False
                elif raw_lower in ["no-vest", "without-vest", "without_vest", "missing-vest"]:
                    class_name = "NO-Vest"
                    is_violation = True
                elif raw_lower in ["gloves", "glove", "safety-gloves", "safety gloves"]:
                    class_name = "Gloves"
                    is_violation = False
                elif raw_lower in ["no-gloves", "no-glove", "without-gloves", "without_gloves", "missing-gloves"]:
                    class_name = "NO-Gloves"
                    is_violation = True
                elif raw_lower in ["boots", "boot", "shoes", "safety-boots", "safety boots"]:
                    class_name = "Boots"
                    is_violation = False
                elif raw_lower in ["no-boots", "no-boot", "without-boots", "without_boots", "missing-boots"]:
                    class_name = "NO-Boots"
                    is_violation = True
                elif raw_lower in ["goggles", "glasses", "safety-goggles", "safety goggles"]:
                    class_name = "Goggles"
                    is_violation = False
                elif raw_lower in ["no-goggles", "no-goggle", "without-goggles", "without_goggles", "missing-goggles"]:
                    class_name = "NO-Goggles"
                    is_violation = True
                elif raw_lower in ["mask", "face-mask", "protective-mask"]:
                    class_name = "Mask"
                    is_violation = False
                elif raw_lower in ["no-mask", "without-mask", "without_mask", "missing-mask"]:
                    class_name = "NO-Mask"
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
                    
                    if person_crop.size > 0:
                        h, w = person_crop.shape[:2]
                        if h > 30 and w > 20:
                            head_crop = person_crop[0:int(h*0.28), :]
                            if head_crop.size > 0:
                                hsv = cv2.cvtColor(head_crop, cv2.COLOR_RGB2HSV if len(head_crop.shape)==3 else cv2.COLOR_BGR2HSV)
                                yellow_mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
                                blue_mask = cv2.inRange(hsv, (90, 80, 80), (130, 255, 255))
                                has_hardhat = (np.sum(yellow_mask) > 500) or (np.sum(blue_mask) > 500)
                            else:
                                has_hardhat = False
                                
                            torso_crop = person_crop[int(h*0.28):int(h*0.65), :]
                            if torso_crop.size > 0:
                                torso_hsv = cv2.cvtColor(torso_crop, cv2.COLOR_RGB2HSV if len(torso_crop.shape)==3 else cv2.COLOR_BGR2HSV)
                                green_vest_mask = cv2.inRange(torso_hsv, (35, 80, 80), (85, 255, 255))
                                orange_vest_mask = cv2.inRange(torso_hsv, (5, 120, 120), (20, 255, 255))
                                has_vest = (np.sum(green_vest_mask) > 1000) or (np.sum(orange_vest_mask) > 1000)
                            else:
                                has_vest = False
                                
                            if has_hardhat:
                                detections.append({
                                    'box': [x1, y1, x1 + int(w*0.8), y1 + int(h*0.25)],
                                    'class_name': 'Hardhat',
                                    'confidence': min(0.98, conf + 0.05),
                                    'is_violation': False,
                                    'type': 'compliant'
                                })
                            else:
                                detections.append({
                                    'box': [x1, y1, x1 + int(w*0.8), y1 + int(h*0.25)],
                                    'class_name': 'NO-Hardhat',
                                    'confidence': 0.88,
                                    'is_violation': True,
                                    'type': 'violation'
                                })

                            if has_vest:
                                detections.append({
                                    'box': [x1, y1 + int(h*0.25), x2, y1 + int(h*0.65)],
                                    'class_name': 'Safety Vest',
                                    'confidence': min(0.95, conf + 0.02),
                                    'is_violation': False,
                                    'type': 'compliant'
                                })
                            else:
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

            detections.append({
                'box': xyxy,
                'class_name': class_name,
                'confidence': conf,
                'is_violation': is_violation,
                'type': 'violation' if is_violation else ('compliant' if class_name in PPE_CLASSES and PPE_CLASSES[class_name]['type']=='compliant' else 'neutral')
            })

        # Apply selected_classes filter across all detections
        if selected_classes:
            allowed_set = set(selected_classes)
            detections = [
                d for d in detections
                if d['class_name'] in allowed_set or d['class_name'].lower() in [s.lower() for s in allowed_set]
            ]

        # Filter duplicates and resolve conflicting detections (e.g., Hardhat vs NO-Hardhat on same head)
        return self._clean_detections(detections)

    def _clean_detections(self, detections):
        if not detections:
            return []

        def compute_iou(box1, box2):
            x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
            x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            if inter == 0:
                return 0.0
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - inter
            return inter / float(union) if union > 0 else 0.0

        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        cleaned = []

        for det in sorted_dets:
            b1 = det['box']
            c1 = det['class_name']
            conf1 = det['confidence']

            # Suppress very low confidence violation false positives if compliant item exists
            if det.get('is_violation', False) and conf1 < 0.28:
                # Check if high confidence compliant box exists near it
                has_comp = any(
                    not d.get('is_violation', False) and compute_iou(b1, d['box']) > 0.25
                    for d in cleaned
                )
                if has_comp:
                    continue

            keep = True
            for existing in cleaned:
                b2 = existing['box']
                c2 = existing['class_name']
                iou = compute_iou(b1, b2)

                # 1. High overlap same/similar class duplicate
                if iou > 0.40 and (c1 == c2 or (c1 in ['Worker', 'person', 'human'] and c2 in ['Worker', 'person', 'human'])):
                    keep = False
                    break

                # 2. Hardhat vs NO-Hardhat conflict
                if iou > 0.25 and {c1, c2} in [{"Hardhat", "NO-Hardhat"}, {"helmet", "no-helmet"}]:
                    keep = False
                    break

                # 3. Vest vs NO-Vest conflict
                if iou > 0.25 and {c1, c2} in [{"Safety Vest", "NO-Vest"}, {"vest", "no-vest"}]:
                    keep = False
                    break

            if keep:
                cleaned.append(det)

        return cleaned
