"""
Utility functions for Construction Site PPE Detection System.
Includes custom visualization drawing, metric extraction, HTML audio alert generation,
and synthetic demo image/video generation for instant out-of-the-box testing.
"""

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import time

from config import PPE_CLASSES


def draw_bounding_boxes(image_input, detections):
    """
    Draw stylized bounding boxes and labels on image.
    :param image_input: PIL Image or numpy array (RGB/BGR)
    :param detections: List of dicts with keys: 'box', 'class_name', 'confidence', 'is_violation', 'type'
    :return: Processed PIL Image
    """
    if isinstance(image_input, np.ndarray):
        img_pil = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    else:
        img_pil = image_input.copy()

    draw = ImageDraw.Draw(img_pil, "RGBA")
    width, height = img_pil.size
    
    # Try to load a nice font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", max(14, int(height * 0.02)))
        font_small = ImageFont.truetype("arial.ttf", max(11, int(height * 0.015)))
    except IOError:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    placed_tags = []

    # Sort detections by box area descending so larger boxes (Workers) are drawn first
    sorted_dets = sorted(
        detections,
        key=lambda d: (int(d['box'][2]) - int(d['box'][0])) * (int(d['box'][3]) - int(d['box'][1])),
        reverse=True
    )

    # Pass 1: Draw bounding box rectangles & corner accents
    for det in sorted_dets:
        box = det['box']
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1, x2 = max(0, min(x1, x2)), min(width, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(height, max(y1, y2))
        
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        cls_name = det['class_name']
        is_violation = det.get('is_violation', False)
        
        class_meta = PPE_CLASSES.get(cls_name, {})
        hex_color = class_meta.get('color_hex', '#EF4444' if is_violation else '#00F0FF')
        
        hex_clean = hex_color.lstrip('#')
        rgb_color = tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
        fill_alpha = (*rgb_color, 25)
        stroke_color = (*rgb_color, 245)
        
        box_width = max(2, int(height * 0.0035))
        
        # Draw box rectangle & corners
        draw.rectangle([x1, y1, x2, y2], fill=fill_alpha, outline=stroke_color, width=box_width)
        
        corner_len = max(6, int(min(x2-x1, y2-y1) * 0.12))
        draw.line([(x1, y1), (x1 + corner_len, y1)], fill=stroke_color, width=box_width+2)
        draw.line([(x1, y1), (x1, y1 + corner_len)], fill=stroke_color, width=box_width+2)
        draw.line([(x2, y1), (x2 - corner_len, y1)], fill=stroke_color, width=box_width+2)
        draw.line([(x2, y1), (x2, y1 + corner_len)], fill=stroke_color, width=box_width+2)
        draw.line([(x1, y2), (x1 + corner_len, y2)], fill=stroke_color, width=box_width+2)
        draw.line([(x1, y2), (x1, y2 - corner_len)], fill=stroke_color, width=box_width+2)
        draw.line([(x2, y2), (x2 - corner_len, y2)], fill=stroke_color, width=box_width+2)
        draw.line([(x2, y2), (x2, y2 - corner_len)], fill=stroke_color, width=box_width+2)

    # Pass 2: Draw label tags with smart collision avoidance
    for det in sorted_dets:
        box = det['box']
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1, x2 = max(0, min(x1, x2)), min(width, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(height, max(y1, y2))
        
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        cls_name = det['class_name']
        conf = det['confidence']
        is_violation = det.get('is_violation', False)
        class_meta = PPE_CLASSES.get(cls_name, {})

        if is_violation:
            hex_color = class_meta.get('color_hex', '#EF4444')
            label_text = f"⚠️ {cls_name} {conf:.0%}"
        else:
            hex_color = class_meta.get('color_hex', '#00F0FF')
            label_text = f"{cls_name} {conf:.0%}"

        hex_clean = hex_color.lstrip('#')
        rgb_color = tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
        
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        tag_w = text_w + 14
        tag_h = text_h + 8

        # Candidate placement positions (clamped to prevent right-edge clipping)
        cand_x = max(2, min(width - tag_w - 4, x1))
        if y1 - tag_h - 2 >= 0:
            cand_y = y1 - tag_h - 2
        else:
            cand_y = y1 + 3

        # Check collision helper
        def has_collision(rx1, ry1, rx2, ry2):
            for px1, py1, px2, py2 in placed_tags:
                if not (rx2 <= px1 or rx1 >= px2 or ry2 <= py1 or ry1 >= py2):
                    return True
            return False

        cur_rx1, cur_ry1 = cand_x, cand_y
        cur_rx2, cur_ry2 = cur_rx1 + tag_w, cur_ry1 + tag_h

        # Shift tag down/left if colliding with another tag
        attempts = 0
        while has_collision(cur_rx1, cur_ry1, cur_rx2, cur_ry2) and attempts < 8:
            cur_ry1 += tag_h + 3
            cur_ry2 = cur_ry1 + tag_h
            if cur_ry2 > height - tag_h:
                cur_rx1 = max(2, cur_rx1 - 25)
                cur_rx2 = cur_rx1 + tag_w
                cur_ry1 = max(2, y1 - tag_h)
                cur_ry2 = cur_ry1 + tag_h
            attempts += 1

        placed_tags.append((cur_rx1, cur_ry1, cur_rx2, cur_ry2))

        # Tag background & text
        tag_bg = (*rgb_color, 240)
        draw.rectangle([cur_rx1, cur_ry1, cur_rx2, cur_ry2], fill=tag_bg)
        draw.text((cur_rx1 + 7, cur_ry1 + 3), label_text, fill=(255, 255, 255, 255), font=font)

    return img_pil


def compute_metrics(detections):
    """
    Computes count summaries from detection list.
    """
    total_persons = 0
    hardhat_count = 0
    vest_count = 0
    no_hardhat_count = 0
    no_vest_count = 0
    other_violations = 0
    
    for d in detections:
        cls_name = d['class_name'].lower()
        if cls_name in ['person', 'worker', 'human']:
            total_persons += 1
        elif cls_name in ['hardhat', 'helmet']:
            hardhat_count += 1
        elif cls_name in ['safety vest', 'vest']:
            vest_count += 1
        elif cls_name in ['no-hardhat', 'no-helmet']:
            no_hardhat_count += 1
        elif cls_name in ['no-vest']:
            no_vest_count += 1
        elif d.get('is_violation', False):
            other_violations += 1
            
    # Total workers estimated
    worker_count = max(total_persons, hardhat_count + no_hardhat_count, vest_count + no_vest_count)
    if worker_count == 0 and len(detections) > 0:
        worker_count = len(detections)
        
    total_violations = no_hardhat_count + no_vest_count + other_violations
    
    # Calculate safety compliance score
    if worker_count > 0:
        compliant_workers = max(0, worker_count - total_violations)
        compliance_pct = round((compliant_workers / worker_count) * 100, 1)
    else:
        compliance_pct = 100.0

    return {
        "worker_count": worker_count,
        "total_violations": total_violations,
        "hardhat_count": hardhat_count,
        "vest_count": vest_count,
        "no_hardhat_count": no_hardhat_count,
        "no_vest_count": no_vest_count,
        "other_violations": other_violations,
        "compliance_pct": compliance_pct,
        "status": "DANGER" if total_violations > 0 else "SAFE"
    }


def generate_audio_alert_html():
    """
    Generates a silent/alert audio beep in HTML5 format to alert user of violations.
    """
    # 0.5s 800Hz sound wave encoded to WAV
    sample_rate = 22050
    duration = 0.4
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Beep waveform
    wave = 0.5 * np.sin(2 * np.pi * 880 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t)
    audio_data = (wave * 32767).astype(np.int16)
    
    # Write WAV header into BytesIO
    buffer = io.BytesIO()
    # Simple WAV writer
    num_samples = len(audio_data)
    data_size = num_samples * 2
    header = bytearray()
    header.extend(b'RIFF')
    header.extend((36 + data_size).to_bytes(4, 'little'))
    header.extend(b'WAVEfmt ')
    header.extend((16).to_bytes(4, 'little'))
    header.extend((1).to_bytes(2, 'little')) # PCM
    header.extend((1).to_bytes(2, 'little')) # Mono
    header.extend((sample_rate).to_bytes(4, 'little'))
    header.extend((sample_rate * 2).to_bytes(4, 'little'))
    header.extend((2).to_bytes(2, 'little'))
    header.extend((16).to_bytes(2, 'little'))
    header.extend(b'data')
    header.extend((data_size).to_bytes(4, 'little'))
    
    buffer.write(header)
    buffer.write(audio_data.tobytes())
    b64_audio = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"""
    <audio autoplay style="display:none;">
        <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
    </audio>
    """


def create_demo_construction_images():
    """
    Creates high quality demo synthetic construction site images for testing.
    """
    images = {}
    
    # 1. Compliant Site Image (2 Workers with Helmets & Vests)
    img1 = Image.new('RGB', (800, 600), color='#1e293b')
    draw1 = ImageDraw.Draw(img1)
    
    # Background Construction Scaffolding
    draw1.rectangle([0, 450, 800, 600], fill='#334155') # Floor
    draw1.rectangle([100, 100, 130, 450], fill='#475569') # Pillar 1
    draw1.rectangle([670, 100, 700, 450], fill='#475569') # Pillar 2
    draw1.line([(100, 150), (700, 150)], fill='#64748b', width=8) # Beam
    draw1.line([(100, 300), (700, 300)], fill='#64748b', width=8) # Beam
    
    # Worker 1 (Left - Compliant)
    # Body (Blue shirt / Green Vest)
    draw1.rectangle([220, 260, 320, 460], fill='#10b981') # Vest
    draw1.ellipse([240, 170, 300, 230], fill='#fde047') # Hardhat yellow
    draw1.arc([235, 195, 305, 235], start=0, end=180, fill='#eab308', width=5)
    draw1.ellipse([245, 200, 295, 250], fill='#fdba74') # Head
    
    # Worker 2 (Right - Compliant)
    draw1.rectangle([480, 250, 580, 460], fill='#059669') # Vest
    draw1.ellipse([500, 160, 560, 220], fill='#3b82f6') # Hardhat blue
    draw1.ellipse([505, 190, 555, 240], fill='#fed7aa') # Head
    
    # Text overlay
    draw1.text((30, 30), "DEMO SCENARIO A: Fully Compliant Construction Site", fill="#94a3b8")
    images["Demo_Compliant_Site.jpg"] = img1
    
    # 2. Non-Compliant Site Image (Worker without Helmet & Vest)
    img2 = Image.new('RGB', (800, 600), color='#1e293b')
    draw2 = ImageDraw.Draw(img2)
    
    draw2.rectangle([0, 450, 800, 600], fill='#334155')
    draw2.line([(50, 200), (750, 200)], fill='#ef4444', width=4) # Warning barrier
    
    # Worker 1 (Center - Violation: No Helmet, No Vest)
    draw2.rectangle([340, 250, 440, 460], fill='#475569') # Dark t-shirt (No vest!)
    draw2.ellipse([360, 180, 420, 240], fill='#fca5a5') # Bare head (No helmet!)
    
    draw2.text((30, 30), "DEMO SCENARIO B: Safety Violation (Missing Helmet & Vest)", fill="#f87171")
    images["Demo_Violation_Site.jpg"] = img2
    
    return images
