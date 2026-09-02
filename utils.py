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

    for det in detections:
        x1, y1, x2, y2 = det['box']
        cls_name = det['class_name']
        conf = det['confidence']
        is_violation = det.get('is_violation', False)
        
        # Determine color
        class_meta = PPE_CLASSES.get(cls_name, {})
        hex_color = class_meta.get('color_hex', '#38BDF8' if not is_violation else '#EF4444')
        
        # Convert hex to RGBA
        hex_clean = hex_color.lstrip('#')
        rgb_color = tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
        fill_alpha = (*rgb_color, 40) # Semi-transparent fill box
        stroke_color = (*rgb_color, 245)
        
        box_width = max(2, int(height * 0.004))
        
        # 1. Semi-transparent fill
        draw.rectangle([x1, y1, x2, y2], fill=fill_alpha, outline=stroke_color, width=box_width)
        
        # 2. Draw corner accents for high-tech look
        corner_len = max(10, int(min(x2-x1, y2-y1) * 0.15))
        # Top-Left
        draw.line([(x1, y1), (x1 + corner_len, y1)], fill=stroke_color, width=box_width+2)
        draw.line([(x1, y1), (x1, y1 + corner_len)], fill=stroke_color, width=box_width+2)
        # Top-Right
        draw.line([(x2, y1), (x2 - corner_len, y1)], fill=stroke_color, width=box_width+2)
        draw.line([(x2, y1), (x2, y1 + corner_len)], fill=stroke_color, width=box_width+2)
        # Bottom-Left
        draw.line([(x1, y2), (x1 + corner_len, y2)], fill=stroke_color, width=box_width+2)
        draw.line([(x1, y2), (x1, y2 - corner_len)], fill=stroke_color, width=box_width+2)
        # Bottom-Right
        draw.line([(x2, y2), (x2 - corner_len, y2)], fill=stroke_color, width=box_width+2)
        draw.line([(x2, y2), (x2, y2 - corner_len)], fill=stroke_color, width=box_width+2)
        
        # 3. Label Tag Header
        label_text = f"{cls_name} {conf:.0%}"
        if is_violation:
            label_text = f"⚠️ {label_text}"
            
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        # Label position
        tag_y1 = max(0, y1 - text_h - 10)
        tag_y2 = max(text_h + 10, y1)
        tag_x1 = x1
        tag_x2 = x1 + text_w + 16
        
        # Tag Background Box
        tag_bg = (*rgb_color, 230)
        draw.rectangle([tag_x1, tag_y1, tag_x2, tag_y2], fill=tag_bg)
        
        # Text shadow & text
        draw.text((tag_x1 + 8, tag_y1 + 4), label_text, fill=(255, 255, 255, 255), font=font)

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
        cls_name = d['class_name']
        if cls_name in ['person', 'Worker', 'Person']:
            total_persons += 1
        elif cls_name in ['Hardhat', 'Helmet']:
            hardhat_count += 1
        elif cls_name in ['Safety Vest', 'Vest']:
            vest_count += 1
        elif cls_name == 'NO-Hardhat':
            no_hardhat_count += 1
        elif cls_name == 'NO-Vest':
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
