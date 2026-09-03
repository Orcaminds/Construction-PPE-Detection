"""
Configuration file for Construction Site PPE Detection System
Contains UI themes, PPE class mappings, color palettes, and default settings.
"""

import os

# App Information
APP_TITLE = "Construction Site Safety & PPE Detection System"
APP_SUBTITLE = "AI-Powered Real-Time Safety Compliance Monitoring using YOLO11x"
APP_ICON = "🦺"

# Default Model Settings
DEFAULT_MODEL = "best.pt" if os.path.exists("best.pt") else "yolo11x.pt"
AVAILABLE_MODELS = {
    "🎯 Custom Fine-Tuned PPE Model (best.pt)": "best.pt",
    "YOLO11x (Extra Large - Highest Accuracy)": "yolo11x.pt",
    "YOLO11l (Large - Fast & Accurate)": "yolo11l.pt",
    "YOLO11m (Medium - Balanced)": "yolo11m.pt",
    "YOLO11s (Small - Fast)": "yolo11s.pt",
    "YOLO11n (Nano - Realtime Ultra Fast)": "yolo11n.pt",
}

DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45

# Standard & Fine-tuned PPE Class Names
# Mapping IDs/Names to Compliant vs Violation status
PPE_CLASSES = {
    # Compliant Items (Distinct vibrant colors for each class)
    "Hardhat": {"type": "compliant", "label": "Hardhat / Helmet", "color_hex": "#10B981", "color_bgr": (129, 185, 16)},
    "helmet": {"type": "compliant", "label": "Hardhat / Helmet", "color_hex": "#10B981", "color_bgr": (129, 185, 16)},
    "Safety Vest": {"type": "compliant", "label": "Safety Vest", "color_hex": "#F97316", "color_bgr": (22, 115, 249)}, # Vibrant Orange
    "vest": {"type": "compliant", "label": "Safety Vest", "color_hex": "#F97316", "color_bgr": (22, 115, 249)},
    "Mask": {"type": "compliant", "label": "Protective Mask", "color_hex": "#0EA5E9", "color_bgr": (233, 165, 14)}, # Sky Blue
    "Goggles": {"type": "compliant", "label": "Safety Goggles", "color_hex": "#3B82F6", "color_bgr": (246, 130, 59)}, # Royal Blue
    "Gloves": {"type": "compliant", "label": "Safety Gloves", "color_hex": "#A855F7", "color_bgr": (247, 85, 168)}, # Neon Purple
    "Boots": {"type": "compliant", "label": "Safety Boots", "color_hex": "#6366F1", "color_bgr": (241, 102, 99)}, # Indigo
    
    # Non-Compliant / Violations ("NO-" missing gear classes with distinct warning colors)
    "NO-Hardhat": {"type": "violation", "severity": "CRITICAL", "label": "⚠️ NO Hardhat", "color_hex": "#EF4444", "color_bgr": (68, 68, 239)},
    "no-helmet": {"type": "violation", "severity": "CRITICAL", "label": "⚠️ NO Hardhat", "color_hex": "#EF4444", "color_bgr": (68, 68, 239)},
    "NO-Vest": {"type": "violation", "severity": "CRITICAL", "label": "⚠️ NO Safety Vest", "color_hex": "#FF3366", "color_bgr": (102, 51, 255)},
    "no-vest": {"type": "violation", "severity": "CRITICAL", "label": "⚠️ NO Safety Vest", "color_hex": "#FF3366", "color_bgr": (102, 51, 255)},
    "NO-Gloves": {"type": "violation", "severity": "WARNING", "label": "⚠️ NO Gloves", "color_hex": "#EAB308", "color_bgr": (8, 179, 234)},
    "no-gloves": {"type": "violation", "severity": "WARNING", "label": "⚠️ NO Gloves", "color_hex": "#EAB308", "color_bgr": (8, 179, 234)},
    "NO-Boots": {"type": "violation", "severity": "WARNING", "label": "⚠️ NO Boots", "color_hex": "#FF5252", "color_bgr": (82, 82, 255)},
    "no-boots": {"type": "violation", "severity": "WARNING", "label": "⚠️ NO Boots", "color_hex": "#FF5252", "color_bgr": (82, 82, 255)},
    "NO-Goggles": {"type": "violation", "severity": "WARNING", "label": "⚠️ NO Goggles", "color_hex": "#E11D48", "color_bgr": (72, 29, 225)},
    "no-goggles": {"type": "violation", "severity": "WARNING", "label": "⚠️ NO Goggles", "color_hex": "#E11D48", "color_bgr": (72, 29, 225)},
    "NO-Mask": {"type": "violation", "severity": "WARNING", "label": "⚠️ NO Mask", "color_hex": "#F59E0B", "color_bgr": (11, 158, 245)},
    "no-mask": {"type": "violation", "severity": "WARNING", "label": "⚠️ NO Mask", "color_hex": "#F59E0B", "color_bgr": (11, 158, 245)},
    
    # Workers / Persons (Electric Cyan)
    "Worker": {"type": "neutral", "label": "Worker / Person", "color_hex": "#00F0FF", "color_bgr": (255, 240, 0)},
    "person": {"type": "neutral", "label": "Worker / Person", "color_hex": "#00F0FF", "color_bgr": (255, 240, 0)},
    "human": {"type": "neutral", "label": "Worker / Person", "color_hex": "#00F0FF", "color_bgr": (255, 240, 0)},
}

# General COCO to PPE Synthetic Mapping (Fallback for base COCO YOLO11x model)
# If fine-tuned PPE weights are not loaded, standard YOLO11 detects persons/ties/backpacks/etc.
COCO_PPE_SIMULATION_MAP = {
    "person": "Worker",
    "backpack": "Equipment Bag",
    "handbag": "Tool Bag",
    "tie": "Safety Lanyard",
    "umbrella": "Safety Canopy"
}

# Custom Streamlit CSS Styling
CUSTOM_CSS = """
<style>
    /* Main Background & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0d1322 100%);
        color: #f3f4f6;
    }
    
    /* Header Banner Styling */
    .header-banner {
        background: linear-gradient(90deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 5px solid #3b82f6;
        padding: 24px 28px;
        border-radius: 16px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        margin-bottom: 24px;
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #38bdf8, #a7f3d0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .header-subtitle {
        color: #9ca3af;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.4);
    }
    
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 6px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #9ca3af;
        font-weight: 600;
    }
    
    /* Status Badges */
    .badge-safe {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-danger {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.6);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre;
        border-radius: 8px;
        font-weight: 600;
        color: #9ca3af;
        border: none !important;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    /* Code/Logs */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
</style>
"""
