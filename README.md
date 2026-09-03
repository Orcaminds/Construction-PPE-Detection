# 🦺 Construction Site Safety & PPE Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![YOLO11 Powered](https://img.shields.io/badge/YOLO11-Ultralytics-blue.svg)](https://github.com/ultralytics/ultralytics)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An **AI-Powered Real-Time Construction Site Safety & Personal Protective Equipment (PPE) Compliance Monitoring System** built with **Ultralytics YOLO11x**, **OpenCV**, and **Streamlit**.

This platform monitors construction sites via images, recorded video streams, and live webcams to ensure workers adhere to mandatory safety compliance rules (helmets, high-visibility vests, gloves, boots, goggles, and protective masks).

---

## 🌟 Key Features

- **🤖 Advanced AI Model Support**:
  - Support for **Ultralytics YOLO11x (Extra Large)** for ultra-high precision detection.
  - Integration with **Custom Fine-Tuned Weights (`best.pt`)** specifically trained for industrial PPE objects.
  - Drag-and-drop custom `.pt` model loader.

- **🛡️ Full PPE Gear & Violation Tracking**:
  - **Compliant Equipment**: Hardhat / Helmet, Safety Vest, Gloves, Boots, Goggles, Protective Mask.
  - **Automated Violation Alerts**: `NO-Hardhat`, `NO-Vest`, `NO-Gloves`, `NO-Boots`, `NO-Goggles`, `NO-Mask`.
  - **Worker PPE Compliance Engine**: Automatically inspects head and torso regions to catch missing safety gear.

- **🎨 High-Contrast Distinct Color Palette**:
  - 🩵 **Worker**: Electric Cyan (`#00F0FF`)
  - 🟩 **Hardhat / Helmet**: Emerald Green (`#10B981`)
  - 🟧 **Safety Vest**: Safety Orange (`#F97316`)
  - 🟥 **NO-Hardhat / Critical Violations**: Crimson Red (`#EF4444`)
  - 🟪 **NO-Vest Violation**: Hot Red-Pink (`#FF3366`)

- **🎯 Smart Non-Overlapping Bounding Box Layout**:
  - 2-Pass rendering engine with collision avoidance to prevent tag text truncation and label overlapping.
  - Multi-spectrum HSV color analysis for accurate detection of White, Yellow, Blue, Red, Orange, and Green hardhats.

- **📊 Comprehensive Safety Dashboard**:
  - Real-time KPI Metric Cards (Workers Detected, Compliance %, Helmets, Vests, Total Violations).
  - Interactive Plotly Pie and Bar Charts for site safety audits.
  - Instant CSV Incident Report Exporter.

- **🔊 Multi-Source Support & Alerts**:
  - Image Upload & Built-in Demo Scenarios.
  - Frame-by-frame Video Stream Processing (MP4, AVI, MOV).
  - Live Webcam Feed Capture.
  - HTML5 Audio Siren Alerts on Safety Violations.

---

## 📁 Repository Structure

```text
PPE_Detection/
│
├── app.py                  # Main Streamlit Web Application Interface
├── yolo_detector.py        # YOLO11 Wrapper, PPE Logic & Compliance Engine
├── utils.py                # Smart Bounding Box Renderer & Metrics Computation
├── config.py               # Color Schemes, Class Mappings & Theme Settings
├── best.pt                 # Custom Fine-Tuned PPE Model Weights
├── yolo11x.pt              # Base YOLO11 Extra Large Model Weights
├── requirements.txt        # Python Dependencies
├── packages.txt            # System Linux Dependencies for Streamlit Cloud
└── README.md               # Documentation
```

---

## ⚡ Quick Start / Local Installation

### Prerequisites
- **Python 3.10+** installed on your system.
- NVIDIA GPU recommended (optional; CPU inference supported).

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Orcaminds/Construction-PPE-Detection.git
   cd Construction-PPE-Detection
   ```

2. **Create a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## ☁️ Streamlit Cloud Deployment

This repository is fully configured for zero-setup deployment on **Streamlit Community Cloud**:

1. Fork/Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and connect your repository.
3. Set Main File Path to `app.py`.
4. Deploy! System packages (`libgl1`, `ffmpeg`) specified in `packages.txt` will automatically install on the Linux container.

---

## 📊 Analytics & Reporting

| Feature | Description |
| :--- | :--- |
| **Workers Count** | Total personnel detected on site |
| **Compliance Score** | `(Compliant Workers / Total Workers) * 100%` |
| **Incident Logging** | Timestamped audit log with confidence scores and bounding box coordinates |
| **CSV Export** | Download full incident report for site management compliance |

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

Developed by **Orcaminds** for Smart Construction & Industrial Safety Compliance.
