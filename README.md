# 🛡️ SmartSurveill — Industrial AI Vision Sentinel

**The Next Generation of Intelligent Autonomous Surveillance for Enterprise Facilities**

SmartSurveill is a high-performance, industrial-grade AI surveillance platform designed for real-time safety compliance, perimeter security, and operational intelligence. Powered by YOLOv8 and a modern Flask-based micro-services architecture, it provides an all-in-one Command Center for modern facilities.

<p align="center">
  <img src="https://img.shields.io/badge/Status-Production--Ready-success?style=for-the-badge&logo=checkmarx" alt="Status">
  <img src="https://img.shields.io/badge/AI-YOLOv8%20|%20DeepSORT-blueviolet?style=for-the-badge&logo=openai" alt="AI">
  <img src="https://img.shields.io/badge/Efficiency-Offline--First-orange?style=for-the-badge&logo=speedtest" alt="Offline">
  <img src="https://img.shields.io/badge/Security-Enterprise--Grade-blue?style=for-the-badge&logo=security" alt="Security">
</p>

---

## 🛠️ Performance Tech Stack & Technologies Built With

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite" />
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5" />
  <img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white" alt="CSS3" />
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript" />
  <img src="https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white" alt="Chart.js" />
</p>

| Component | Technology |
| :--- | :--- |
| **Core AI** | YOLOv8 (Nano/Small), DeepSORT (Tracking), EasyOCR |
| **Hardware** | psutil (Real-time CPU/RAM/Disk Monitoring) |
| **Server** | Flask (Python 3.10+), Multi-threaded Camera Streams |
| **Database** | SQLite3 (Persistent DB), JSON Configuration |
| **Visualization** | Chart.js (Interactive Analytics), Matplotlib |
| **UI/UX** | Poppins Typography, CSS3 Modern Tokens, FontAwesome 6 |

---

## 🚀 Enterprise Command Center Features
  
### 🖥️ Professional UI/UX
- **Advanced Sidebar Navigation**: Collapsible, icon-driven sidebar with professional hover effects and active state tracking.
- **Multi-Camera Quick Access**: Top-tier control panel for starting, stopping, and monitoring individual camera nodes (Gate, Room, Parking).
- **2x2 Live Grid**: High-performance video streaming with resizable panels and fullscreen support.
- **Theme Engine**: Seamless switching between professional Light and Industrial Dark modes.

### 🛡️ Safety & Compliance Monitoring
- **PPE Detection**: Real-time monitoring for safety helmets and person orientation.
- **Vehicle Intelligence**: Automatic Number Plate Recognition (ANPR) and precise km/h speed estimation.
- **Driver Monitoring**: Seatbelt compliance and mobile usage detection while driving.
- **Distraction Detection**: Identification of personnel talking on phones while walking.
- **Sleeping On Duty**: AI-driven inactivity tracking with 2-minute threshold alerting.
- **Gender & Population Classification**: Active multi-person demographic tracking for enclosed environments.

### 📍 Security & Perimeter Control
- **Pathway Restriction**: Polygonal zone monitoring with "Restricted Area Breach" alerts using Euclidean algorithms.
- **Night Sentinel**: Automated intrusion detection during scheduled night hours (10 PM - 6 AM).
- **Human Inactivity**: Monitoring for personnel lying down or staying motionless in sensitive zones.

### 📊 Business Intelligence & Reporting
- **Analytical Reports**: Visual insights using Chart.js, including hourly activity trends and vehicle distribution.
- **Smart Search**: Multi-dimensional filtering (Date, Type, Plate, Camera) with instant results.
- **Automated PDFs**: Daily performance reports generated and archived automatically, downloadable instantly.
- **Audit Trails**: Professional terminal-style system logs for execution auditing.

---

## 📂 System Architecture
The repository follows a clean-separation architecture, decoupling the heavyweight AI backend from the lightweight visual frontend.

```bash
SmartSurveill/
├── backend/            # 🧠 The Intelligent Core
│   ├── config/         # Camera, ROI, and Admin Settings (JSON)
│   ├── models/         # Local AI Weights (YOLO, OCR, Gender)
│   ├── reports/        # Automated PDF Generation & Archiving
│   ├── main.py         # Multi-threaded Flask API & Routing
│   └── detector.py     # Unified Vision Pipeline
├── frontend/           # 🎨 The Visual Command Center
│   ├── static/         # Premium Industrial CSS & JS Modules
│   └── templates/      # Jinja2 Layout-based Dashboard System
├── .gitattributes      # Git LFS configuration for AI models
├── requirements.txt    # Global dependency manifests
└── README.md           # Professional Documentation
```

---

## 🏁 Quick Start Guide

### 1️⃣ Environment Setup
```bash
git lfs install
git clone https://github.com/VishnuVardhanCodes/Smart-Surveillance.git
cd Smart-Surveillance
pip install -r requirements.txt
```

### 2️⃣ Configuration
Open `backend/config/cameras.json` to link your CCTV feeds or webcams. Adjust thresholds in the **Admin Settings Panel** via the dashboard UI.

### 3️⃣ Launch the Sentinel
```bash
python backend/main.py
```
🌐 **Dashboard URL:** `http://localhost:5000`

---

## 🛡️ Security & Privacy Assurance
- **Zero Cloud Dependence**: All video frames are processed locally; no raw video leaves your network.
- **Offline-First**: Designed for industrial zones with limited internet connectivity.
- **System Stability**: Real-time health monitoring ensures the AI engine runs within hardware limits.

---
**Crafted for Innovation by [VishnuVardhanCodes](https://github.com/VishnuVardhanCodes) | ITC Internship 2026**
**Corporate Identity: ITC PSPD BCM Division**
