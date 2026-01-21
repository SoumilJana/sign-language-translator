# 🖐️ SignVision - Sign Language Translator

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5.6-3178C6?logo=typescript)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Integrated-orange?logo=google)
![License](https://img.shields.io/badge/License-EULA-red)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🌍 Overview

**SignVision** is an AI-powered sign language translator that converts hand signs into readable text in **real time**. The project includes multiple versions:

| Version | Platform | Model | Status | Description |
|---------|----------|-------|--------|-------------|
| **Desktop v1** | Windows/Mac/Linux | MLP (sklearn) | ✅ Stable | Original release |
| **Desktop v2** | Windows/Mac/Linux | MLP + T5 Grammar | ⚠️ Experimental | With AI grammar correction |
| **Web App** | Browser | LSTM (ONNX) | ✅ Stable | Browser-based, no install needed |

> 🎯 Focused on accessibility  
> 💡 Built for innovation  
> ⚙️ Powered by AI

---

## ⚡ Quick Start

### Option 1: Desktop App v1 (Stable - Recommended)

```bash
# Clone repository
git clone https://github.com/SoumilJana/sign-language-translator.git
cd sign-language-translator

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Run stable version (from project root)
python app/main.py
```

### Option 2: Desktop App v2 (Experimental)

```bash
# Same setup as above, then:
cd desktop-app-v2
python main.py
```

> ⚠️ **Note:** v2 includes T5-based grammar correction which requires additional model downloads on first run.

### Option 3: Web Application

The web application is available in a separate folder. See [WebApp Setup](#web-application-setup) below.

---

## 📁 Project Structure

```
sign-language-translator/
├── README.md               # This file
├── LICENSE                 # EULA License
├── requirements.txt        # Python dependencies
│
├── app/                    # 🖥️ MAIN DESKTOP APP (v1 - Stable)
│   ├── main.py             # Run with: python app/main.py
│   ├── ui.kv               # Kivy UI layout
│   └── assets/             # Model & resources
│
├── desktop-app-v1/         # 📦 BACKUP of original GitHub release
│   └── (same structure as app/)
│
├── desktop-app-v2/         # 🧪 EXPERIMENTAL VERSION
│   ├── main.py             # With T5 grammar correction
│   ├── sentence_generator.py # AI sentence generation
│   ├── ui.kv
│   └── assets/
│
├── training/               # 🧠 TRAINING SCRIPTS (Desktop App)
│   ├── collect_data.py     # Capture hand landmarks
│   ├── convert_dataset.py  # Prepare training data
│   └── train_model2.py     # Train MLP classifier → model.p
│
├── newTrain/               # 🧠 TRAINING SCRIPTS (Web App - LSTM)
│   ├── scripts/
│   │   ├── collect_data.py
│   │   ├── train_model.py
│   │   └── convert_to_onnx.py
│   └── README.md
│
└── docs/                   # 📚 DOCUMENTATION
    ├── LSTM_TRAINING_PIPELINE.md
    └── T5_INTEGRATION.md
```

---

## 🧩 Tech Stack

### Desktop Application

| Component | Technology | Purpose |
|-----------|------------|---------|
| 🧠 Core | Python 3.10 | Primary language |
| 👋 Hand Tracking | MediaPipe | Landmark detection |
| 🖼️ Vision | OpenCV | Image processing |
| 🎨 UI | Kivy / KivyMD | Cross-platform GUI |
| 🔉 Speech | pyttsx3 | Text-to-speech |
| ✨ Grammar (v2) | T5 Transformers | AI sentence correction |

### Web Application

| Component | Technology | Purpose |
|-----------|------------|---------|
| ⚛️ Frontend | React 19 + TypeScript | Modern web UI |
| 🏗️ Build | Vite | Fast bundling |
| 🧠 ML | ONNX Runtime Web | Browser inference |
| 👋 Tracking | MediaPipe | Real-time detection |

---

## 🧪 Training Your Own Model

### For Desktop App (MLP)
Scripts in `training/` folder:
1. `collect_data.py` - Capture landmarks via webcam
2. `convert_dataset.py` - Convert to training arrays
3. `train_model2.py` - Train classifier → `model.p`

### For Web App (LSTM)
Scripts in `newTrain/scripts/`:
1. `collect_data.py` - Capture sequences
2. `train_model.py` - Train LSTM network
3. `convert_to_onnx.py` - Export → `model.onnx`

See `docs/LSTM_TRAINING_PIPELINE.md` for details.

---

## 🌐 Web Application Setup

The web app is in a separate `WebApp/` folder (not inside this repo).

```bash
cd ../WebApp    # Navigate to WebApp folder
npm install     # Install dependencies
npm run dev     # Start dev server at http://localhost:5173
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [LSTM Training](docs/LSTM_TRAINING_PIPELINE.md) | Web app model training |
| [T5 Integration](docs/T5_INTEGRATION.md) | Grammar correction in v2 |

---

## 👨‍💻 Author and License

**Developed by:** [Soumil Jana](mailto:janasoumil1005@gmail.com)

📧 Email: janasoumil1005@gmail.com  
🔗 GitHub: [@SoumilJana](https://github.com/SoumilJana)

### License (EULA)

- ✅ Personal, non-commercial use allowed
- ✅ Educational study permitted
- ❌ Commercial use prohibited without permission
- ❌ Redistribution/modification not allowed

See [LICENSE](LICENSE) for full terms.

---

## 🏁 Acknowledgements

* [MediaPipe](https://mediapipe.dev/) — Hand tracking
* [OpenCV](https://opencv.org/) — Image processing
* [Kivy](https://kivy.org/) — GUI framework
* [Hugging Face](https://huggingface.co/) — T5 model
* [ONNX Runtime](https://onnxruntime.ai/) — Browser ML

---

<p align="center">
  <b>"Technology is best when it brings people together."</b><br>
  — Matt Mullenweg
</p>
