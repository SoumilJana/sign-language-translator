# 🖐️ Sign Language Translator

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Integrated-orange?logo=google)
![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🌍 Overview

The **Sign Language Translator** is an AI-powered desktop application that converts hand signs into readable text in **real time**.
Developed using **OpenCV**, **MediaPipe**, and a **custom-trained neural network**, it promotes inclusivity by enabling smoother communication between the **hearing** and **hearing-impaired** communities.

> 🎯 Focused on accessibility
> 💡 Built for innovation
> ⚙️ Powered by AI

---

## 📑 Table of Contents

1. [Quick Start](#-quick-start)
2. [How It Works](#-how-it-works)
3. [Tech Stack](#-tech-stack)
4. [Training Scripts](#-training-scripts)
5. [Installation](#-installation)
6. [Author and Copyright](#-author-and-copyright)
7. [Acknowledgements](#-acknowledgements)

---

## ⚡ Quick Start

Run the following commands to set up and launch the application:

```bash
# Clone repository
git clone https://github.com/SoumilJana/sign-language-translator.git
cd sign-language-translator

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate  # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Run application
python app/main.py
```

---

## 🧠 How It Works

<p align="center">
  <img src="docs/infographic_pipeline.png" width="750" alt="Sign Language Translator Workflow Infographic"/>
</p>

1. **Hand Detection:** MediaPipe identifies and tracks 3D hand landmarks.
2. **Feature Extraction:** OpenCV processes frames and normalizes coordinates.
3. **Prediction:** The neural network model classifies gestures into letters or words.
4. **Output:** Recognized signs appear as readable text and can be converted to speech.

---

## 🧩 Tech Stack

| Component             | Technology        | Purpose                                 |
| --------------------- | ----------------- | --------------------------------------- |
| 🧠 Core Engine        | **Python 3.10**   | Primary programming environment         |
| 👋 Hand Tracking      | **MediaPipe**     | Landmark detection and gesture tracking |
| 🖼️ Vision Processing | **OpenCV**        | Image capture and frame processing      |
| 🎨 User Interface     | **Kivy / KivyMD** | Cross-platform GUI framework            |
| 🔉 Speech Output      | **pyttsx3**       | Text-to-speech support                  |

---

## 🧪 Training Scripts

Located in the `training/` folder, these scripts allow you to build and train your own gesture recognition model.

| Script                 | Description                                                                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **collect_data.py**    | Captures hand landmark data through webcam. Each gesture (like A, B, or space) is labeled and saved as `.npy` files for dataset creation.               |
| **convert_dataset.py** | Normalizes and converts collected landmark data into numerical arrays (`your_landmarks.npy`, `your_labels.npy`) suitable for model training.            |
| **train_model2.py**    | Trains an MLPClassifier on the processed dataset, evaluates accuracy, and saves the trained model as `model.p`.                                         |
| **runF2.py**           | Launches the live sign language translator. Loads the trained model, predicts gestures in real-time, displays text, and uses text-to-speech for output. |

---

## 💻 Installation

<details>
  <summary>Expand installation steps</summary>

```bash
# Clone the repository
git clone https://github.com/SoumilJana/sign-language-translator.git
cd sign-language-translator

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate  # macOS / Linux

# Install all dependencies
pip install -r requirements.txt

# Launch the application
python app/main.py
```

</details>

---

## 👨‍💻 Author and Copyright

> **Developed by:** [Soumil Jana](mailto:janasoumil1005@gmail.com)
> **All Rights Reserved.**

⚠️ **Usage Policy:**
This project is protected under copyright.
You may **study and learn** from the code for educational purposes only.
**Copying, redistribution, or modification** without explicit written permission is strictly prohibited.

---

## 🏁 Acknowledgements

* [MediaPipe](https://mediapipe.dev/) — Hand tracking and landmark detection
* [OpenCV](https://opencv.org/) — Image processing and vision utilities
* [Kivy](https://kivy.org/) — Modern GUI development for cross-platform applications

---

<p align="center">
  <b>“Technology is best when it brings people together.”</b><br>
  — Matt Mullenweg
</p>
