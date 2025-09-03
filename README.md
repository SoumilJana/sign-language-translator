# Sign Language Translator

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red)

---

## Overview

The **Sign Language Translator** is an AI-powered tool that converts hand signs into readable text.  
It uses **OpenCV** and **MediaPipe** for hand tracking and a **custom ML model** for recognizing letters and words from sign language gestures.

The goal is to improve accessibility and bridge communication gaps for the hearing-impaired community.

---

## Demo

![Demo GIF](docs/demo.gif)

*Real-time sign language recognition using a webcam feed.*

> ⚠️ **All Rights Reserved**  
> This project is the intellectual property of **Soumil Jana**.  
> You may **view and study** the code for personal learning purposes only.  
> **Copying, redistributing, or using** this code without explicit written permission is strictly prohibited.  
> Unauthorized use or plagiarism in assignments, projects, or publications will be considered a violation of copyright law.  
> For permissions or collaboration inquiries, contact: janasoumil1005@gmail.com

---

## Table of Contents

1. [Quick Start](#quick-start)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [Acknowledgements](#acknowledgements)

---

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/SoumilJana/sign-language-translator.git
cd sign-language-translator
````

2. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run inference:

```bash
python backend/inference.py
```

5. Run the backend model

```bash
python backend\runF2.py
```
6. Run the frontend skeleton

```bash
npm.cmd run dev
```

---

## Installation

<details>
  <summary>Click to expand full installation steps</summary>

```bash
# Clone the repository
git clone https://github.com/SoumilJana/sign-language-translator.git
cd sign-language-translator

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

</details>

---

## Usage

<details>
  <summary>Click to expand usage instructions</summary>

### 1. Training the Model

```bash
python training/train_model.py
```

* Place datasets in `data/kaggle_asl_dataset/` or preprocessed `.npy` files in `data/kaggle_asl_dataset_npy/`.

### 2. Running Inference

```bash
python backend/inference.py
```

* Displays webcam feed and outputs recognized letters/words.
* Results are saved to `outputs/output_sentence.txt`.

</details>

---

## Acknowledgements

* [MediaPipe](https://mediapipe.dev/) – Hand tracking
* [OpenCV](https://opencv.org/) – Computer vision utilities

---
