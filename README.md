#  Facial Emotion Recognition via MediaPipe Blendshapes

**End-to-end pipeline for anonymized facial emotion dataset creation and ML model training.**

Built as part of a Master's thesis at Universidad Politécnica de Cartagena (Industria 4.0, 2024).

---

## 🔍 Overview

This project addresses a core challenge in emotion recognition research: **collecting facial expression data without storing images**, protecting participant privacy while enabling effective ML model training.

The solution consists of two complementary tools:

| Tool | Stack | Purpose |
|---|---|---|
| **Dataset Capture App** | Flask · MediaPipe · JavaScript | Real-time blendshape extraction via webcam → anonymized CSV dataset |
| **ML Training App** | PyQt5 · TensorFlow · scikit-learn | Train, configure and evaluate ML models on blendshape datasets |

---

## 🔒 Privacy by Design

Most facial expression datasets store raw images — which are personal data under GDPR/LOPDGDD. This pipeline takes a different approach:

- MediaPipe's **Face Landmarker** extracts **52 blendshape values** per frame (numerical scores for facial deformations like smile, brow raise, jaw open)
- These values describe facial geometry **without storing any image**
- It is mathematically impossible to reconstruct the original face from blendshape vectors
- ~80–90 labeled samples are captured per emotion in a 5-second session

---

## 🏗️ Pipeline Architecture

```
Webcam (browser)
      ↓
MediaPipe Face Landmarker (JS)
      ↓
52 Blendshape values [0–1] per frame
      ↓
Flask backend → CSV dataset
      [ts, id, patologia, sexo, edad, 52 blendshapes, emocion]
      ↓
ML Training App (PyQt5)
      ↓
Trained model (.h5 / .pkl)
```

---

## 📁 Project Structure

```
├── App_captura_Blendshapes/
│   ├── app.py               # Flask backend — receives and stores blendshape data
│   ├── templates/
│   │   └── index.html       # Webcam interface with emotion buttons
│   └── static/
│       ├── script.js        # MediaPipe integration + blendshape capture logic
│       └── style.css
│
└── App_Entrenamiento_ML/
    ├── main.py              # Entry point — dataset import and model selection
    ├── mainwindow_ui.py     # Main window UI (PyQt5)
    ├── NN_Model.py          # Neural Network — config, training, evaluation
    ├── NN_ui.py             # Neural Network UI
    ├── KNN_Model.py         # K-Nearest Neighbors
    ├── KNN_ui.py
    ├── SVC_Model.py         # Support Vector Machine
    ├── SVC_ui.py
    ├── DT_Model.py          # Decision Tree
    └── DT_ui.py
```

---

## 🧠 Supported ML Models

All models are configurable from the GUI — no coding required:

- **Neural Network** (TensorFlow/Keras) — configurable layers, neurons, activation functions, optimizer, epochs
- **K-Nearest Neighbors** — configurable K
- **Support Vector Machine (SVC)** — configurable kernel and parameters
- **Decision Tree** — configurable depth and criteria

### Model Input
Each sample consists of **54 features**:
- `sexo` (encoded: 0=female, 1=male)
- `edad` (age)
- 52 MediaPipe blendshape scores

### Target Classes
`Neutral (0)` · `Enfadado/Angry (1)` · `Triste/Sad (2)` · `Feliz/Happy (3)`

---

## 📊 Evaluation Output

After training, the app displays:
- Training & validation loss/accuracy curves
- Confusion matrix (seaborn heatmap)
- ROC curves per class with AUC scores
- Full classification report (precision, recall, F1)
- Permutation feature importance (Neural Network)
- Model download as `.h5` (Keras) or `.pkl` (scikit-learn)

---

## ⚙️ Setup

### Dataset Capture App
```bash
cd App_captura_Blendshapes
pip install flask pandas
python app.py
# Open http://localhost:5000
```

### ML Training App
```bash
cd App_Entrenamiento_ML
pip install pyqt5 tensorflow scikit-learn matplotlib seaborn pandas scikeras
python main.py
```

> Tested with Python 3.12.3 and TensorFlow 2.17

---

## 📋 Dataset Structure

Generated CSV files follow this column order:

```
ts | id | patologia | sexo | edad | [52 blendshapes] | emocion
```

- `ts` — epoch timestamp in milliseconds (ensures no duplicate records)
- `id` — participant identifier (no personal data)
- `patologia` — optional: Ninguna / Parkinson / Hipertensión
- `sexo` — hombre / mujer
- `edad` — age
- 52 blendshape columns (browDownLeft, browInnerUp, mouthSmileLeft, jawOpen, etc.)
- `emocion` — Neutral / Feliz / Triste / Enfadado

---

## 🎯 Key Design Decisions

**Why blendshapes instead of images?**
Images require significantly more storage and compute, and constitute personal data under EU regulation. Blendshapes capture the same expressive information as numerical vectors — lightweight, fast to process, and inherently anonymized.

**Why a personalized dataset approach?**
Research showed that landmark patterns relevant to emotion classification differ significantly by age group and biological sex. A personalized dataset allows training models specialized to a specific individual — useful in assistive robotics and medical monitoring contexts.

**Why PyQt5 for the training app?**
To make ML model training accessible without requiring coding knowledge. The GUI abstracts away implementation details while exposing key hyperparameters (layers, neurons, activation, optimizer, epochs, train/test split).

---

## 📈 Results

Training on a personalized single-user dataset (~989 samples, 4 emotion classes):
- **Accuracy: 100%** on test set
- AUC: 1.00 for all 4 classes
- No overfitting observed across 10 epochs

> Note: personalized models are optimized for a specific individual and are not directly transferable to other users due to inter-individual variability in facial expression patterns.

---

## 🔬 Context

This project was developed as part of:
- **Master's Thesis** — Máster Universitario en Industria 4.0, UPCT (2024)
- **Research project** — Plataforma JUNO+ (TED2021-130942A-C22), Universidad Politécnica de Cartagena

---

## 👩‍💻 Author

**María Dolores Belchí Martínez**  
Electronics Engineer | AI & Computer Vision 
[LinkedIn](https://www.linkedin.com/in/maría-dolores-belchí-martínez) · [GitHub](https://github.com/mdbelchi)

---

## 📄 License

MIT License
