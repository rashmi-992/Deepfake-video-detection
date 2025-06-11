
# 🧠 Deepfake Video Detection Using ResNeXt + LSTM

This project is a complete pipeline for detecting deepfake videos using deep learning techniques. It includes video preprocessing, a hybrid CNN-LSTM model for spatiotemporal learning, training logic, and prediction with heatmap visualizations.

## 📂 Project Structure

```
deepfake-detection/
├── preprocessing_video.py       # Frame extraction and face cropping
├── model_and_train_csv_.py      # Dataset creation, model training & evaluation
├── predict.py                   # Inference with heatmap and confidence
├── Gobal_metadata.csv           # CSV with video labels: REAL or FAKE
└── checkpoint.pt                # Saved model checkpoint
```

## 🚀 Features

- ✅ End-to-end pipeline: From video preprocessing to prediction.
- 🎥 Face-based frame extraction for robust feature learning.
- 🧠 Hybrid model: ResNeXt CNN + LSTM for spatial-temporal learning.
- 📊 Training and validation with performance plots and confusion matrix.
- 🔥 Heatmap-based interpretability for predictions.

## 🛠️ Requirements

Make sure you're using **Google Colab GPU runtime** with the following libraries:

```bash
pip install face_recognition
pip install dlib
```

Also requires: `torch`, `torchvision`, `opencv-python`, `matplotlib`, `numpy`, `seaborn`, `pandas`, `sklearn`

## 📦 Dataset

- Videos are expected in `/content/drive/MyDrive/Dataset/`
- Ground truth labels in `Gobal_metadata.csv` with format: `filename,label` where label ∈ {REAL, FAKE}

## 🧼 1. Preprocessing

Extracts faces from videos and resizes them to `112x112`. Videos with too few frames are ignored.

```bash
python preprocessing_video.py
```

Outputs preprocessed videos to: `/content/drive/MyDrive/New_data/`

## 🏋️‍♂️ 2. Model Training

- Uses ResNeXt50 as a feature extractor
- LSTM layers capture temporal features
- Classifies video as REAL or FAKE

```bash
python model_and_train_csv_.py
```

Includes:
- Corrupted video filtering
- Data loader creation
- Model training with accuracy/loss plots
- Confusion matrix visualization

## 🔍 3. Inference & Visualization

Run prediction on new videos with confidence score and Class Activation Map (CAM)-based heatmap.

```bash
python predict.py
```

Sample output:
```
/content/drive/MyDrive/Dataset/id0_0003.mp4
Confidence: 97.35%
Prediction: FAKE
```

Also generates a heatmap overlay on the last frame.

## 🧠 Model Architecture

```text
Input Video (20 frames) 
     ↓
Face Detection & Resize (112x112)
     ↓
ResNeXt50 (CNN)
     ↓
LSTM (sequence modeling)
     ↓
Fully Connected → Softmax (REAL / FAKE)
```

## 📊 Results

- ✅ Average Accuracy: ~90% on validation set
- 📉 Loss and accuracy plotted per epoch
- 🔥 Heatmaps show regions influencing prediction

## 📝 Citation & Credits

- FaceForensics++ Dataset: https://github.com/ondyari/FaceForensics
- DFDC: https://www.kaggle.com/c/deepfake-detection-challenge
- Torchvision ResNeXt: https://pytorch.org/vision/stable/models.html

## 📌 Notes

- Training assumes your Google Drive is mounted and dataset paths are accessible.
- Works best with a GPU-enabled runtime.
- Code is modular and customizable for other face-based video classification tasks.
