# MarathiOCR-Prototype (Colab-ready)

This package includes a full project scaffold for a Marathi/Devanagari handwritten character recognition prototype.
It contains training scripts, model code, a Flask demo, and instructions to run training in Google Colab (where TensorFlow is available).

Files of interest:
- src/train.py  : training script using ImageDataGenerator
- src/model.py  : CNN model architecture
- src/utils.py  : image preprocessing helper
- src/predict.py: single-image prediction helper
- src/webapp.py : Flask web demo (uses predict.py)
- templates/index.html : minimal web UI
- requirements.txt : Python packages required

This archive includes no real Devanagari dataset. Download a Devanagari dataset (Kaggle) and place it under data/train and data/val before training.
