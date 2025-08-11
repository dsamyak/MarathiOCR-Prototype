# src/predict.py
import sys, json, numpy as np, os
from model import load_model
from utils import preprocess_image_for_model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'marathi_char_model.h5')
MAPPING_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'class_mapping.json')

def predict(img_path):
    model = load_model(MODEL_PATH)
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    img_arr = preprocess_image_for_model(img_path)
    img_arr = np.expand_dims(img_arr, axis=0)
    preds = model.predict(img_arr)
    idx = int(np.argmax(preds, axis=1)[0])
    label = mapping.get(str(idx), mapping.get(idx))
    return label

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    print(predict(sys.argv[1]))
