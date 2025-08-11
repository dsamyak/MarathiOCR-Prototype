# src/utils.py
import cv2
import numpy as np
IMG_H = 64
IMG_W = 64

def preprocess_image_for_model(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape
    scale = min(IMG_W/w, IMG_H/h)
    nw, nh = int(w*scale), int(h*scale)
    img_resized = cv2.resize(img, (nw, nh))
    canvas = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
    x_off = (IMG_W - nw)//2
    y_off = (IMG_H - nh)//2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = img_resized
    arr = canvas.astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=-1)
    return arr
