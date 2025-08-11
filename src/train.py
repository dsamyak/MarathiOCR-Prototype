# src/train.py
import os, json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model, save_model, IMG_H, IMG_W
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
MODEL_OUT = os.path.join(os.path.dirname(__file__), '..', 'models', 'marathi_char_model.h5')
MAPPING_OUT = os.path.join(os.path.dirname(__file__), '..', 'models', 'class_mapping.json')
BATCH_SIZE = 16
EPOCHS = 10

def main():
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=8,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.06,
        zoom_range=0.08,
        validation_split=0.0
    )
    train_gen = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_H, IMG_W),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_H, IMG_W),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    num_classes = len(train_gen.class_indices)
    print("Detected classes:", num_classes)
    model = build_model(num_classes)
    model.summary()
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
    cb = [
        ModelCheckpoint(MODEL_OUT, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor='val_loss', patience=6, verbose=1, restore_best_weights=True)
    ]
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=cb
    )
    mapping = {v:k for k,v in train_gen.class_indices.items()}
    os.makedirs(os.path.dirname(MAPPING_OUT), exist_ok=True)
    with open(MAPPING_OUT, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False)
    print("Saved class mapping to", MAPPING_OUT)

if __name__ == "__main__":
    main()
