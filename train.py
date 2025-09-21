import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Kaggle downloaded dataset folder
DATASET_DIR = os.path.join(BASE_DIR, "plantvillage-dataset")
DATA_DIR = os.path.join(BASE_DIR, "datasheet")  # Folder where train/val will be created
MODEL_PATH = os.path.join(BASE_DIR, "leaf_disease_model.h5")

# -----------------------------
# Dataset classes
# -----------------------------
# Automatically detect class folders from Kaggle dataset
CLASSES = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
print("Detected classes:", CLASSES)

TRAIN_SPLIT = 0.8  # 80% training, 20% validation

# -----------------------------
# Create train/val folders
# -----------------------------
for cls in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, "train", cls), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "val", cls), exist_ok=True)

# -----------------------------
# Split dataset into train/val
# -----------------------------
for cls in CLASSES:
    cls_source = os.path.join(DATASET_DIR, cls)
    files = os.listdir(cls_source)
    random.shuffle(files)
    split_idx = int(len(files) * TRAIN_SPLIT)

    # Copy files to train
    for f in files[:split_idx]:
        shutil.copy(os.path.join(cls_source, f), os.path.join(DATA_DIR, "train", cls, f))
    # Copy files to val
    for f in files[split_idx:]:
        shutil.copy(os.path.join(cls_source, f), os.path.join(DATA_DIR, "val", cls, f))

print("Dataset prepared. Train/val folders created.")

# -----------------------------
# Data Generators
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# -----------------------------
# Build CNN Model
# -----------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# Train model
# -----------------------------
EPOCHS = 10
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# -----------------------------
# Save model
# -----------------------------
model.save(MODEL_PATH)
print(f"Training complete! Model saved at: {MODEL_PATH}")
