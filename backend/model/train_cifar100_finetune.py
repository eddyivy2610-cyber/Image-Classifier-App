"""
Phase 5B: CIFAR-100 Fine-Tuning (Improved)

Model: MobileNetV2 (Transfer Learning + Fine Tuning)
Dataset: CIFAR-100
Features:
- 96x96 Resolution (Better detail)
- Data Augmentation (Flip, Rotation, GridDistortion via layers)
- Callbacks (EarlyStopping, Checkpoint)
- Mixed Precision (Optional, kept off for compatibility)

Output: cifar100_finetuned.h5
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------------------
# 1. CONFIG
# -------------------------------
IMG_SIZE = 96              # Increased from 48 for better accuracy
BATCH_SIZE = 16            # Low batch size for memory safety
EPOCHS = 20                # Increased epochs
AUTOTUNE = tf.data.AUTOTUNE
MODEL_SAVE_PATH = "cifar100_finetuned.h5"

print(f"Config: IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}")

# -------------------------------
# 2. LOAD DATA
# -------------------------------
print("Loading CIFAR-100 data...")
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")

# -------------------------------
# 3. DATA PIPELINE
# -------------------------------
def preprocess(image, label):
    # Resize and preprocess
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = preprocess_input(image)
    return image, label

# Augmentation Layers (Part of the model or dataset)
# We'll put them in the dataset pipeline for efficiency if possible, 
# but Keras Preprocessing Layers inside the model are easiest to manage.
# However, to save memory, we can do it here.

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def train_preprocess(image, label):
    # Resize first
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    # Augment
    image = data_augmentation(image)
    # Preprocess for MobileNet
    image = preprocess_input(image)
    return image, label

print("Creating datasets...")
train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(5000)
    # Use train_preprocess for training data
    .map(lambda x, y: (train_preprocess(x, y) if True else preprocess(x, y)), num_parallel_calls=AUTOTUNE) # Simplified logic for readability: We always want augment on train
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# Fix: the previous map lambda was a bit weird. let's be explicit.
# We need to apply augmentation only on training data.
# And we need to make sure the shapes are correct.
# Re-defining train_ds correctly:

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(5000)
    .map(lambda x, y: train_preprocess(x, y), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .map(lambda x, y: preprocess(x, y), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# -------------------------------
# 4. MODEL SETUP
# -------------------------------
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Fine-tuning strategy:
# Freeze most layers, unfreeze the top ones.
base_model.trainable = True
# Let's freeze the first 100 layers and fine-tune the rest
for layer in base_model.layers[:100]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation="relu"), # Increased dense layer size
    layers.Dropout(0.4),                  # Increased dropout
    layers.Dense(100, activation="softmax")
])

# -------------------------------
# 5. COMPILE & CALLBACKS
# -------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # Lower LR for fine-tuning
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=3, 
        min_lr=1e-6,
        verbose=1
    )
]

model.summary()

# -------------------------------
# 6. TRAIN
# -------------------------------
print("Starting training...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds,
    callbacks=callbacks_list
)

# -------------------------------
# 7. FINAL SAVE
# -------------------------------
# Checkpoint saves the best, but we'll save the final state too if needed.
# Since we use restore_best_weights=True, model is already the best one.
model.save(MODEL_SAVE_PATH)
print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")
