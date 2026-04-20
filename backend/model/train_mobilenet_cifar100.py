"""
Phase 4: CIFAR-100 Image Classification

Model: MobileNetV2 (Transfer Learning)
Dataset: CIFAR-100
Output: cifar100_model.h5
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------------------
# 1. SYSTEM CONFIG (LOW MEMORY)
# -------------------------------
IMG_SIZE = 48              # small resize to save memory
BATCH_SIZE = 8             # very small batch
EPOCHS = 10                # keep modest
AUTOTUNE = tf.data.AUTOTUNE

tf.keras.backend.clear_session()

# -------------------------------
# 2. LOAD CIFAR-100 DATA
# -------------------------------
(x_train, y_train), (x_test, y_test) = cifar100.load_data(
    label_mode="fine"
)

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)

# -------------------------------
# 3. DATA PIPELINE (STREAMING)
# -------------------------------
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = preprocess_input(image)
    return image, label

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(4000)
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# Free raw arrays from RAM
del x_train
del x_test

# -------------------------------
# 4. BUILD MODEL (TRANSFER LEARNING)
# -------------------------------
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained weights
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(100, activation="softmax")  # CIFAR-100
])

model.summary()

# -------------------------------
# 5. COMPILE MODEL
# -------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------
# 6. TRAIN MODEL
# -------------------------------
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds
)

# -------------------------------
# 7. EVALUATE
# -------------------------------
test_loss, test_acc = model.evaluate(test_ds)
print(f"Final CIFAR-100 Accuracy: {test_acc:.4f}")

# -------------------------------
# 8. SAVE MODEL
# -------------------------------
model.save("cifar100_model.h5")
print("Model saved as cifar100_model.h5")
