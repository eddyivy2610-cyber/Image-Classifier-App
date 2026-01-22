"""
PHASE 4: Fine-Tuning MobileNetV2
Dataset: CIFAR-10
Output: cifar_model_v4.h5
Environment: CPU / Low RAM
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------------------
# 1. CONFIG
# -------------------------------
IMG_SIZE = 64
BATCH_SIZE = 8
EPOCHS_FROZEN = 5
EPOCHS_FINE = 5
AUTOTUNE = tf.data.AUTOTUNE

# -------------------------------
# 2. LOAD DATA
# -------------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# -------------------------------
# 3. DATA PIPELINE
# -------------------------------
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = preprocess_input(image)
    return image, label

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(3000)
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

del x_train, x_test

# -------------------------------
# 4. MODEL
# -------------------------------
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

# -------------------------------
# 5. TRAIN (FROZEN)
# -------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    epochs=EPOCHS_FROZEN,
    validation_data=test_ds
)

# -------------------------------
# 6. FINE-TUNING
# -------------------------------
base_model.trainable = True

for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    epochs=EPOCHS_FINE,
    validation_data=test_ds
)

# -------------------------------
# 7. EVALUATE
# -------------------------------
loss, acc = model.evaluate(test_ds)
print(f"Final Accuracy: {acc:.4f}")

# -------------------------------
# 8. SAVE
# -------------------------------
model.save("cifar_model_v4.h5")
print("Model saved as cifar_model_v4.h5")
