"""
Phase 3: Transfer Learning with MobileNetV2

Target System:
Dataset: CIFAR-10
Output: cifar_model.h5
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------------------
# 1. CONFIGURATION 
# -------------------------------
IMG_SIZE = 48          
BATCH_SIZE = 16        
EPOCHS = 8            
AUTOTUNE = tf.data.AUTOTUNE

# -------------------------------
# 2. LOAD DATA 
# -------------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# -------------------------------
# 3. TF.DATA PIPELINE (STREAMING)
# -------------------------------
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = preprocess_input(image)
    return image, label

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(5000)
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


del x_train
del x_test

# -------------------------------
# 4. BUILD TRANSFER LEARNING MODEL
# -------------------------------
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze entire base model
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation="softmax")
])

model.summary()

# -------------------------------
# 5. COMPILE
# -------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------
# 6. TRAIN
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
print(f"Final Test Accuracy: {test_acc:.4f}")

# -------------------------------
# 8. SAVE MODEL
# -------------------------------
model.save("cifar_model.h5")
print("Model saved as cifar_model.h5")
