"""
Image Classification Using CNNs (Phase 1 Prototype)

Dataset: CIFAR-10
Framework: TensorFlow (Keras API)
Environment: Local CPU
"""

# -------------------------------
# 1. Import Required Libraries
# -------------------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10


# -------------------------------
# 2. Load Dataset
# -------------------------------

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)


# -------------------------------
# 3. Data Preprocessing
# -------------------------------

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# -------------------------------
# 4. Define CNN Architecture
# -------------------------------
model = models.Sequential([
    # Convolutional Layer 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    # Convolutional Layer 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten and Fully Connected Layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),

    # Output Layer (10 classes)
    layers.Dense(10, activation='softmax')
])

model.summary()


# -------------------------------
# 5. Compile the Model
# -------------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# -------------------------------
# 6. Train the Model
# -------------------------------
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)


# -------------------------------
# 7. Evaluate the Model
# -------------------------------
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")


# -------------------------------
# 8. Plot Training Performance
# -------------------------------
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')

plt.tight_layout()
plt.show()


# -------------------------------
# 9. Sample Predictions
# -------------------------------
class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

predictions = model.predict(x_test[:5])

for i in range(5):
    predicted_class = np.argmax(predictions[i])
    true_class = y_test[i][0]

    print(f"Image {i + 1}: "
          f"Predicted = {class_names[predicted_class]}, "
          f"Actual = {class_names[true_class]}")
