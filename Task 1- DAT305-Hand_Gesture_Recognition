#Step 1: Import Required Libraries

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

#Step 2: Define Helper Functions
# 1. Function to Load and Preprocess Images:
#Converts images to grayscale.
#Resizes them to a fixed size.
#Applies Gaussian blur.
#Normalizes pixel values.
def preprocess_image(image_path, size=(64, 64)):
    # Read the image 
    image = cv2.imread(image_path) 
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to specified size
    resized_image = cv2.resize(gray_image, size)
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
    
    # Normalize pixel values to [0, 1]
    normalized_image = blurred_image / 255.0
    
    return normalized_image

#Function to Load Dataset:
#Iterates over directories and processes all images.
#Stores images and their labels.
def load_dataset(data_dir, size=(64, 64)):
    images = []
    labels = []
    classes = sorted(os.listdir(data_dir))
    
    for label, gesture_class in enumerate(classes):
        gesture_path = os.path.join(data_dir, gesture_class)
        for image_name in os.listdir(gesture_path):
            image_path = os.path.join(gesture_path, image_name)
            try:
                processed_image = preprocess_image(image_path, size)
                images.append(processed_image)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    
    return np.array(images), np.array(labels), classes

#Step 3: Split Data into Training, Validation, and Testing Sets
# Path to dataset folder
data_dir = "C:/dat305/datasets/hand_gestures"

# Load dataset
images, labels, classes = load_dataset(data_dir, size=(64, 64))

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Add a channel dimension for CNNs
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

#Step 4: Visualize Sample Images (Exploratory Analysis)
def plot_sample_images(images, labels, classes, n=10):
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(classes[labels[i]])
        plt.axis('off')
    plt.show()

# Visualize 10 sample images from the training set
plot_sample_images(X_train, y_train, classes, n=10)

#Step 5: Build the CNN Model
#We will use TensorFlow/Keras to define and train the CNN.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Define the CNN model
def create_cnn_model(input_shape=(64, 64, 1), num_classes=10):
    model = Sequential()
    
    # Convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    # Flatten and Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Regularization
    
    model.add(Dense(num_classes, activation='softmax'))  # Output layer for classification
    
    return model

# Instantiate the model
input_shape = X_train.shape[1:]  # (64, 64, 1)
num_classes = len(classes)
model = create_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Summary of the model
model.summary()

#Step 6: Train the CNN
#Train the model on the training set and validate it on the validation set.
# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping]
)

#Step 7: Evaluate the Model
#Evaluate the model's performance on the test set.
# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

#Step 8: Visualize Training Performance
#Plot the training and validation accuracy/loss over epochs.
import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

plot_training_history(history)

#Step 9: Save the Model
#Save the trained model for future use.
model.save('hand_gesture_model.h5')
print("Model saved as 'hand_gesture_model.h5'.")

#Step 10: Evaluate the Model with Advanced Metrics
#1Confusion Matrix
#Use the confusion_matrix from sklearn to analyze predictions.
#2Precision, Recall, and F1-Score
#Use classification_report to compute these metrics for each class.
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class indices

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plot the Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

# Classification Report
report = classification_report(y_test, y_pred_classes, target_names=classes)
print("Classification Report:\n")
print(report)

#Step 11: Improve Model Generalization
#1.Data Augmentation
#To improve generalization, incorporate data augmentation during training.
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the data generator
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

# Fit the generator to the training data
datagen.fit(X_train)

# Train the model with augmented data
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=20,
    callbacks=[early_stopping]
)

#2. Regularization Techniques
#Add dropout and batch normalization layers to reduce overfitting:
#Increase dropout rate in fully connected layers.
#Use L2 regularization in dense layers.

#Step 12: Optimize the Model for Deployment
#To make the model lightweight and efficient for real-time applications:
#1.Convert to TensorFlow Lite:
# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('hand_gesture_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Model converted to TensorFlow Lite and saved as 'hand_gesture_model.tflite'.")

#2.Quantization:
#Optimize the model further for mobile or edge devices by applying quantization during conversion:
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_quantized = converter.convert()

# Save the quantized model
with open('hand_gesture_model_quantized.tflite', 'wb') as f:
    f.write(tflite_model_quantized)
print("Quantized TensorFlow Lite model saved as 'hand_gesture_model_quantized.tflite'.")
