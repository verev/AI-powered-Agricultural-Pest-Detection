# AI-powered-Agricultural-Pest-Detection
Use ML and computer vision to identify and monitor crop pests, enabling farmers to take timely action and minimize yield losses.
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define a simple CNN model
def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Assuming binary classification (pest/no pest)
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Assuming you have a dataset directory structured as:
# dataset/
#   train/
#       pest/
#       no_pest/
#   validation/
#       pest/
#       no_pest/
def load_and_preprocess_data(train_dir, validation_dir, img_height, img_width):
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=20,
        class_mode='binary')
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_height, img_width),
        batch_size=20,
        class_mode='binary')
    
    return train_generator, validation_generator

# Model training
def train_model(model, train_generator, validation_generator, epochs):
    history = model.fit(
        train_generator,
        steps_per_epoch=100,  # Depends on your dataset size and batch size
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=50)  # Depends on your validation dataset size and batch size
    return history

# Plot training results
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    EPOCHS = 10
    
    model = create_model((IMG_HEIGHT, IMG_WIDTH, 3))
    train_dir = 'dataset/train'
    validation_dir = 'dataset/validation'
    train_generator, validation_generator = load_and_preprocess_data(train_dir, validation_dir, IMG_HEIGHT, IMG_WIDTH)
    history = train_model(model, train_generator, validation_generator, EPOCHS)
    plot_training_history(history)
