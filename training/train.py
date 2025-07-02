import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = "./output"  # Use the output from generate_variants.py
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 64
SEED = 42

# Load dataset (grayscale, no augmentation)
train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset='training'
)
val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset='validation'
)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Model definition
model = models.Sequential([
    layers.Input(shape=(*IMAGE_SIZE, 1)),
    layers.Rescaling(1./255),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

model.save('model.keras')

# Plot training history
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig('training_history.png')
plt.show()

# 7. Make prediction on a single image (UPDATED FOR GRAYSCALE)
# def predict_image(image_path):
#     img = tf.keras.utils.load_img(
#         image_path,
#         target_size=IMAGE_SIZE,
#         color_mode='grayscale'  # CHANGED TO GRAYSCALE
#     )
#     img_array = tf.keras.utils.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)  # Create batch dimension

#     predictions = model.predict(img_array)
#     predicted_class = tf.argmax(predictions[0]).numpy()
#     confidence = tf.reduce_max(predictions[0]).numpy()
    
#     return predicted_class, confidence

# Example usage:
