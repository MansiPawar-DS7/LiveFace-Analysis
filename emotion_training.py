import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50                                                #ResNet50: A pre-trained CNN model trained on ImageNet dataset
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models                                                      #layers, models: Used to build neural network layers
from tensorflow.keras.optimizers import Adam                                                     ##backend (K): Clears previous TensorFlow sessions
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint         #ReduceLROnPlateau: Reduces learning rate if validation loss stops improving,
from tensorflow.keras import backend as K                                                        #EarlyStopping: Stops training early to prevent overfitting
from sklearn.utils.class_weight import compute_class_weight                                      #ModelCheckpoint: Saves the best model automatically
import matplotlib.pyplot as plt                                                                  #compute_class_weight: Handles unbalanced datasets
                                                                                                 

IMG_SIZE = 224          #pixel
BATCH_SIZE = 32
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 50
CONFIDENCE_THRESHOLD = 0.35

K.clear_session()                      #This removes old models from memory becoz sometimes Tensorflow keep previous data in memory which can cause memory error in future

# DATA GENERATORS
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=30,                                 #this is Data Augmentation (meaning increase th amount of data by slightly modifying existing data, it helps model to learn better)
    zoom_range=0.2,                                    #rotate face, zoom, shift left/right, flip horizontally
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    "AffectNet/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',                   #emotion dataset is multi-class classification.
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    "AffectNet/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# CLASS WEIGHTS  --- Handling Imbalanced Dataset
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weights = dict(enumerate(weights))
emotion_labels = list(train_generator.class_indices.keys())   #it will use the labels during predictions like Happy, sad, fear, etc

# BUILD MODEL
base_model = ResNet50(
    weights='imagenet',
    include_top=False,                                        #Removes the original classification layer, becoz ImageNet predicts 1000 objects, but we want emotions
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

#Freeze ResNet 
base_model.trainable = False                                  #becoz we only want to train for our emotion layer first

model = models.Sequential([
    base_model,                                               #ResNet50 will extract deep features from face images
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),                              # Normalize data to improve Training Stability
    layers.Dropout(0.5),                                      #Randomly removes 50% neurons during training
    layers.Dense(512, activation='relu'),                     #relu helps learn complex patters, Fully connected layer with 512 neurons
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(len(emotion_labels), activation='softmax')   #Softmax outputs probability for each emotion
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),                        #learning rate (0.0001) --- small learning rate = stable training
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),      #Multi class classification, label_smoothing=0.1--- this will convert values into "0.9 and 0.1" instead of "0 and 1
    metrics=['accuracy']                 # label_smoothing converts labels like 1→0.9 and 0→0.1                                   
)

#CALLBACK
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    ModelCheckpoint('best_emotion_model.h5', save_best_only=True)
]

#PHASE 1: Training
print("Starting Phase 1: Training Head...")

history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weights,
    callbacks=callbacks
)

#PHASE 2: Fine Tuning
print("Starting Phase 2: Fine-Tuning Base Model...")

base_model.trainable = True

for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE2,
    class_weight=class_weights,
    callbacks=callbacks
)

#COMBINE HISTORY
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']

loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

epochs_range = range(1, len(acc) + 1)

#ACCURACY GRAPH
plt.figure(figsize=(8,5))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Accuracy Trend Across Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("accuracy_graph.png")
plt.show()

#LOSS GRAPH
plt.figure(figsize=(8,5))
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Loss Trend Across Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("loss_graph.png")
plt.show()

#PREDICTION FUNCTION
def predict_emotion_improved(face_img, model):

    try:
        img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
        img = np.expand_dims(img, axis=0).astype('float32')
        img = preprocess_input(img)

        preds = model.predict(img, verbose=0)[0]
        idx = np.argmax(preds)
        confidence = preds[idx]

        if confidence < CONFIDENCE_THRESHOLD:
            return "Processing...", confidence

        return emotion_labels[idx], confidence

    except Exception as e:
        return "Error", 0.0