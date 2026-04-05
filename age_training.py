import tensorflow as tf                                                              #tensorflow: Main deep learning framework
from tensorflow.keras.preprocessing.image import ImageDataGenerator                  #ImageDataGenerator: Loads images and performs preprocessing & augmentation
from tensorflow.keras.applications import MobileNetV2                                #MobileNetV2: Pre-trained CNN model used for learning
from tensorflow.keras import layers, models                                          #layers, models: Used to build neural network layers

# Image generators
#preparing the training data
train_gen = ImageDataGenerator(         
    rescale=1./255,                             #pixel: 0-255 ---- coverting 0-1, this will help neural network to train faster and more stable
    validation_split=0.2,                       #validation / testing = 20% and Training = 80%
    horizontal_flip=True,                       
    zoom_range=0.2
)

val_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

#Loading training data
train_data = train_gen.flow_from_directory(
    "utkface_age",
    target_size=(224, 224),                    #image resize (MobileNet require this size)
    batch_size=32,                             # 32 images process at a time
    class_mode="categorical",                 
    subset="training"
)

#Loading Validation data ---- Validation data is used to check model performance during training 
val_data = val_gen.flow_from_directory(
    "utkface_age",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Build MobileNetV2 base
base_model = MobileNetV2(                        # MobileNetV2 is a lightweight CNN architecture
    weights="imagenet",                          #tells the model to use the features it learned from ImageNet
    include_top=False,
    input_shape=(224,224,3)                      #224x224 pixels with 3 color channels (RGB)
)

#freeze MobileNetV2
base_model.trainable = False                    #MobileNet layers will NOT update during training, Only the new layers we add will train Because MobileNet already learned useful features


# Number of age classes
num_classes = train_data.num_classes            #it counts how many age groups exist

# Build model
model = models.Sequential([
    base_model,                                        #it extract features
    layers.GlobalAveragePooling2D(),                  
    layers.BatchNormalization(),                       #normalize data   
    layers.Dense(128, activation="relu"),              
    layers.Dropout(0.5),                               #Randomly disables neurons during training to prevent overfitting
    layers.Dense(num_classes, activation="softmax")    #Gives probability for each age group and then find out high probability class
])

# Compile the model (Before fine-tunning)
model.compile(                        
    optimizer="adam",                                  #adjust weight to minimize loss
    loss="categorical_crossentropy",                   #used for multi class classification
    metrics=["accuracy"]                               #show predicted accuracy
)

# Train the Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30                                        # it will see the entire dataset 30 times
) 

# Fine-tuning
#Early layers detect basic features (edges, textures), which are already useful. Fine-tuning only the last 30 layers allows the model to learn complex facial patterns without losing general features.
base_model.trainable = True                       #unfreeze MobileNetV2
for layer in base_model.layers[:-30]:             #Only last 30 layers of MobileNet will train
    layer.trainable = False                       #first all layers will be frozed and last 30 layers
                                                  #we train the last 30 layers so the model can adapt to our specific task (age prediction)

#Recompile model    
model.compile(                                  
    optimizer=tf.keras.optimizers.Adam(1e-5),   # Uses a smaller learning rate for fine-tuning
    loss="categorical_crossentropy",            #A very small learning rate is used during fine-tuning so that the pretrained MobileNetV2 weights are adjusted slowly without destroying the knowledge learned from ImageNet
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs = 40                           #training again on 40 epoch to increase the accuracy
)

# Save model
model.save("training_age_model.h5")


#We compile the model twice because after changing the trainable layers for fine-tuning, the model must be recompiled so the optimizer can update the correct weights.

#during training it print : training accuracy, validation accuracy, loss, validation loss
