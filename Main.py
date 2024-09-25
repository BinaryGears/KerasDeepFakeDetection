"""CNN is based on the structure proposed in this paper: https://doi.org/10.1109/ACCESS.2023.3251417"""

"""
Please respect the license attached to this project found in the LICENSE file
if the LICENSE file is missing, please refer to the LICENSE found at this github repo below: 
https://github.com/BinaryGears/KerasDeepFakeDetection/tree/main
"""
import pandas as pd
import os
from tensorflow import keras


# Model parameters
num_classes = 10
input_shape = (160, 160, 3)

"Training data"
df1 = pd.read_csv("images/train/image_labels.csv")
datagen1 = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)
train_generator = datagen1.flow_from_dataframe(
    dataframe=df1,
    directory="images/train/",
    x_col="filename",
    y_col="class",
    class_mode="categorical",
    target_size=(160, 160),
    batch_size=64
)

"Validation data"
df2 = pd.read_csv("images/val/image_labels.csv")
datagen2 = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)
validation_generator = datagen2.flow_from_dataframe(
    dataframe=df2,
    directory="images/val/",
    x_col="filename",
    y_col="class",
    class_mode="categorical",
    target_size=(160, 160),
    batch_size=64
)

"The entire network"
model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(8, (3,3), activation='leaky_relu'),
        keras.layers.BatchNormalization(),

        keras.layers.Conv2D(16, (3, 3), activation='leaky_relu'),
        keras.layers.Conv2D(16, (3, 3), activation='leaky_relu'),
        keras.layers.BatchNormalization(),
        keras.layers.AveragePooling2D((2,2)),

        keras.layers.Conv2D(32, (3, 3), activation='leaky_relu'),
        keras.layers.Conv2D(32, (3, 3), activation='leaky_relu'),
        keras.layers.Conv2D(32, (3, 3), activation='leaky_relu'),
        keras.layers.BatchNormalization(),
        keras.layers.AveragePooling2D((2, 2)),

        keras.layers.Conv2D(64, (3, 3), activation='leaky_relu'),
        keras.layers.Conv2D(64, (3, 3), activation='leaky_relu'),
        keras.layers.Conv2D(64, (3, 3), activation='leaky_relu'),
        keras.layers.Conv2D(64, (3, 3), activation='leaky_relu'),
        keras.layers.BatchNormalization(),
        keras.layers.AveragePooling2D((2, 2)),

        keras.layers.Conv2D(128, (3, 3), activation='leaky_relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(256, (3, 3), activation='leaky_relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(32),
        keras.layers.Activation(activation='leaky_relu'),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(16),
        keras.layers.Activation(activation='leaky_relu'),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(16),
        keras.layers.Activation(activation='leaky_relu'),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(2),
        keras.layers.Activation(activation='sigmoid')
    ]
)

"""The stuff in here is just kind of guesswork for now FROM:"""
model.compile(
    loss=keras.losses.CategoricalCrossentropy,
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.CategoricalAccuracy(name="acc"),
    ],
)

"Number of rows processed in one iteration of training"
batch_size = 64
"The number of times the layer is ran for a specific image"
epochs = 1

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath="model_at_epoch_{epoch}.keras"),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
]


train = model.fit(train_generator,
          epochs=epochs,
          batch_size=batch_size,
          )

val = model.fit(validation_generator,
                epochs=epochs,
                batch_size=batch_size,
                )

"""TO HERE:"""



"""
print_out_GUI = model.summary()
"""

model.summary()