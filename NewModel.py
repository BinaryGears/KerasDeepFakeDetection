"""
Please respect the license attached to this project found in the LICENSE file
if the LICENSE file is missing, please refer to the LICENSE found at this github repo below:
https://github.com/BinaryGears/KerasDeepFakeDetection/tree/main
"""

import pandas as pd
from keras.src.utils.module_utils import tensorflow
from tensorflow import keras
import visualkeras
from PIL import ImageFont

# Model parameters
num_classes = 10
input_shape = (256, 256, 3)
"Number of rows processed in one iteration of training"
batch_size = 32
"The number of times the layer is ran for a specific image"
epochs = 2

"Training data"
df1 = pd.read_csv("images/train/image_labels.csv")
datagen1 = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
train_generator = datagen1.flow_from_dataframe(
    dataframe=df1,
    directory="images/train/",
    x_col="filename",
    y_col="class",
    class_mode="categorical",
    target_size=(256, 256),
    batch_size=batch_size,
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
    target_size=(256, 256),
    batch_size=batch_size
)


"The entire network"
model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(8, (3,3)),
        keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                           alpha_regularizer=None,
                           alpha_constraint=None,
                           shared_axes=None,
                           ),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),

        keras.layers.Conv2D(8, (3, 3)),
        keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                           alpha_regularizer=None,
                           alpha_constraint=None,
                           shared_axes=None,
                           ),
        keras.layers.Conv2D(16, (3, 3)),
        keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                           alpha_regularizer=None,
                           alpha_constraint=None,
                           shared_axes=None,
                           ),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),

        keras.layers.Conv2D(32, (3, 3)),
        keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                           alpha_regularizer=None,
                           alpha_constraint=None,
                           shared_axes=None,
                           ),
        keras.layers.Conv2D(32, (3, 3)),
        keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                           alpha_regularizer=None,
                           alpha_constraint=None,
                           shared_axes=None,
                           ),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(64, (3, 3)),
        keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                           alpha_regularizer=None,
                           alpha_constraint=None,
                           shared_axes=None,
                           ),
        keras.layers.Conv2D(64, (3, 3)),
        keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                           alpha_regularizer=None,
                           alpha_constraint=None,
                           shared_axes=None,
                           ),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(128, (3, 3)),
        keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                           alpha_regularizer=None,
                           alpha_constraint=None,
                           shared_axes=None,
                           ),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(256, (3, 3)),
        keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                           alpha_regularizer=None,
                           alpha_constraint=None,
                           shared_axes=None,
                           ),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(32),
        keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                           alpha_regularizer=None,
                           alpha_constraint=None,
                           shared_axes=None,
                           ),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(16),
        keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                           alpha_regularizer=None,
                           alpha_constraint=None,
                           shared_axes=None,
                           ),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(16),
        keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                           alpha_regularizer=None,
                           alpha_constraint=None,
                           shared_axes=None,
                           ),
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

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath="model_at_epoch_{epoch}.keras", monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)
]


train = model.fit(train_generator,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=validation_generator,
          )

"""TO HERE:"""



"""
print_out_GUI = model.summary()
"""
"""
font = ImageFont.truetype("arial.ttf",32)
visualkeras.layered_view(model, font=font, to_file='outputLegend.png', legend=True)
visualkeras.layered_view(model, font=font, to_file='outputLegendDim.png', legend=True, show_dimension=True)
model.summary()
"""