"""
Please respect the license attached to this project found in the LICENSE file
if the LICENSE file is missing, please refer to the LICENSE found at this github repo below:
https://github.com/BinaryGears/KerasDeepFakeDetection/tree/main
"""

import pandas as pd
from keras.src.utils.module_utils import tensorflow
"""
import visualkeras
from PIL import ImageFont
"""

class Model:
    # Model parameters
    num_classes = 10
    input_shape = (256, 256, 3)
    "Number of rows processed in one iteration of training"
    batch_size = 64
    "The number of times the layer is ran for a specific image"
    epochs = 4

    "Training data"
    df1 = pd.read_csv("images/train/image_labels.csv")
    datagen1 = tensorflow.keras.preprocessing.image.ImageDataGenerator(
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
    datagen2 = tensorflow.keras.preprocessing.image.ImageDataGenerator(
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
    model = tensorflow.keras.Sequential(
        [
            tensorflow.keras.layers.Input(shape=input_shape),
            tensorflow.keras.layers.Conv2D(8, (3,3)),
            tensorflow.keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                                          alpha_regularizer=None,
                                          alpha_constraint=None,
                                          shared_axes=None,
                               ),
            tensorflow.keras.layers.BatchNormalization(),
            tensorflow.keras.layers.MaxPooling2D((2,2)),

            tensorflow.keras.layers.Conv2D(8, (3, 3)),
            tensorflow.keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                               alpha_regularizer=None,
                               alpha_constraint=None,
                               shared_axes=None,
                               ),
            tensorflow.keras.layers.Conv2D(16, (3, 3)),
            tensorflow.keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                               alpha_regularizer=None,
                               alpha_constraint=None,
                               shared_axes=None,
                               ),
            tensorflow.keras.layers.BatchNormalization(),
            tensorflow.keras.layers.MaxPooling2D((2,2)),

            tensorflow.keras.layers.Conv2D(32, (3, 3)),
            tensorflow.keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                               alpha_regularizer=None,
                               alpha_constraint=None,
                               shared_axes=None,
                               ),
            tensorflow.keras.layers.Conv2D(32, (3, 3)),
            tensorflow.keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                               alpha_regularizer=None,
                               alpha_constraint=None,
                               shared_axes=None,
                               ),
            tensorflow.keras.layers.BatchNormalization(),
            tensorflow.keras.layers.MaxPooling2D((2, 2)),

            tensorflow.keras.layers.Conv2D(64, (3, 3)),
            tensorflow.keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                               alpha_regularizer=None,
                               alpha_constraint=None,
                               shared_axes=None,
                               ),
            tensorflow.keras.layers.Conv2D(64, (3, 3)),
            tensorflow.keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                               alpha_regularizer=None,
                               alpha_constraint=None,
                               shared_axes=None,
                               ),
            tensorflow.keras.layers.BatchNormalization(),
            tensorflow.keras.layers.MaxPooling2D((2, 2)),

            tensorflow.keras.layers.Conv2D(128, (3, 3)),
            tensorflow.keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                               alpha_regularizer=None,
                               alpha_constraint=None,
                               shared_axes=None,
                               ),
            tensorflow.keras.layers.BatchNormalization(),
            tensorflow.keras.layers.MaxPooling2D((2, 2)),

            tensorflow.keras.layers.Conv2D(256, (3, 3)),
            tensorflow.keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                               alpha_regularizer=None,
                               alpha_constraint=None,
                               shared_axes=None,
                               ),
            tensorflow.keras.layers.BatchNormalization(),
            tensorflow.keras.layers.MaxPooling2D((2, 2)),

            tensorflow.keras.layers.Flatten(),
            tensorflow.keras.layers.Dropout(0.5),

            tensorflow.keras.layers.Dense(32),
            tensorflow.keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                               alpha_regularizer=None,
                               alpha_constraint=None,
                               shared_axes=None,
                               ),
            tensorflow.keras.layers.Dropout(0.5),

            tensorflow.keras.layers.Dense(16),
            tensorflow.keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                               alpha_regularizer=None,
                               alpha_constraint=None,
                               shared_axes=None,
                               ),
            tensorflow.keras.layers.Dropout(0.5),

            tensorflow.keras.layers.Dense(16),
            tensorflow.keras.layers.PReLU(alpha_initializer=tensorflow.initializers.constant(0.25),
                               alpha_regularizer=None,
                               alpha_constraint=None,
                               shared_axes=None,
                               ),
            tensorflow.keras.layers.Dropout(0.5),

            tensorflow.keras.layers.Dense(2),
            tensorflow.keras.layers.Activation(activation='sigmoid')
        ]
    )


    model.compile(
        loss=tensorflow.keras.losses.CategoricalCrossentropy(),
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[
            tensorflow.keras.metrics.CategoricalAccuracy(name="acc"),
        ],
    )

    model.save("modelfolder/model.hdf5", overwrite=True, save_format=None)
    model.save("modelfolder/model.keras", overwrite=True, save_format=None)


    model.fit(train_generator,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=validation_generator,
              )

    """
    model.summary()
    
    font = ImageFont.truetype("arial.ttf",32)
    visualkeras.layered_view(model, font=font, to_file='outputLegend.png', legend=True)
    visualkeras.layered_view(model, font=font, to_file='outputLegendDim.png', legend=True, show_dimension=True)
    """