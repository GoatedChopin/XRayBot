# Imports
# TensorFlow and tf.keras
import tensorflow.keras as keras

# Notes
"""
Each image is 1024 x 1024 in size, so the input layer will need to accommodate that fact.

Although a Neural Network creating predictions based on the picture alone is perfectly possible, 
I'd prefer to build one that also takes into account Patient Age, Gender, and the View Position of the XRay.

Additionally, the output layer will need to be 1 x 14, with one slot for each condition, where a zeros matrix will represent No Findings.

Images should be fed to the model in a (image no., width, height) ndarray
"""

class image_only:
  def __init__(self):
    self.model = keras.Sequential([
        # keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(1024,1024)),
        keras.layers.Conv2D(16, (1,1), activation='relu', input_shape=(1024,1024,1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(32, (1,1), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (1,1), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(14)
    ])

    self.model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    
    
