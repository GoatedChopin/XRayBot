from xray_scanner import image_only_from_generator
import numpy as np
from PIL import Image
import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

# Change images_path as needed
list_of_files = []
images_path = "Categories"
target_size = (1024, 1024)
train_transformer = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)
# target_size is up for change, depending on how large the images to train on the model ought to be.
training_generator = train_transformer.flow_from_directory(images_path, target_size=(256,256), batch_size=2, color_mode="grayscale", class_mode="categorical", subset="training", seed=42)
validation_generator = train_transformer.flow_from_directory(images_path, target_size=(256,256), batch_size=2, color_mode="grayscale", class_mode="categorical", subset="validation", seed=42)

num_classes = 3

image_only_from_generator = image_only_from_generator(num_classes)
image_only_from_generator.model.fit_generator(training_generator, validation_data = validation_generator, steps_per_epoch = (training_generator.n//training_generator.batch_size), validation_steps = (validation_generator.n//validation_generator.batch_size), epochs = 25)

