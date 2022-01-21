from xray_scanner import image_only
import numpy as np
from PIL import Image
import os
import pandas as pd

# load in images as a np array
# image_array = np.ndarray((1024,1024))
subset = pd.read_csv("subset.csv")

list_of_files = []
images_path = "Images"
for (dirpath, dirnames, filenames) in os.walk(images_path):
    for filename in filenames:
        if filename.endswith('.png'): 
            list_of_files.append(os.sep.join([dirpath, filename]))

image_list = []
conditions_list = []
for image_path in list_of_files:
    image = Image.open(image_path)
    image_list.append(np.asarray(image))
    # image_array = np.append(image_array, np.asarray(image), axis = 0)
    
image_array = np.asarray(image_list)

image_only = image_only()
image_only.model.fit(image_array)

