from xray_scanner import image_only
import numpy as np
from PIL import Image
import os
import pandas as pd

subset = pd.read_csv("subset.csv")

# Change images_path as needed
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
    conditions = np.asarray(subset.loc[subset["Image_Index"]==image_path.split("\\")[1]]["Conditions"].to_list()[0][1:-1].split(".")[0:14]).astype('float32')
    image_list.append(np.asarray(image))
    conditions_list.append(conditions)
    print(image)
    print(conditions)
    
image_array = np.asarray(image_list)
conditions_array = np.asarray(conditions_list)

# Do a train-test split that corresponds to the same indexes in both image_array and conditions_array

image_only = image_only()
image_only.model.fit(image_array, conditions_array)

