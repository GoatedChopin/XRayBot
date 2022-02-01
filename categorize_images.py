import os
import shutil
import pandas as pd
from parse_patient_csv import split_finding

subset = pd.read_csv("subset.csv")
patient_data = pd.read_csv("Data_Entry_2017.csv")
list_of_files = []
for (dirpath, dirnames, filenames) in os.walk(images_path):
    for filename in filenames:
        if filename.endswith('.png'): 
            list_of_files.append(os.sep.join([dirpath, filename]))

            
def categorize_files():
    for file in list_of_files:
        image_index = file.split("\\")[1]]
        findings = parse_conditions(split_finding(patient_data.loc[patient_data["Image_Index"]==image_index]["Finding"]))
        # Create a folder for the finding category if not existent, move the image to its matching folder.
        # Create a new folder manually for all the new categories
        root = os.getcwd() + r"\Categories"
        if not os.path.exists(root):
            os.makedirs(root)
        newpath = root + findings 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        shutil.move(file, newpath + r"\" + image_index)

def parse_conditions(conditions):
    conditions = {"Atelectasis":0, "Consolidation":1, "Infiltration":2, "Pneumothorax":3, "Edema":4, "Emphysema":5, "Fibrosis":6, "Effusion":7, "Pneumonia":8, "Pleural_Thickening":9, "Cardiomegaly":10, "Nodule":11, "Mass":12, "Hernia":13}
    rev_conditions = {v:k for k,v in dict_1.items()}
    output_list = []
    for i in range(len(conditions)):
        if(conditions[i] == 1):
            output.append(rev_conditions[i])
    output_string = ""
    for condition in output:
        output_string = output_string + condition
    return(output_string)
  
if __name__ == "__main__":
    categorize_files()
