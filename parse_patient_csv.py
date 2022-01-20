import pandas as pd
import numpy as np

patient_csv = "Data_Entry_2017.csv"
patient_data = pd.read_csv(patient_csv)

conditions = {Atelectasis:0, Consolidation:1, Infiltration:2, Pneumothorax:3, Edema:4, Emphysema:5, Fibrosis:6, Effusion:7, Pneumonia:8, Pleural_thickening:9, Cardiomegaly:10, Nodule:11, Mass:12, Hernia:13}

def split_finding(finding):
  split = finding.split(r"|")
  cond_list = np.zeros(14)
  for condition in split:
    if (condition != "No Finding"):
      cond_list[conditions[condition]] = 1
  return(cond_list)

if __name__ == "__main__":
  pd.get_dummies(data = patient_data, columns = ["Patient Gender", "View Position"])
  # What happens after the get_dummies? I need to select the binary variables in the following section, rather than the original ones.
  
  columns = ["Image Index", "Patient Age", "Patient Gender", "View Position"]
  subset_data = patient_data[columns]
  subset_data["conditions"] = [split_finding(i) for i in patient_data["Finding Labels"]]
  pd.to_csv(r"\relevant_patient_data.csv")
