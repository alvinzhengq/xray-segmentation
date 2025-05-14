import torch
from load_directions import convert_ct_xray
import os


import pandas as pd

# Path to the metadata CSV file
csv_path = "Totalsegmentator_dataset_small_v201/meta.csv"  # Change this if your file is in a different location

df = pd.read_csv(csv_path, sep=";")
thorax_scans_upper = df[df.apply(lambda row: row.astype(str).str.contains("thorax-neck", case=False).any(), axis=1)]
thorax_scans_lower = df[df.apply(lambda row: row.astype(str).str.contains("neck-thorax-abdomen-pelvis", case=False).any(), axis=1)]
thorax_ids = thorax_scans_upper["image_id"].tolist() + thorax_scans_lower["image_id"].tolist()
# thorax_ids = thorax_scans_lower["image_id"].tolist()
thorax_scans_upper_list = thorax_scans_upper["image_id"].tolist()
thorax_scans_lower_list = thorax_scans_lower["image_id"].tolist()
print("Scan IDs containing 'neck':")
for scan_id in thorax_ids:
    print(scan_id)

dataset_path = "Totalsegmentator_dataset_small_v201"
output_directory = "xray_dataset_thorax"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
for folder in thorax_ids:
    sub_output_directory = os.path.join(output_directory, folder)
    if not os.path.exists(sub_output_directory):
        os.makedirs(sub_output_directory)
    ct_directory = os.path.join(dataset_path, folder)
    try:
        if folder in thorax_scans_upper_list:
            convert_ct_xray(ct_directory, sub_output_directory)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            convert_ct_xray(ct_directory, sub_output_directory, translations=torch.tensor([[0.0, 800.0, 150.0]], device=device))
    except Exception as e:
        print(f"Error processing {folder}: {e}")
        continue

    print(f"Converted {folder} to x-ray format.")