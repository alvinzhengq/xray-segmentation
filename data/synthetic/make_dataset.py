import os
import pandas as pd

from load_dataset import convert_ct_xray

csv_path = "meta.csv"
dataset_path = "segmentator"
output_directory = "../training/all_ap_pa/"

df = pd.read_csv(csv_path, sep=";")

scans = df[
    df.apply(
        lambda row: row.astype(str).str.contains("thorax", case=False).any()
        and not row.astype(str).str.contains("abdomen", case=False).any(),
        axis=1,
    )
]

scan_ids = scans["image_id"].tolist()
print(f"{len(scan_ids)} scans containing ONLY 'Thorax'")

for i, folder in enumerate(scan_ids):
    sub_output_directory = os.path.join(output_directory, folder)
    if not os.path.exists(sub_output_directory):
        os.makedirs(sub_output_directory)

    ct_directory = os.path.join(dataset_path, folder)
    convert_ct_xray(ct_directory, sub_output_directory)

    print(f"Converted {folder} to x-ray format, {i/len(scan_ids) * 100:.2f}% of scans processed, {i+1} scans total.")
