import pandas as pd
import numpy as np
import cv2
import os

csv_path = "A_Z Handwritten Data.csv"   # 改成你的檔名
output_dir = "az_dataset"

os.makedirs(output_dir, exist_ok=True)

data = pd.read_csv(csv_path, header=None)

labels = data.iloc[:, 0]
images = data.iloc[:, 1:]

for i in range(len(data)):
    label_idx = int(labels.iloc[i])      # 0~25
    label = chr(label_idx + ord('A'))    # 0->A, 1->B ...

    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    img = images.iloc[i].values.reshape(28, 28).astype(np.uint8)

    filename = os.path.join(label_dir, f"{i}.png")
    cv2.imwrite(filename, img)

    if i % 10000 == 0:
        print(f"{i} done")

print("Done!")