"""
Please respect the license attached to this project found in the LICENSE file
if the LICENSE file is missing, please refer to the LICENSE found at this github repo below:
https://github.com/BinaryGears/KerasDeepFakeDetection/tree/main
"""

import pandas as pd
import os

training_path_fake = "images/train/fake_image"
training_path_real = "images/train/real_image"
validation_path_fake = "images/train/fake_image"
validation_path_real = "images/val/real_image"

files = os.listdir(training_path_fake)

filename_list = []
category_list = []

for f in files:
    filename_list.append(training_path_fake + f)
    category_list.append("fake")

data = {
    "filename": filename_list,
    "class": category_list
}

files = os.listdir(training_path_real)

for f in files:
    filename_list.append(training_path_real + f)
    category_list.append("real")

data = {
    "filename": filename_list,
    "class": category_list
}

training_df = pd.DataFrame(data)
training_df.to_csv("images/train/image_labels.csv", index = False)


files = os.listdir(validation_path_fake)

filename_list = []
category_list = []

for f in files:
    filename_list.append(validation_path_fake + f)
    category_list.append("fake")

data = {
    "filename": filename_list,
    "class": category_list
}

files = os.listdir(validation_path_real)

for f in files:
    filename_list.append(validation_path_real + f)
    category_list.append("real")

data = {
    "filename": filename_list,
    "class": category_list
}

training_df = pd.DataFrame(data)
training_df.to_csv("images/val/image_labels.csv", index = False)