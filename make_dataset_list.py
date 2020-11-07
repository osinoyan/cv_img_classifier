import torch
import pandas as pd
import numpy as np
import scipy.io

origin_df = pd.read_csv('./data/training_labels.csv')
train = origin_df

# write train_list.mat
img_paths = []
labels = []
for index, row in train.iterrows():
    img_path = "./data/train/" + "{:06d}".format(row['id']) + ".jpg"
    img_paths.append(img_path)
    labels.append(row['label'])

img_paths = np.array(img_paths)
labels = np.array(labels)
scipy.io.savemat('./data/train_list.mat', {
    'img_paths': img_paths,
    'labels': labels}
)
