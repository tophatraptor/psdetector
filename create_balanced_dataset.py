import os
import torch
import random
import numpy as np
from pprint import pprint
import scipy.ndimage
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import imread
import shutil

def get_id(file_path):
    return os.path.splitext(
        os.path.basename(file_path)
    )[0]

def get_dims(image_path):
    height, width, _ = imread(image_path).shape
    return (width, height)

orig_path = '/media/jay/NVME/PSBattles/raw'
preprocessed_path = '/media/jay/NVME/PSBattles/preprocessed'

selected_path = preprocessed_path

original_path = os.path.join(selected_path, 'originals')
ps_path = os.path.join(selected_path, 'photoshops')

_, _, original_fnames = next(os.walk(original_path))

id_to_original = {}

for x in original_fnames:
    id_to_original[x[:-4]] = os.path.join(original_path, x)

original_ids = [x[:-4] for x in original_fnames]

_, folder_list, _ = next(os.walk(ps_path))

print("Original number of photoshops:", len(folder_list))

# make sure that elements of one list are in both
folder_list = [x for x in folder_list if x in original_ids]
original_ids = [x for x in original_ids if x in folder_list]

print("New number of photoshops:", len(folder_list))

original_images = [os.path.join(original_path, x) for x in original_fnames]
photoshopped_images = []

image_map = {}

true_count = 0
for x in folder_list:
    image_map[x] = []
    orig_shape = get_dims(id_to_original[x])
    target = os.path.join(ps_path, x)
    root, folders, files = next(os.walk(target))
    fnames = [os.path.join(target, fname) for fname in files]
    for fname in fnames:
        true_count += 1
        if selected_path == preprocessed_path:
            photoshopped_images.append(fname)
            image_map[x].append(fname)
        elif get_dims(fname) == orig_shape:
            # this check is slow, we only need to do it for non-preprocessed images
            photoshopped_images.append(fname)
            image_map[x].append(fname)

print("{} original images".format(len(original_images)))
print("{} matching dimension photoshopped images".format(len(photoshopped_images)))
print("{} overall photoshopped images".format(true_count))

# total number of samples for our dataset
num_samples = 30000
num_samples = min(num_samples, len(photoshopped_images), len(original_images))
print("Taking {} samples".format(num_samples))
X_paths = random.choices(original_images, k = num_samples) + random.choices(photoshopped_images, k = num_samples)
# Labels: +1 if data is a fake, else -1 if it's original
Y_labels = [1] * num_samples + [-1] * num_samples

# 60/20/20 test/train/validation split
X_train, X_test, y_train, y_test = train_test_split(
    X_paths,
    Y_labels,
    test_size = 0.4,
)

X_test, X_val, y_test, y_val = train_test_split(
    X_test, 
    y_test, 
    test_size = 0.5,
)

print("{} train examples".format(len(X_train)))
print("{} test examples".format(len(X_test)))
print("{} val examples".format(len(X_val)))


def clear_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def copy_data(folder_name, X, y):
    orig_dir = os.path.join(folder_name, 'original')
    photoshop_dir = os.path.join(folder_name, 'photoshopped')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if not os.path.exists(orig_dir):
        os.makedirs(orig_dir)
    if not os.path.exists(photoshop_dir):
        os.makedirs(photoshop_dir)
    subfolders = ['original', 'photoshopped']
    for folder in subfolders:
        out = os.path.join(folder_name, folder)
        for i, fname in enumerate(X):
            label = 'original' if y[i] == 1 else 'photoshopped'
            if label != folder:
                continue
            basename = os.path.basename(fname)
            shutil.copy(fname, os.path.join(out, basename))
            
tups = [
    (X_train, y_train, 'train'),
    (X_test, y_test, 'test'),
    (X_val, y_val, 'val'),
]

clear_folder('train')
clear_folder('test')
clear_folder('val')

for X, y, label in tups:
    copy_data(label, X, y)