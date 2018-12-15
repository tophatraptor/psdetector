import os
import numpy as np
import shutil
import skimage
from skimage import feature
from skimage.color import rgb2gray
import sklearn.feature_extraction
import cv2
import multiprocessing as mp

def get_id(file_path):
    return os.path.splitext(
        os.path.basename(file_path)
    )[0]

def get_dims(image_path):
    try:
        height, width, _ = cv2.imread(image_path).shape
    except:
        return (0, 0) # we check for this later and discard these images
    return (width, height)

def add_noise(image):
    return (skimage.util.random_noise(image, mode = 'gaussian') * 255).astype('uint8')

KERNEL_SIZE = 7

def add_gaussian_blur(image):
    return cv2.GaussianBlur(image, (KERNEL_SIZE, KERNEL_SIZE), 0)

def add_median_blur(image):
    return cv2.medianBlur(image, KERNEL_SIZE)

def get_matching_image_paths(x):
    image_list = []
    orig_shape = get_dims(id_to_original[x])
    # skip invalid original images
    if orig_shape == (0, 0):
        return None
    target = os.path.join(ps_path, x)
    root, folders, files = next(os.walk(target))
    fnames = [os.path.join(target, fname) for fname in files]
    for fname in fnames:
        if get_dims(fname) == orig_shape:
            image_list.append(fname)
        if len(image_list) == 3:
            break
    return (x, image_list)

# rescales image to 800 pixels in height
# cbased on resize_image from preprocess_images.py
def rescale_image(image):
    height, width = image.shape[0], image.shape[1]
    if height <= 800:
        return image
    rescale_fac = 800 / height
    dim = (int(width * rescale_fac), int(height * rescale_fac))
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

def get_patches(img, patch_size, stride, num_patches, seed = None):
    """
    Get patches of the given image with the given patch size and
    given stride. Output the patches to disk, each with
    filenames <output_folder>/<orig_filename>_<patch_index>.jpg.

    Inputs:
        img: 2D or 3D(for color) numpy array
        patch_size: int of dimensions of a square patch
        stride: int for horizontal stride pixel distance
        filename_prefix: string

    Output: None
    """
    if img is None:
        return
    all_patches = sklearn.feature_extraction.image.extract_patches_2d(
        image=img,
        patch_size=(patch_size, patch_size),
        max_patches = num_patches,
        random_state = seed
    )
    return all_patches

orig_path = '/media/jay/NVME/PSBattles/raw'
preprocessed_path = '/media/jay/NVME/PSBattles/preprocessed'
distortion_path = '/media/jay/NVME/PSBattles/blur_patches'
edge_path = '/media/jay/NVME/PSBattles/edge_patches'

if os.path.exists(distortion_path):
    shutil.rmtree(distortion_path)
os.makedirs(distortion_path)

if os.path.exists(edge_path):
    shutil.rmtree(edge_path)
os.makedirs(edge_path)

opath = os.path.join(distortion_path, 'original')
gpath = os.path.join(distortion_path, 'gaussian')
mpath = os.path.join(distortion_path, 'median')
npath = os.path.join(distortion_path, 'noise')

regular_patch_path = os.path.join(edge_path, 'regular')
modified_edge_path = os.path.join(edge_path, 'modified')

for directory in [opath, gpath, mpath, npath]:
    os.makedirs(directory)

for directory in [regular_patch_path, modified_edge_path]:
    os.makedirs(directory)

selected_path = orig_path

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

pool = mp.Pool(mp.cpu_count())
results = pool.map(get_matching_image_paths, folder_list)
pool.close()
pool.join()

print("Finished generating output")

image_map = {}
for x in results:
    if x is not None:
        folder_name, image_list = x
        image_map[folder_name] = image_list

# 200 X 200 patches
PATCH_SIZE = 200
# 25 pixel stride between patches
STRIDE = 25
# Number of patches to sample
NUM_PATCHES = 100
# color difference threshold
THRESHOLD = 30

CANNY_THRESHOLD = 50

data_tups = []
for imgid in image_map:
    data_tups.append((imgid, image_map[imgid]))

def preprocess_map(tup):
    imgid, file_list = tup
    if not file_list:
        return
    orig_image = cv2.imread(id_to_original[imgid])
    # rescale here so we don't do it every time we have to get a patch
    orig_image = rescale_image(orig_image)
    canny = feature.canny(rgb2gray(orig_image), sigma = 1.5)
    noisy = add_noise(orig_image)
    gblur = add_gaussian_blur(orig_image)
    mblur = add_median_blur(orig_image)
    # consulted: https://stackoverflow.com/questions/27035672/cv-extract-differences-between-two-images
    # for generating the difference masks
    paths = [opath, gpath, mpath, npath]
    images = [orig_image, gblur, mblur, noisy]
    for i, image in enumerate(images):
        current_seed = np.random.randint(0, 2**32 - 1)
        patches = get_patches(images[i], PATCH_SIZE, STRIDE, NUM_PATCHES, seed = current_seed)
        canny_patches = get_patches(canny, PATCH_SIZE, STRIDE, NUM_PATCHES, seed = current_seed)
        for j, patch in enumerate(patches):
            if np.sum(canny_patches[i]) < CANNY_THRESHOLD:
                continue
            cv2.imwrite(os.path.join(paths[i], '{}_{}.jpg'.format(imgid, j)), patch)
    for i, fname in enumerate(file_list):
        # this is a slow process
        modified_image = rescale_image(cv2.imread(fname))
        canny = feature.canny(rgb2gray(modified_image), sigma = 1.5)
        current_seed = np.random.randint(0, 2**32 - 1)
        patches = get_patches(modified_image, PATCH_SIZE, STRIDE, NUM_PATCHES, seed = current_seed)
        diff = cv2.absdiff(orig_image, modified_image)
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        imask = mask > THRESHOLD
        height, width, _ = modified_image.shape
        # if our image is mostly new things, we want to continue
        if np.sum(imask) > (height * width / 2):
            continue
        newmask = (imask * 255).astype('uint8')
        mask_patches = get_patches(imask, PATCH_SIZE, STRIDE, 5 * NUM_PATCHES, seed = current_seed)
        canny_patches = get_patches(canny, PATCH_SIZE, STRIDE, 5 * NUM_PATCHES, seed = current_seed)

        regular_patches = []
        edge_patches = []

        for i, x in enumerate(patches):
            # make sure there's actually something in the image patch
            # this avoids things like bokeh blur from hitting us too hard
            if np.sum(canny_patches[i]) > CANNY_THRESHOLD:
                score = np.sum(mask_patches[i])
                # empty mask patch = we're likely not in or near the modified area
                if score == 0:
                    regular_patches.append(x)
                # mask patch with a high score = we're just outside of the region
                # setting the upper limit because we want the neural net to learn distortion
                # at the edges, so we mainly need to be looking outside of the model
                elif 1000 < score < 5000:
                    edge_patches.append(x)
        sample_size = min(len(regular_patches), len(edge_patches), 200)
        basename = os.path.splitext(os.path.basename(fname))[0]
        for i in range(sample_size):
            cv2.imwrite(os.path.join(regular_patch_path, '{}_{}.jpg'.format(basename, i)),
            regular_patches[i])
            cv2.imwrite(os.path.join(modified_edge_path, '{}_{}.jpg'.format(basename, i)),
            edge_patches[i])

pool = mp.Pool(mp.cpu_count())
pool.map(preprocess_map, data_tups)
pool.close()
pool.join()
