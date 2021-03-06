import cv2
import glob
import numpy as np
import os


IMAGE_SIZE = 400

def get_images_in_dir(dir_name):
    images = glob.glob(dir_name + '/**/*.jpg', recursive=True)
    other_extensions = ['jpeg', 'png', 'tga']
    for ext in other_extensions:
        images.extend(glob.glob(dir_name + '/**/*.' + ext, recursive=True))
    return images

def resize_image(image, size):
    # consulted https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    height, width = image.shape[0], image.shape[1]
    max_dim = max([height, width])
    mask = np.zeros((max_dim, max_dim, 3), dtype="uint8")
    x_pos, y_pos = int((max_dim - width) / 2), int((max_dim - height) / 2)
    mask[y_pos:y_pos+height, x_pos:x_pos+width] = image[0:height, 0:width, :]
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
    return mask

def main():
    # get all images in originals
    originals = get_images_in_dir('originals')
    photoshops = get_images_in_dir('photoshops')

    print(len(originals))
    print(len(photoshops))

    # make output directory
    dir1, dir2 = 'processed/originals', 'processed/photoshops'
    if not os.path.exists(dir1):
        os.makedirs(dir1)
    if not os.path.exists(dir2):
        os.makedirs(dir2)

    total = 0

    # get all images in photoshops
    for filename in photoshops:
        img = cv2.imread(filename, 1)
        if img is not None:
            img = resize_image(img, IMAGE_SIZE)
            fpath = 'processed/' + "/".join(filename.split('/')[:-1])
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            cv2.imwrite('processed/' + filename, img)

            total += 1
            if total % 1000 == 0:
                print(str(total) + " photos processed.")
        else:
            print("NoneType Error: {}".format(filename))

    print("DONE. {} photos processed in total.".format(total))
main()
