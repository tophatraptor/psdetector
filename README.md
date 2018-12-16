# PSDetector

## Description
This is an assembly of some basic deep learning tools one can use to develop, train, and test neural networks.

This folder contains source code for the Computer Vision Final project developed by Euirim Choi and Jay Dhanoa.

This series of files is meant to work with the PS-Battles dataset (detailed and available for download from here: https://arxiv.org/abs/1804.04866)

## Files

### Data manipulation and training
#### `preprocess_images.py`
This script takes in various images from the psbattles dataset and will yield versions that have been rescaled and padded with black to 400 x 400. Note that this does not perform balancing of the dataset (the psbattles dataset, at the time of us running the project, had a ratio of originals to photoshops of about ~1:8).
####  `create_balanced_dataset.py`
This script will take the output of `preprocess_images.py`, and copy balanced split proportions of the dataset into train, test, and validation splits.
#### `generate_patches_datasets.py`
This script is meant to process the original PS-Battles dataset. It will, in turn, yield two datasets.

Both datasets generate 200x200 patches from the image, where the patch is centered on some identified feature of the image. For identifying these aforementioned features, we used OpenCV's implementation of the canny feature detector, and set a threshold for each patch to contain some edge of interest. This was motivated in part by the observation that quite a few entries in the PSBattles images had either relatively plain backgrounds, or the backgrounds were largely blurred out due to a bokeh/depth of field effect. At either rate, this would have introduced a relatively nontrivial number of patches where there was both no meaningful object, and there was unlikely to be some sort of image manipulation. Since our image detection efforts were largely biased towards splicing (by virtue of the nature of the PSBattles dataset), this seemed like a reasonable first step in developing a feature-rich dataset.

The first is our image distortion dataset, which is composed of four distortions: none (e.g. originals), gaussian blur, median blur, and gaussian noise. Gaussian blur and median blur were both performed on images with a gaussian kernel of size 7. The objective of this dataset was to develop a classifier that could meaningfully/accurately predict specifically what type of image distortion this image had undergone. Please refer to the writeup for this project for example figures of each of these distortions. All of the samples for these distorted image patches were taken from the original set of images.

The second dataset we worked with was the edge patch dataset. Our objective here is to create a dataset specifically of regions around spliced-in images, with the goal of training a classifier that can detect/recognize artifacts or other visual irregularities caused by splicing (e.g. noise, relative differences in image sharpness/intensity). To that end, for this portion, we sampled exclusively from photoshopped images. Our first preprocessing step here was to develop a mask for the image highlighting the region that had been modified. We then filtered out images where, in spite of being the same size as the original, were mostly different (e.g. had a color distance of >30 in most of the image).

Using both the mask and the canny edge detection as a reference, we randomly sample patches from the image. If our mask indicates this region of the image is relatively empty, then we classify it as an 'original'/unmodified patch. We select this patch if we find that  10-50% of this patch contained a spliced-in region. We selected this threshold on the basis that we wish to primarily observe the region surrounding the splice itself, in order to pick up on noise/distortion due to the splice. We additionally wish to capture enough of the spliced-in region that detection of differences in noise/sharpness could be a potential feature that the classifier could use. We discarded other examples.

### Training

#### `convnet.py`
Contains a simple convolutional network and an implementation of AlexNet, both of which are designed to train on the 400x400 image patches output by `preprocess_images.py`

#### `filterdetection.py`
Contains a modified simple convolutional neural network and a modified implementation of AlexNet, designed to train on the 200x200 image patches from the distortion dataset. Note that in addition to modifying AlexNet for input parameter size, we conducted additional experimentation with different kernel and layer sizes to deal with the different scale of features that we encountered in this training set.
