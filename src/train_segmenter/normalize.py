import os
import numpy as np
from nilearn.image import new_img_like
from utils import read_image, resize, read_image_files
import nibabel as nib

def reslice_image_set(config, in_files, image_shape, label_indices=None):

    images = read_image_files(config, in_files, image_shape=image_shape, label_indices=label_indices)
    
    return images


def get_complete_foreground(config, training_data_files):
    for i, set_of_files in enumerate(training_data_files):
        subject_foreground = get_foreground_from_set_of_files(config, set_of_files)
        if i == 0:
            foreground = subject_foreground
        else:
            foreground[subject_foreground > 0] = 1

    return new_img_like(read_image(config, training_data_files[0][-1]), foreground)


def get_foreground_from_set_of_files(config, set_of_files, background_value=0, tolerance=0.00001, return_image=False):
    for i, image_file in enumerate(set_of_files):        
        image = read_image(config, image_file)
        is_foreground = np.logical_or(image.get_data() < (background_value - tolerance),
                                      image.get_data() > (background_value + tolerance))
        if i == 0:
            foreground = np.zeros(is_foreground.shape, dtype=np.uint8)

        foreground[is_foreground] = 1
    if return_image:
        return new_img_like(image, foreground)
    else:
        return foreground


def normalize_data(data, mean, std):
    data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
    data /= std[:, np.newaxis, np.newaxis, np.newaxis]
    return data