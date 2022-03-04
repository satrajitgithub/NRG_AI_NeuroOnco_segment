'''
Author: Satrajit Chakrabarty, satrajit.chakrabarty@wustl.edu
Copyright (c) 2021, Computational Imaging Lab, School of Medicine, Washington University in Saint Louis

Redistribution and use in source and binary forms, for any purpose, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''


import os
import pickle

import nibabel as nib
import numpy as np
from nilearn.image import reorder_img, new_img_like

from src.utils.sitk_utils import resample_to_spacing, calculate_origin_offset


def reslice_image_set(in_files, image_shape):
    images = read_image_files(in_files, image_shape=image_shape)
    return images


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


def read_image_files(image_files, image_shape=None):
    image_list = list()

    for index, image_file in enumerate(image_files):
        print("Reading: {}".format(os.path.abspath(image_file)))
        image_list.append(read_image(image_file, image_shape=image_shape, interpolation="linear"))

    return image_list


def read_image(in_file, image_shape=None, interpolation='linear'):
    if os.path.exists(os.path.abspath(in_file)):
        image = nib.load(os.path.abspath(in_file))
        image = fix_shape(image)  # Removes extra fourth axis if present
        image = fix_canonical(image)  # Converts all image files to RAS orientation
        image = z_score(image)  # Normalizes image using mean and std values of brain area after excluding top/bottom 5th percentile values
        return resize(image, new_shape=image_shape, interpolation=interpolation)
    else:
        raise ValueError(f"Cannot find: {os.path.abspath(in_file)}")


def check_unique_elements(np_array):
    # Extract the end-points of the 3D bbox from the tumor mask
    unique, counts = np.unique(np_array, return_counts=True)
    return dict(zip(unique, counts))


def fix_shape(image):
    if image.shape[-1] == 1:
        return image.__class__(dataobj=np.squeeze(image.get_data()), affine=image.affine)
    return image


def resize(image, new_shape, interpolation="linear"):
    image = reorder_img(image, resample=interpolation)
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)
    new_data = resample_to_spacing(image.get_data(), image.header.get_zooms(), new_spacing,
                                   interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    return new_img_like(image, new_data, affine=new_affine)


def fix_canonical(image):
    file_ort = nib.aff2axcodes(image.affine)

    if file_ort != ('R', 'A', 'S'):
        return nib.as_closest_canonical(image)
    else:
        return image
def z_score(image):

    input_numpy = image.get_fdata()
    input_numpy_nonzero = (input_numpy[input_numpy > 0])
    vol_mean = np.mean(input_numpy_nonzero[(input_numpy_nonzero < np.percentile(input_numpy_nonzero, 95)) & (input_numpy_nonzero > np.percentile(input_numpy_nonzero,
                                                                                                                                                 5))])  # Calculating the mean of the input image of the entire image (changed from just ROI to deal with pixels outside the ROI)
    vol_std = np.std(input_numpy_nonzero[(input_numpy_nonzero < np.percentile(input_numpy_nonzero, 95)) & (
                input_numpy_nonzero > np.percentile(input_numpy_nonzero, 5))])  # Calculating the std of the input image of the entire image (changed from just ROI to deal with pixels outside the ROI)
    # print("The mean and std of the ROI are "+str(vol_mean)+" and "+str(vol_std)) # Printing the values
    normalize_numpy = (input_numpy - vol_mean) / vol_std  # Normalizing the input image using the mean and std

    return new_img_like(image, normalize_numpy, affine=image.affine)