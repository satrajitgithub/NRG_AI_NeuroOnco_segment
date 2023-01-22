import numpy as np
import nibabel as nib
from nilearn.image import new_img_like, resample_to_img
import random
import itertools

def flip_image(image, axis):
    try:
        new_data = np.copy(image.get_data())
        for axis_index in axis:
            new_data = np.flip(new_data, axis=axis_index)
    except TypeError:
        new_data = np.flip(image.get_data(), axis=axis)
    return new_img_like(image, data=new_data)


def random_flip_dimensions(flip_axes):
    
    axis = list()
    for flip_axis in flip_axes:
        if np.random.choice([True, False]):
            axis.append(flip_axis)
    return axis


def distort_image(image, flip_axis=None):
    
    if flip_axis: image = flip_image(image, flip_axis)

    return image


def augment_data(data, truth, affine, flip=[0,1,2]):
    
    n_dim = len(truth.shape)
            
    if flip:
        flip_axis = random_flip_dimensions(flip)
    else:
        flip_axis = None

    data_list = list()

    for data_index in range(data.shape[0]):
        image = get_image(data[data_index], affine)
        data_list.append(resample_to_img(distort_image(image, flip_axis=flip_axis), image, interpolation="continuous").get_data())

    data = np.asarray(data_list)
    truth_image = get_image(truth, affine)
    truth_data = resample_to_img(distort_image(truth_image, flip_axis=flip_axis), truth_image, interpolation="nearest").get_data()
    return data, truth_data


def get_image(data, affine, nib_class=nib.Nifti1Image):
    return nib_class(dataobj=data, affine=affine)
