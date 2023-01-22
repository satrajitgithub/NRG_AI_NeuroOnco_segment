import pickle
import os
import collections

import nibabel as nib
import numpy as np
from nilearn.image import reorder_img, new_img_like
from scipy import ndimage

import SimpleITK as sitk
import numpy as np


def calculate_origin_offset(new_spacing, old_spacing):
    return np.subtract(new_spacing, old_spacing)/2


def sitk_resample_to_spacing(image, new_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear, default_value=0.):
    zoom_factor = np.divide(image.GetSpacing(), new_spacing)
    new_size = np.asarray(np.ceil(np.round(np.multiply(zoom_factor, image.GetSize()), decimals=5)), dtype=np.int16)
    offset = calculate_origin_offset(new_spacing, image.GetSpacing())
    reference_image = sitk_new_blank_image(size=new_size, spacing=new_spacing, direction=image.GetDirection(),
                                           origin=image.GetOrigin() + offset, default_value=default_value)
    return sitk_resample_to_image(image, reference_image, interpolator=interpolator, default_value=default_value)


def sitk_resample_to_image(image, reference_image, default_value=0., interpolator=sitk.sitkLinear, transform=None,
                           output_pixel_type=None):
    if transform is None:
        transform = sitk.Transform()
        transform.SetIdentity()
    if output_pixel_type is None:
        output_pixel_type = image.GetPixelID()
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(interpolator)
    resample_filter.SetTransform(transform)
    resample_filter.SetOutputPixelType(output_pixel_type)
    resample_filter.SetDefaultPixelValue(default_value)
    resample_filter.SetReferenceImage(reference_image)
    return resample_filter.Execute(image)


def sitk_new_blank_image(size, spacing, direction, origin, default_value=0.):
    image = sitk.GetImageFromArray(np.ones(size, dtype=np.float).T * default_value)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    image.SetOrigin(origin)
    return image


def resample_to_spacing(data, spacing, target_spacing, interpolation="linear", default_value=0.):
    image = data_to_sitk_image(data, spacing=spacing)
    if interpolation is "linear":
        interpolator = sitk.sitkLinear
    elif interpolation is "nearest":
        interpolator = sitk.sitkNearestNeighbor
    else:
        raise ValueError("'interpolation' must be either 'linear' or 'nearest'. '{}' is not recognized".format(
            interpolation))
    resampled_image = sitk_resample_to_spacing(image, new_spacing=target_spacing, interpolator=interpolator,
                                               default_value=default_value)
    return sitk_image_to_data(resampled_image)


def data_to_sitk_image(data, spacing=(1., 1., 1.)):
    if len(data.shape) == 3:
        data = np.rot90(data, 1, axes=(0, 2))
    image = sitk.GetImageFromArray(data)
    image.SetSpacing(np.asarray(spacing, dtype=np.float))
    return image


def sitk_image_to_data(image):
    data = sitk.GetArrayFromImage(image)
    if len(data.shape) == 3:
        data = np.rot90(data, -1, axes=(0, 2))
    return data


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


def get_affine(in_file):
    return read_image(in_file).affine


def read_image_files(config, image_files, image_shape=None, label_indices=None):

    label_indices = [label_indices]

    image_list = list()

    for index, image_file in enumerate(image_files):
        # If it is a GT set interpolation to nearest otherwise interpolation is linear
        if (label_indices is None and (index + 1) == len(image_files)) or (label_indices is not None and index in label_indices):
            interpolation = "nearest"
        else:
            interpolation = "linear"
        print("Reading: {}".format(image_file))
        image_list.append(read_image(config, image_file, image_shape=image_shape, interpolation=interpolation))

    return image_list


def read_image(config, in_file, image_shape=None, interpolation='linear'):
    image = nib.load(os.path.abspath(in_file))

    image = fix_shape(image) # Removes extra fourth axis if present
    
    print(f"Image is in {nib.aff2axcodes(image.affine)} space")
    
    # Converts all image files to RAS orientation
    image = fix_canonical(image) 
    
    # Normalizes image using mean and std values of brain area after excluding top/bottom 5th percentile values
    if config['zscore'] and config['truth'][0] not in in_file: 
        image = z_score(image) 

    if image_shape:
        return resize(image, new_shape=image_shape, interpolation=interpolation)
    else:
        return image

def check_unique_elements(np_array):
    # Extract the end-points of the 3D bbox from the tumor mask
    unique, counts = np.unique(np_array, return_counts = True)
    return str(dict(zip(unique,counts)))

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
    
    if file_ort != ('R','A','S'):
        print("Converting to canonical (RAS orientation)")
        return nib.as_closest_canonical(image)
    else:
        # print("Image already canonical (RAS orientation)")
        return image


def z_score(image):

    input_numpy = image.get_fdata()
    input_numpy_nonzero = (input_numpy[input_numpy > 0])
    vol_mean = np.mean(input_numpy_nonzero[(input_numpy_nonzero < np.percentile(input_numpy_nonzero, 95)) & (input_numpy_nonzero > np.percentile(input_numpy_nonzero,
                                                                                                                                                 5))])  # Calculating the mean of the input image of the entire image (changed from just ROI to deal with pixels outside the ROI)
    vol_std = np.std(input_numpy_nonzero[(input_numpy_nonzero < np.percentile(input_numpy_nonzero, 95)) & (
                input_numpy_nonzero > np.percentile(input_numpy_nonzero, 5))])  # Calculating the std of the input image of the entire image (changed from just ROI to deal with pixels outside the ROI)
    normalize_numpy = (input_numpy - vol_mean) / vol_std  # Normalizing the input image using the mean and std
    print(f"The mean and std before z-scoring are {np.mean(input_numpy)} and {np.std(input_numpy)}") # Printing the values
    print(f"The mean and std of ROI are {vol_mean} and {vol_std}") # Printing the values
    print(f"The mean and std after z-scoring are {np.mean(normalize_numpy)} and {np.std(normalize_numpy)}") # Printing the values

    return new_img_like(image, normalize_numpy, affine=image.affine)