import numpy as np
import nibabel as nib
import os 
import sys
import glob
import shutil
import pandas as pd


def get_whole_tumor_mask(data):
    return data > 0


def get_tumor_core_mask(data):
    return np.logical_or(data == 1, data == 4)


def get_enhancing_tumor_mask(data):
    return data == 4

def get_nonenh_tumor_mask(data):
    return data == 1

def get_edema_mask(data):
    return data == 2


def check_unique_elements(input_array):
    if isinstance(input_array, np.ndarray):
        pass
    else:
        input_array = np.array(input_array)

    # Extract the end-points of the 3D bbox from the tumor mask
    unique, counts = np.unique(input_array, return_counts = True)
    return dict(zip(unique,counts))

# This function is for saving the output similar to input
def save_numpy_like(image_numpy, nifti_image):
    new_header = nifti_image.header.copy()
    image_affine = nifti_image.affine
    output_nifti = nib.nifti1.Nifti1Image(image_numpy, image_affine, header=new_header)
    return output_nifti


def refine_seg(prediction_image, modality_list):
    pred = prediction_image.get_fdata()
    wt_vol = np.count_nonzero(get_whole_tumor_mask(pred))

    enh_vol = np.count_nonzero(get_enhancing_tumor_mask(pred))
    enh_perc = enh_vol/wt_vol

    # tc_vol = np.count_nonzero(get_tumor_core_mask(pred))
    # tc_perc = tc_vol/wt_vol

    nonenh_vol = np.count_nonzero(get_nonenh_tumor_mask(pred))
    nonenh_perc = nonenh_vol/wt_vol

    edema_vol = np.count_nonzero(get_edema_mask(pred))
    edema_perc = edema_vol/wt_vol

    # make sure t1c present, otherwise tumor prediction is only WT, and will mess up following refinement
    # + c1 = if there is <2% enhancement, then all to nonenh
    condition_for_refinement = (('T1c' in modality_list) & (enh_perc < 0.02))

    if condition_for_refinement:
        print("refining")
        # entire thing is non-enh
        pred[pred>0] = 1
    else:
        pass

    pred_refined = save_numpy_like(pred, prediction_image)

    return pred_refined

    