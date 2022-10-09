'''
Author: Satrajit Chakrabarty, satrajit.chakrabarty@wustl.edu
Copyright (c) 2021, Computational Imaging Lab, School of Medicine, Washington University in Saint Louis

Redistribution and use in source and binary forms, for any purpose, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''


import copy
import glob
import os
import pprint
import shutil

import SimpleITK as sitk
import nibabel as nib
import pandas as pd
from radiomics import featureextractor

from src.data import check_unique_elements, z_score

def get_submask_dict(classes_present, mask_arr):
    submask_dict = {}
    for tumor_idx, tumor_label in classes_present.items():
            submask = copy.deepcopy(mask_arr)

            # Set only target label to 1, everything else to 0
            submask[submask != tumor_idx], submask[submask == tumor_idx] = 0, 1

            submask_dict[tumor_label] = submask

    # Besides unique regions (1,2,4), we add Whole Tumor or WT region (1+2+4) which is the entire tumor
    submask_WT = copy.deepcopy(mask_arr)
    submask_WT[submask_WT > 0] = 1
    submask_dict['WT'] = submask_WT

    # Besides unique regions (1,2,4), we add Tumor Core or TC region (1+4) which is the entire tumor excluding edema
    submask_TC = copy.deepcopy(mask_arr)
    submask_TC[(submask_TC == 1) | (submask_TC == 4)] = 1
    submask_TC[(submask_TC != 1) & (submask_TC != 4)] = 0
    submask_dict['TC'] = submask_TC

    return submask_dict

def func_shape(submask_dict):
    """
    Extracts all shape features
    """
    
    df_shape_per_tumor_label = []

    for submask_label, submask in submask_dict.items():
    
        print(f"Calculating shape features for: tumor_label = {submask_label}")

        # kwargs_3d = {'binWidth': 1, 'interpolator': None, 'resampledPixelSpacing': None, 'verbose': False, 'force2D': False}

        image = sitk.GetImageFromArray(submask)
        mask = sitk.GetImageFromArray(submask)

        # extractor = featureextractor.RadiomicsFeatureExtractor(**kwargs_3d)
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()

        extractor.enableFeatureClassByName('shape')

        extractor_op = extractor.execute(image, mask)
        features = {i[len('original_'):]: j for i, j in zip(extractor_op.keys(), extractor_op.values()) if i.startswith('original')}
        df_shape = pd.DataFrame(features, index=[0])
        df_shape = df_shape.add_prefix(submask_label + "_")
        df_shape_per_tumor_label.append(df_shape)

    df_shape_all_tumor_label = pd.concat(df_shape_per_tumor_label, axis=1)

    return df_shape_all_tumor_label

def func_texture(submask_dict, session_path, modality_list):
    """
    Extracts all firstorder; shape; glcm; glrlm; glszm; gldm; ngtdm features
    """
    
    kwargs_3d = {'binWidth': 1, 'interpolator': None, 'resampledPixelSpacing': None, 'verbose': False, 'force2D': False}

    df_per_tumor_label_per_modality = []

    for submask_label, submask in submask_dict.items():
    
        print(f"Calculating firstorder/texture features for: tumor_label = {submask_label}")

        mask = sitk.GetImageFromArray(submask)

        for modality_name in modality_list:

            img_arr = nib.load(os.path.join(session_path, modality_name + '_stripped.nii.gz'))
            img_arr = z_score(img_arr)
            image = sitk.GetImageFromArray(img_arr.get_fdata())

            # extractor = featureextractor.RadiomicsFeatureExtractor(**kwargs_3d)
            extractor = featureextractor.RadiomicsFeatureExtractor()
            extractor.disableAllFeatures()
            for featureclass in ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']:
                extractor.enableFeatureClassByName(featureclass)
            extractor_op = extractor.execute(image, mask)
            features = {i[len('original_'):]: j for i, j in zip(extractor_op.keys(), extractor_op.values()) if i.startswith('original')}
            df_radiomics = pd.DataFrame(features, index=[0])
            df_radiomics = df_radiomics.add_prefix(submask_label + "_" + modality_name + "_")
            df_per_tumor_label_per_modality.append(df_radiomics)

    df_all_tumor_label_all_modality = pd.concat(df_per_tumor_label_per_modality, axis=1)

    return df_all_tumor_label_all_modality


def calculate_radiomics(session_path, path_to_mask, target_path, tumor_labels, modality_list):

    mask_arr = nib.load(path_to_mask).get_fdata()
    mask_arr_class_distribution = check_unique_elements(mask_arr)

    classes_present = {i:j for i,j in tumor_labels.items() if int(i) in mask_arr_class_distribution.keys()}

    # If there are classes in the tumor mask that are not in the pre-defined dictionary - then raise a warning
    alien_classes = set(mask_arr_class_distribution.keys()).difference(set(tumor_labels.keys())).difference(set([0.0]))
    if alien_classes:
        warnings.warn(f"Found an unknown class = {alien_classes} in the tumor segmentation mask. This will be skipped during calculation of radiomic features")

    print(f"Class distribution of segmented tumor: {mask_arr_class_distribution}")
    print(f"Radiomic features will be calculated for: {classes_present}")

    submask_dict = get_submask_dict(classes_present, mask_arr)

    for i,j in submask_dict.items():
        print(i, check_unique_elements(j))   


    df_radiomics_list = []

    # Tumor Shape feature
    df_shape = func_shape(submask_dict)
    df_radiomics_list.append(df_shape)

    # Tumor firstorder; glcm; glrlm; glszm; gldm; ngtdm features
    df_other = func_texture(submask_dict, session_path, modality_list)

    df_radiomics_list.append(df_other)

    # Concat all features above
    df_radiomics = pd.concat(df_radiomics_list, axis=1)

    # Add session name as index
    df_radiomics.to_csv(os.path.join(target_path, f'radiomics.csv'), index=False)


def calculate_radiomic_features_per_session(session_path, output_path):
    tumor_labels = {1: 'NC', 2: 'ED', 4: 'EC'}
    
    print("[RADIOMICS] Received following arguments for calculating radiomics:")

    present_scans = glob.glob(os.path.abspath(os.path.join(session_path, "*stripped.nii.gz")))
    modality_list = [i.split(os.sep)[-1].split('_')[0] for i in present_scans]
    print(f"Found modalities = {modality_list}")

    args_for_calculate_radiomics = {"session_path": session_path,
                                    "path_to_mask": glob.glob(os.path.abspath(os.path.join(output_path, 'prediction.nii.gz')))[0],
                                    "target_path": output_path,
                                    "tumor_labels": tumor_labels,
                                    "modality_list": modality_list}

    for line in pprint.pformat(args_for_calculate_radiomics).split('\n'):
        print(line)

    calculate_radiomics(**args_for_calculate_radiomics)
