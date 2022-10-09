import glob
import logging
import os
import argparse
import nibabel as nib
from keras.models import load_model
from keras_contrib.layers import InstanceNormalization
import numpy as np
import tensorflow as tf
from functools import partial
from pathlib import Path
from medpy.metric.binary import dc, hd, asd, assd, ravd
import sys
sys.path.append(os.environ['SCRIPT_ROOT']) # Adds higher directory to python modules path.

from configs.default_config import config
from src.create_radiomics import calculate_radiomic_features_per_session
from src.data import reslice_image_set, resize
import src.metrics
from src.visualize import plot_prediction_3d, get_zoomed_data
from src.prediction import run_validation_case_segmentation
from src.refinement_heuristic import refine_seg

def load_old_model(model_file, n_labels):
    metrics_seg = src.metrics.partial(src.metrics.dice_coef_multilabel, numLabels=n_labels)
    metrics_seg.__setattr__('__name__', 'dice_coef_multilabel')

    custom_objects = {'weighted_dice_coefficient': src.metrics.weighted_dice_coefficient,
                      'weighted_dice_coefficient_loss': src.metrics.weighted_dice_coefficient_loss,
                      'dice_coef_multilabel': metrics_seg,
                      "InstanceNormalization": InstanceNormalization,
                      'label_0_dice_coef': src.metrics.get_label_dice_coefficient_function(0),
                      'label_1_dice_coef': src.metrics.get_label_dice_coefficient_function(1),
                      'label_2_dice_coef': src.metrics.get_label_dice_coefficient_function(2),
                      'label_3_dice_coef': src.metrics.get_label_dice_coefficient_function(3),
                      'label_4_dice_coef': src.metrics.get_label_dice_coefficient_function(4)
                      }

    return load_model(model_file, custom_objects=custom_objects)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Code for running Segmentation on GBM')
    parser.add_argument('--evaluate', default=False, action='store_true', help="plot segmentation results per case + evaluate metrics (DSC/Hausdorff)")
    parser.add_argument('--radiomics', default=False, action='store_true', help="Calculate radiomic features using pyradiomics")

    args = vars(parser.parse_args())

    print(args)

    print("Checking if gpu is available",tf.test.is_gpu_available())

    # Setting the basepath of the folder inside which everything will be stored
    subject_dir = os.path.join('/input')
    output_dir = os.path.join('/output')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config["basepath"] = subject_dir
    model_files = glob.glob(os.path.join(os.environ['SCRIPT_ROOT'], "trained_models", "*.h5"))

    print("*************************************************************************************************")
    print("*" * 40 + " [ PREDICTION] " + "*" * 40)
    print("*************************************************************************************************")

    print("[INFO] Available models:")
    for i in model_files: print(i)

    required_scans = [os.path.abspath(os.path.join(subject_dir, modality + ".nii.gz")) for modality in config["training_modalities"]]
    present_scans = glob.glob(os.path.abspath(os.path.join(subject_dir, "*stripped.nii.gz")))
    present_scans = [os.path.abspath(i) for i in required_scans if i in present_scans]
    modality_list = [i.split(os.sep)[-1].split('_')[0] for i in present_scans]

    original_dim = nib.load(present_scans[0]).get_fdata().shape
    print(f"Expected to find all or a subset of these scans: \n {required_scans}")
    print(f"Found modalities = {modality_list} from following scans =")
    for scan_name in present_scans: print(f"* {scan_name}")

    images = reslice_image_set(present_scans, config["image_shape"])
    subject_data = [image.get_fdata() for image in images]

    test_data = np.asarray(subject_data)[np.newaxis]
    affine = np.asarray(images[0].affine)

    if modality_list == ['T1c', 'T2', 'Flair']:
        model = load_old_model([i for i in model_files if "_T1c+T2+FLAIR_" in i][0], 4)
    elif modality_list == ['T1c', 'Flair']:
        model = load_old_model([i for i in model_files if "_T1c+FLAIR_" in i][0], 4)
    elif modality_list == ['T1c', 'T2']:
        model = load_old_model([i for i in model_files if "_T1c+T2_" in i][0], 4)
    elif modality_list == ['T2', 'Flair']:
        model = load_old_model([i for i in model_files if "_T2+FLAIR_" in i][0], 2)
    elif modality_list == ['T2']:
        model = load_old_model([i for i in model_files if "_T2_" in i][0], 2)
    elif modality_list == ['T1c']:
        model = load_old_model([i for i in model_files if "_T1c_" in i][0], 2)
    elif modality_list == ['Flair']:
        model = load_old_model([i for i in model_files if "_FLAIR_" in i][0], 2)
    else:
        raise ValueError(f"Modality list = {modality_list}, could not find a suitable model..")

    prediction_image = run_validation_case_segmentation(affine, test_data, model, config['labels'])
    prediction_image = refine_seg(prediction_image, modality_list)

    # Only if pred nonzero, go into following loop
    if np.count_nonzero(prediction_image.get_fdata()):
        pred_upsample = resize(prediction_image, original_dim, interpolation="nearest")
        nib.save(pred_upsample, os.path.join(output_dir, 'prediction.nii.gz'))
        
        # Calculate radiomic features
        if args['radiomics']:
            calculate_radiomic_features_per_session(subject_dir, output_dir)

        # Plot and/or evaluate segmentation for QC
        if args["evaluate"]:
            # data: (1,#modalities,H,W,C), gt: (1,1,H,W,C), pred: (1,1,H,W,C)
            data = np.concatenate([nib.load(i).get_fdata()[None, ...] for i in present_scans])[None, ...]
            pred = nib.load(os.path.join(output_dir, 'prediction.nii.gz')).get_fdata()[None, None, ...]
            gt_path = os.path.join(subject_dir, config["truth"][0] + ".nii.gz")    

            # If GT exists, calculate metrics
            if os.path.exists(gt_path):
                print(f"Found groundtruth = {gt_path}. Evaluating segmentation, calculating metrics..")
                gt = nib.load(gt_path).get_fdata()[None, None, ...]
                metrics = ["dc", "hd", "asd", "assd", "ravd"]
                calc_metrics = [round(func(np.squeeze(pred),np.squeeze(gt)),3) for func in [dc, hd, asd, assd, ravd]]
                ax_title = {i:j for i,j in zip(metrics, calc_metrics)}
                print(f"Segmentation performance: {ax_title}")
            else:    
                gt = np.zeros_like(pred)               
                ax_title = ""


            plot_prediction_3d(data, modality_list, gt, pred, ax_title=ax_title, outfile=os.path.join(output_dir, f'seg.png'))

            # lower zoom_out_factor gives a more magnified view of the tumor
            data, gt, pred = get_zoomed_data(data, gt, pred, zoom_out_factor = 10)
            plot_prediction_3d(data, modality_list, gt, pred, ax_title=ax_title, outfile=os.path.join(output_dir, f'seg_zoomed.png'))

    else:
        print("Segmentation all zeros, please recheck the input scans.")

