import copy
import glob
import os
import pathlib
import pickle
import pprint
import random
import shutil

import pandas as pd

random.seed(9001)
import numpy as np

import nibabel as nib
from pathlib import Path

import logging

def return_none():
    return None

def set_data_paths(config, fold, exp):
    basepath = config['basepath']

    # Path to which training/validation/test hdf5 files will be written to
    config["data_file_tr"] = os.path.abspath(basepath + "fold{}_data_tr.h5".format(fold))
    config["data_file_val"] = os.path.abspath(basepath + "fold{}_data_val.h5".format(fold))

    # Path to which pickle files containing training/validation/test indices will be written to
    config["training_file"] = os.path.abspath(basepath + "training_ids.pkl")
    config["validation_file"] = os.path.abspath(basepath + "validation_ids.pkl")

    config["data_file_test"] = os.path.abspath(basepath + "fold{}_data_test.h5".format(fold))
    config["testing_file"] = os.path.abspath(basepath + "testing_ids.pkl")

def create_training_validation_testing_files(logger, config, path_to_sessions):
    training_files = list()
    subject_ids_tr = list()

    for subject_dir in path_to_sessions:
        subject_ids_tr.append(os.path.basename(subject_dir))
        subject_files = list()
        for modality in config["training_modalities"] + config["truth"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        training_files.append(tuple(subject_files))

    training_files = [list(i) for i in training_files]  # converting to list of lists from list of tuples

    logger.info("[SUBJECT_IDS] " + str(len(subject_ids_tr)) + " " + str(subject_ids_tr))

    return training_files, subject_ids_tr

def print_data_info(logger, config):
    logger.info("~" * 60 + " [CONFIG] " + "~" * 60)

    for line in pprint.pformat(config).split('\n'):
        logger.debug(line)

    logger.info("~" * 60 + " [INFO] " + "~" * 60)

    logger.info(f"[INFO] Total #sessions (train/val/test): {len(config['training_sessions'])}/{len(config['validation_sessions'])}/{len(config['testing_sessions'])}")
    

def create_logger(config):
    log_path = os.path.join(config["basepath"], "training_log.txt")
    LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(message)s"
    logging.basicConfig(filename=log_path,
                        filemode='a',
                        format=LOG_FORMAT,
                        level=logging.DEBUG)

    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path)
    logger.addHandler(fh)

    logging.getLogger('matplotlib.font_manager').disabled = True

    return logger


def create_basepath_and_code_snapshot(fold, config, config_file_name):
    # Create the basepath folder if it does not already exist
    if not os.path.exists(config["basepath"]):
        pathlib.Path(config["basepath"]).mkdir(parents=True, exist_ok=True)

    # save the config as a pickle file
    with open(os.path.join(config["basepath"], 'fold{}_config.pickle'.format(fold)), 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # snapshot current code
    if not os.path.exists(os.path.join(config["basepath"], 'code_snapshot/')):
        pathlib.Path(os.path.join(config["basepath"], 'code_snapshot/')).mkdir(parents=True, exist_ok=True)

    shutil.copy2(__file__, os.path.abspath(os.path.join(config["basepath"], 'code_snapshot')))
    shutil.copy2(os.path.join(Path(os.path.dirname(__file__)).parent, 'config_files', config_file_name + ".py"), os.path.abspath(os.path.join(config["basepath"], 'code_snapshot')))