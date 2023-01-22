import glob
import os
from collections import defaultdict
from random import shuffle

from routine.config_utils import set_data_paths, return_none

config = defaultdict(return_none)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
config['labels'] = (0, 1, 2, 4)
config['n_labels'] = len(config['labels'])
config['path_to_data'] = '/scratch/satrajit.chakrabarty/data/'

config["all_modalities"] = ["T1_stripped", "T1c_stripped", "T2_stripped", "Flair_stripped"]
config['zscore'] = True # zscores each scan

config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["truth"] = ["OTMultiClass"]
config["nb_channels"] = len(config["training_modalities"])
config["truth_channel"] = config["nb_channels"]

config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.
config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))  # (1,128,128,128)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
config["batch_size"] = 5
config["validation_batch_size"] = 5
config["n_epochs"] = 300  # cutoff the training after this many epochs
config["initial_learning_rate"] = 0.0005  # 0.001, 0.0005, 0.00025
config["patience"] = 20  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 100  # training will be stopped after this many epochs without the validation loss improving
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Augmentation parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For volume data: specify one or more of [0,1,2] eg: [0], [0,1], [1,2], [0,1,2] etc
config["flip"] = [0, 1, 2]  # augments the data by randomly flipping an axis during training

config["augment"] = config["flip"] 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ File paths ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
config["overwrite"] = False  # If True, will overwrite previous files. If False, will use previously written files.

# Split data into 5 folds from excel
def set_fold(fold, exp):
    config["fold"] = fold

    # Setting the basepath of the folder inside which everything will be stored
    config["basepath"] = "/scratch/satrajit.chakrabarty/seg_experiments/Exp" + exp + "/" + "fold" + fold + "/"

    # Split data into n folds
    sessions = glob.glob(os.path.join(config['path_to_data'], 'BRATS2021_glioma_seg', '*'))

    shuffle(sessions)

    config['training_sessions'] = sessions[: int(len(sessions) * .8)]  # 80% cases for training
    config['validation_sessions'] = sessions[int(len(sessions) * .8):int(len(sessions) * .9)]  # 10% cases for validation
    config['testing_sessions'] = sessions[int(len(sessions) * .9):]  # 10% cases for testing

    set_data_paths(config, fold, exp) 

    config["model_file"] = os.path.abspath(config["basepath"] + "modelSegmenter_ep{epoch:03d}_dice_{val_dice_coef_multilabel:.4f}_vloss{val_loss:.4f}.h5")
    config["log_file"] = os.path.abspath(config["basepath"] + "training_segmentation.log")



