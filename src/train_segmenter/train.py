import random
import time
random.seed(9001)

import importlib

from config_utils import *
from data import write_data_to_file, open_data_file
from generator import get_training_and_validation_generators_segmentation
from model import segmentation_model

import pickle
import os 

from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, Callback, TensorBoard
from keras.engine import Model

import tensorflow as tf
import numpy as np

from metrics import *

K.common.set_image_dim_ordering('th')


class LastEpochModelSaverCallback(Callback):
    def __init__(self, modelpath, number_of_epochs):
        super(LastEpochModelSaverCallback, self).__init__()
        self.modelPath = modelpath
        self.numberOfEpochs = number_of_epochs

    def on_epoch_end(self, epoch, logs=None):
        # print(list(logs.keys()))
        if epoch+1 == self.numberOfEpochs:
            print("Last model saved at: {}modelSegmenter_ep{:03d}_dice_{:.4f}_vloss{:.4f}.h5".format(self.modelPath, self.numberOfEpochs, logs["val_dice_coef_multilabel"], logs["val_loss"]))
            self.model.save("{}modelSegmenter_ep{:03d}_dice_{:.4f}_vloss{:.4f}.h5".format(self.modelPath, self.numberOfEpochs, logs["val_dice_coef_multilabel"], logs["val_loss"]))
            
        else:
            pass

class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn


    def on_epoch_end(self, epoch, logs={}):
        msg = "{Epoch: %i} %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.print_fcn(msg)

def train_model_segmentation(config, logger, model, training_generator, validation_generator,
                        training_steps_per_epoch, validation_steps_per_epoch):

    model_file=config["model_file"]
    log_file=config["log_file"]
    model_save_path = config["basepath"]

    learning_rate_drop=config["learning_rate_drop"]
    learning_rate_patience=config["patience"]
    early_stopping_patience=config["early_stop"]
    n_epochs=config["n_epochs"]


    history = model.fit_generator(generator=training_generator,
                                steps_per_epoch=training_steps_per_epoch,
                                epochs=n_epochs,
                                initial_epoch=0,
                                validation_data=validation_generator,
                                validation_steps=validation_steps_per_epoch,
                                callbacks=[ModelCheckpoint(model_file, monitor="val_dice_coef_multilabel", mode="max", save_best_only=False),
                                           LastEpochModelSaverCallback(model_save_path, n_epochs),
                                           CSVLogger(log_file, append=True),
                                           EarlyStopping(patience=early_stopping_patience, verbose=1, restore_best_weights=False),
                                           ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience, verbose=1),
                                           LoggingCallback(logger.info)],
                                use_multiprocessing=True)


    with open(os.path.join(config["basepath"], 'model_history'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


if __name__ == "__main__":

    fold = 1 # choose between 1-5 to set fold out as validation fold and training on rest of the folds
    exp = "dev" # name of experiment set by user
    
    # config_file_name should be f"config_{exp}.py" e.g., in this case config_dev.py
    config_file_name = f"config_{str(exp)}"
    
    # The file gets executed upon import, as expected.
    config_file = importlib.import_module(config_file_name)

    # Then you can use the module like normal
    set_fold = config_file.set_fold
    config = config_file.config

    overwrite = config["overwrite"]
    set_fold(fold, exp)

    create_basepath_and_code_snapshot(fold, config, config_file_name)

    # Create and configure logger
    logger = create_logger(config)

    logger.info("***************************************************************************************************************")
    logger.info("*" * 50 + " [ EXPERIMENT #{} ]".format(exp) + " [ FOLD #{} ]".format(fold) + "*" * 50)
    logger.info("***************************************************************************************************************")

    logger.info("~" * 60 + " [CONFIG] " + "~" * 60)

    print_data_info(logger, config)

    # convert input images into an hdf5 file
    data_file_opened_tr_list = list()
    data_file_opened_val_list = list()

    logger.info("\n" + "=" * 30 + " [TRAINING FILES] " + "=" * 30)
    training_files, subject_ids_tr = create_training_validation_testing_files(logger, config, path_to_sessions=config["training_sessions"])

    logger.info("\n" + "=" * 30 + " [VALIDATION FILES] " + "=" * 30)
    validation_files, subject_ids_val = create_training_validation_testing_files(logger, config, path_to_sessions=config["validation_sessions"])

    logger.info("\n" + "=" * 30 + " [TEST FILES] " + "=" * 30)
    test_files, subject_ids_test = create_training_validation_testing_files(logger, config, path_to_sessions=config["testing_sessions"])

    
    start_time = time.time()

    if overwrite or not os.path.exists(config["data_file_tr"]):
        logger.info("\n" + "=" * 30 + ": [TRAINING] write_data_to_file" + "=" * 30 + "\n")
        write_data_to_file(logger, config, training_files, config["data_file_tr"], subject_ids_tr)
        
    data_file_opened_tr = open_data_file(config["data_file_tr"])

    if overwrite or not os.path.exists(config["data_file_val"]):
        logger.info("\n" + "=" * 30 + ": [VALIDATION] write_data_to_file" + "=" * 30 + "\n")
        write_data_to_file(logger, config, validation_files, config["data_file_val"], subject_ids_val)
    
    data_file_opened_val = open_data_file(config["data_file_val"])

    if overwrite or not os.path.exists(config["data_file_test"]):
        if len(config['testing_sessions']):
            logger.info("\n" + "=" * 30 + ": [TESTING] write_data_to_file" + "=" * 30 + "\n")
            write_data_to_file(logger, config, test_files, config["data_file_test"], subject_ids_test)
            data_file_opened_test = open_data_file(config["data_file_test"])

    logger.info("[TIME-WRITE_DATA_TO_FILE] " + str(time.time() - start_time) + " seconds")

    start_time = time.time()
    
    logger.info("Number of training sessions: " + str(data_file_opened_tr.root.data.shape))
    logger.info("Number of validation sessions: " + str(data_file_opened_val.root.data.shape))
    logger.info("Number of testing sessions: " + str(data_file_opened_test.root.data.shape))
    
    data_file_opened_tr_list.append(data_file_opened_tr)
    data_file_opened_val_list.append(data_file_opened_val)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Calling Generators ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators_segmentation(
        logger,
        config,
        data_file_opened_tr_list,
        data_file_opened_val_list,

        training_keys_file=[config["training_file"]],
        validation_keys_file=[config["validation_file"]],
        n_labels=config["n_labels"],
        labels=config["labels"],

        batch_size=config["batch_size"],
        overwrite=overwrite,

        validation_batch_size=config["validation_batch_size"],
        augment=config["augment"],
        augment_flip=config["flip"])

    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Compile model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model_seg = segmentation_model(input_shape=config["input_shape"], n_labels=config['n_labels'], initial_learning_rate=config["initial_learning_rate"])
    model_seg.summary(print_fn=lambda x: logger.info(x), line_length=150)
    
    
    train_model_segmentation(config, logger, model_seg,
                             training_generator=train_generator,
                             validation_generator=validation_generator,
                             training_steps_per_epoch=n_train_steps,
                             validation_steps_per_epoch=n_validation_steps)

    for data_file in data_file_opened_tr_list:
        data_file.close()

    for data_file_val in data_file_opened_val_list:
        data_file_val.close()
