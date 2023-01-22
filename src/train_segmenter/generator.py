import math
import os
import copy
from random import shuffle
import itertools
import nibabel as nib
import numpy as np
import keras
import pandas as pd
import random
import string
from collections import Counter
from utils import pickle_dump, pickle_load
from augment import augment_data


def get_training_and_validation_generators_segmentation(logger,
                                                        config,
                                                        data_file_tr,
                                                        data_file_val,

                                                        n_labels,
                                                        training_keys_file,
                                                        validation_keys_file,
                                                        labels,

                                                        batch_size,
                                                        overwrite=False,

                                                        augment=False,
                                                        augment_flip=True,
                                                        validation_batch_size=None):

    list_of_training_list = list()
    list_of_validation_list = list()

    for i in range(len(data_file_tr)):
        training_list, validation_list = get_training_validation_lists(data_file_tr[i], 
                                                                        data_file_val[i], 
                                                                        training_file=training_keys_file[i], 
                                                                        validation_file=validation_keys_file[i], 
                                                                        overwrite=overwrite)
        list_of_training_list.append(training_list)
        list_of_validation_list.append(validation_list)


    training_generator = data_generator_classification_shuffle(data_file_tr,
                                                        list_of_training_list,
                                                        n_labels,
                                                        labels,

                                                        batch_size=batch_size,
                                                        augment=augment,
                                                        augment_flip=augment_flip)

    validation_generator = data_generator_classification_shuffle(data_file_val,
                                                          list_of_validation_list,
                                                          n_labels,
                                                          labels,
                                                          batch_size=validation_batch_size)

    
    total_tr_cases = [data_file_tr[i].root.data.shape[0] for i in range(len(data_file_tr))]
    total_val_cases = [data_file_val[i].root.data.shape[0] for i in range(len(data_file_val))]

    steps_train = get_number_of_steps(sum(total_tr_cases),batch_size)
    steps_val = get_number_of_steps(sum(total_val_cases),validation_batch_size)

    logger.info("\n" + "[#TR_STEPS] " + str(steps_train))
    logger.info("[#VAL_STEPS] " + str(steps_val))

    return training_generator, validation_generator, steps_train, steps_val

def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return 1
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1


def get_training_validation_lists(data_file_tr, data_file_val, training_file, validation_file, overwrite=False):
    """
    Splits the data into the training and validation indices list.
    :param data_file_tr: pytables hdf5 data file for training
    :param data_file_val: pytables hdf5 data file for validation
    :param training_file:
    :param validation_file:
    :param overwrite:
    :return:
    """
    
    # print("Creating training and validation lists...")
    nb_samples_tr = data_file_tr.root.data.shape[0] # Number of training data
    nb_samples_val = data_file_val.root.data.shape[0] # Number of validation data

    training_list = list(range(nb_samples_tr)) # List of integers: [0, 1, .. upto nb_samples_tr]
    validation_list = list(range(nb_samples_val)) # List of integers: [0, 1, .. upto nb_samples_val]

            
    pickle_dump(training_list, training_file)
    pickle_dump(validation_list, validation_file)
    return training_list, validation_list

def data_generator_classification_shuffle(data_file,
                                        index_list, 
                                        n_labels,                                        
                                        labels, 
                                        
                                        batch_size=1,

                                        augment=False, 
                                        augment_flip=True):

    orig_index_list = index_list[0]

    while True:
        x_list = list()
        y_list = list()
        
        index_list = copy.copy(orig_index_list) # List of integers: [0, 1, .. upto nb_samples_tr or nb_samples_val]

        
        shuffle(index_list) # if index_list was [0,1,2,3] before, after shuffling it becomes [3,1,0,2] or some other shuffled version


        while len(index_list) > 0: # while atleast 1 case is available

            
            index = index_list.pop()

            add_data(x_list,
                    y_list,
                    data_file[0],
                    int(index), 
                    augment=augment, 
                    augment_flip=augment_flip)

            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                yield convert_data_per_dataset_segmentation(x_list, y_list, n_labels=n_labels, labels=labels)  # this works as the generator

                x_list = list()
                y_list = list()
                

def save_numpy_2_nifti(image_numpy, reference_nifti_filepath, output_path):
    nifti_image = nib.load(reference_nifti_filepath)
    new_header = header=nifti_image.header.copy()
    image_affine = nifti_image.affine
    output_nifti = nib.nifti1.Nifti1Image(image_numpy, None, header=new_header)
    nib.save(output_nifti, output_path)

def add_data(x_list, y_list, data_file, index, augment=False, augment_flip=False):
    """
    Adds data from the data file to the given lists of feature and target data
    :param x_list: list of data to which data from the data_file will be appended.
    :param y_list: list of data to which the target data from the data_file will be appended.
    :param data_file: hdf5 data file.
    :param index: index of the data file from which to extract the data.
    :param augment: if True, data will be augmented according to the other augmentation parameters (augment_flip and
    augment_distortion_factor)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :return:
    """

    # ~~~~~~~~~~~~~~~~ Read data (specified by index) from hdf5 data_file ~~~~~~~~~~~~~~~~
    data, truth = get_data_from_file(data_file, index)
    affine = data_file.root.affine[index]
    
    # ~~~~~~~~~~~~~~~~ Augment data ~~~~~~~~~~~~~~~~
    if augment:
        affine = data_file.root.affine[index]
        data, truth = augment_data(data, truth, affine, flip=augment_flip)


    truth = truth[np.newaxis]

    x_list.append(data)
    y_list.append(truth)


def get_data_from_file(data_file, index):
    x, y = data_file.root.data[index], data_file.root.truth[index, 0]
    return x, y

def convert_data_per_dataset_segmentation(x_list, y_list, n_labels, labels):
    '''
    Extended from convert_data() for Segmentation purposes.
    '''

    x = np.asarray(x_list) # This is the data for input.       shape = (batch_size, number_of_channels, image_shape[0], image_shape[1], image_shape[2])
    y = np.asarray(y_list) # This is the GT for segmentation,  shape = (batch_size, 1, image_shape[0], image_shape[1], image_shape[2])

    if labels == (0, 1):
        # Prepare segmentation GT: Take OTMulticlass and make it binary by merging all tumor labels to 1
        y[y>0] = 1

    # y.shape after get_multi_class_labels = (batch_size, number_of_seg_classes, image_shape[0], image_shape[1], image_shape[2])
    y_segmentation = get_multi_class_labels(y, n_labels=n_labels, labels=labels)

    return (x, {"segm_op": y_segmentation})

def get_multi_class_labels(data, n_labels, labels=None):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    return y