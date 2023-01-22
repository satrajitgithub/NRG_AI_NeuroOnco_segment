from functools import partial

from keras import backend as K
from keras.layers import *
from keras.engine import Model
from keras.optimizers import Adam
from keras import regularizers


K.set_image_data_format("channels_first")


def create_localization_module(input_layer, n_filters, regularizer = None):
    convolution1 = create_convolution_block(input_layer, n_filters, regularizer = regularizer)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1), regularizer = regularizer)
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2), regularizer = None):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters, regularizer = regularizer)
    return convolution

def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first", regularizer = None):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters, regularizer = regularizer)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters, regularizer = regularizer)

    return convolution2


def create_convolution_block(input_layer, n_filters, name=None, batch_normalization=False, kernel=(3, 3, 3), activation=LeakyReLU,
                             padding='same', strides=(1, 1, 1), regularizer=None):
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides, name=name, kernel_regularizer=regularizer)(input_layer)
    
    try:
        from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
    except ImportError:
        raise ImportError("Install keras_contrib in order to use instance normalization."
                          "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
    layer = InstanceNormalization(axis=1)(layer)

    return activation()(layer)


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# https://github.com/keras-team/keras/issues/9395#issuecomment-379276452
def dice_coef_multilabel(y_true, y_pred, numLabels):
    
    dice=0

    for index in range(numLabels):
        dice += dice_coefficient(y_true[:,index,:,:,:], y_pred[:,index,:,:,:])

    return dice


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred, axis=axis) + smooth/2)/(K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return 1 - weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


def segmentation_model(input_shape,
                        n_labels=4,
                        initial_learning_rate=0.0005,
                        n_segmentation_levels=3,                        
                        include_label_wise_dice_coefficients=True,
                        regularizer=None,
                        depth = 5,
                        n_base_filters=16):

    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Encoding path ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()

    for level_number in range(depth):
        n_level_filters = (2 ** level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters, regularizer=regularizer)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2), regularizer=regularizer)

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=0.3, regularizer=regularizer)

        summation_layer = Add()([in_conv, context_output_layer])  # number of summation_layers = depth
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Decoding path ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number], regularizer=regularizer)

        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)


        localization_output = create_localization_module(concatenation_layer, level_filters[level_number], regularizer=regularizer)
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1), kernel_regularizer=regularizer)(current_layer))
            # todo: should there be an activation after the conv3D block here? compare with  https://github.com/MIC-DKFZ/BraTS2017/blob/master/network_architecture.py

    # Accumulate deep supervision
    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation('sigmoid', name="segm_op")(output_layer)
    
    # Define segmentation model
    model_seg = Model(inputs=inputs, outputs=activation_block)


    metrics_seg = partial(dice_coef_multilabel, numLabels=n_labels)
    metrics_seg.__setattr__('__name__', 'dice_coef_multilabel')

    if not isinstance(metrics_seg, list):
        metrics_seg = [metrics_seg]

    label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
    metrics_seg = metrics_seg + label_wise_dice_metrics

    model_seg.compile(optimizer=Adam(lr=initial_learning_rate), loss=weighted_dice_coefficient_loss, metrics=metrics_seg)

    return model_seg