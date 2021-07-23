import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))  # if using jupyter kernel
from tensorflow.keras import backend as K
import tensorflow as tf


def construct_categorical_focal_loss(gamma=2.):
    """
    Softmax version of focal loss.
      FL = - (1 - p_c)^gamma * log(p_c)
      where p_c = probability of correct class

    Parameters:
      gamma: Focusing parameter in modulating factor (1-p)^gamma
        (Default: 2.0, as mentioned in the paper)

    Returns:
    loss: Focal loss function for training.

    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
    """

    def categorical_focal_loss(y_true, y_pred, sample_weight=None):
        """
        Parameters:
            y_true: Tensor of one-hot encoded true class values.
            y_pred: Softmax output of model corresponding to predicted
                class probabilities.

        Returns:
            focal_loss: Output tensor of pixelwise focal loss values.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss (downweights easy samples where the probability of
        #   correct class is high)
        focal_loss = K.pow(1 - y_pred, gamma) * cross_entropy

        # Loss is a tensor which is reduced implictly by TensorFlow using
        #   sample weights passed during training/evaluation
        return focal_loss

    return categorical_focal_loss


def weighted_categorical_crossentropy(y_true, y_pred, sample_weight=None):
    """
    Categorical crossentropy across all lead times with IceNet's sample weighting.

    Parameters:
        y_true: Tensor of one-hot encoded true class values.
        y_pred: Softmax output of model corresponding to predicted
            class probabilities.

    Returns:
        loss: Output tensor of pixelwise loss values.
    """

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    cross_entropy = - y_true * K.log(y_pred)

    return cross_entropy


def weighted_categorical_crossentropy_single_leadtime(y_true, y_pred, sample_weight=None):
    """
    Categorical crossentropy for a single lead time with IceNet's sample weighting.

    This assumes y_true corresponds to a singlea lead time (no lead time dimension)

    Used for computing lead-time-wise temperature scaling factors for the
    IceNet ensemble model.

    :param y_pred: A tensor resulting from a softmax
    :param y_true: A tensor of the same shape as `y_pred`
    :param sample_weight: A tensor of sample weights for weighting the CCE
    :return: Output tensor.
    """

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    # cce = tf.keras.losses.CategoricalCrossentropy()

    cross_entropy = - y_true * K.log(y_pred) * sample_weight

    return cross_entropy
