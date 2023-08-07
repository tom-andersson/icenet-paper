import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))  # if using jupyter kernel
import config
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, UpSampling2D, concatenate, MaxPooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D
from tensorflow.keras.layers import Dropout, LeakyReLU, Add
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Reshape, GlobalMaxPooling2D, Activation, multiply
from tensorflow.keras.layers import Lambda, Concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Permute



def ResidualConv2D(filters, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal'):
    """
    这个ResidualConv2D函数定义了一个残差块，其中包括快捷连接和两个自定义可分离卷积层。
    快捷连接通过直接将输入连接到输出，提供了一种跳过一些层的机制，有助于缓解深度网络中的梯度消失问题。
    """
    def layer(input_tensor):
       
        shortcut = Conv2D(filters, (1, 1), padding=padding, kernel_initializer=kernel_initializer)(input_tensor)
        x = CustomSeparableConv2D(filters, kernel_size, activation=None, padding=padding, kernel_initializer=kernel_initializer)(input_tensor)
        x = LeakyReLU(alpha=0.1)(x)
        x = CustomSeparableConv2D(filters, kernel_size, activation=None, padding=padding, kernel_initializer=kernel_initializer)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Add()([shortcut, x])

        return x
    return layer

@tf.keras.utils.register_keras_serializable(name='CustomSeparableConv2D')
class CustomSeparableConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding='SAME', kernel_initializer='he_normal', activation='relu',
                 **kwargs):
        super(CustomSeparableConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = tf.keras.activations.get(activation)
        self.padding = padding.upper()
        if kernel_initializer == 'he_normal':
            self.kernel_initializer = tf.keras.initializers.HeNormal()
        else:
            self.kernel_initializer = tf.keras.initializers.GlorotUniform()

    def build(self, input_shape):
        input_channels = input_shape[-1]

        # Depthwise convolution
        self.depthwise_filter = self.add_weight(
            shape=(self.kernel_size, self.kernel_size) + (input_channels, 1),
            initializer=self.kernel_initializer,
            trainable=True,
            name='depthwise_filter'
        )

        # Pointwise convolution
        self.pointwise_filter = self.add_weight(
            shape=(1, 1, input_channels, self.filters),
            initializer=self.kernel_initializer,
            trainable=True,
            name='pointwise_filter'
        )

    def call(self, inputs):
        depthwise_output = tf.nn.depthwise_conv2d(
            inputs,
            self.depthwise_filter,
            strides=(1, 1, 1, 1),
            padding=self.padding
        )

        pointwise_output = tf.nn.conv2d(
            depthwise_output,
            self.pointwise_filter,
            strides=(1, 1, 1, 1),
            padding=self.padding
        )

        # pointwise_output = convolution_2d(depthwise_output,
        # self.pointwise_filter, self.padding.lower())

        output = self.activation(pointwise_output)
        return output

    def get_config(self):
        config = super(CustomSeparableConv2D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation': tf.keras.activations.serialize(self.activation)
        })
        return config


def channel_attention(input_feature, ratio=8):
    """
    该函数实现了通道注意力机制，这是一种能够使模型专注于输入特征的重要通道的技术。
    它通过对输入特征执行全局平均池化和全局最大池化，并学习每个通道的权重来实现。
    然后，这些权重与原始输入相乘，从而强调重要通道的贡献并减弱不重要通道的贡献。
    """
    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]
    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    """
    这个函数实现了一种称为空间注意力机制的技术，它使模型能够关注输入特征中的重要空间位置。
    这是通过计算输入特征的平均值和最大值，然后将它们连接并通过一个卷积层来实现的。
    最后，得到的空间注意力图与原始输入相乘，从而强调输入中重要位置的特征，并减弱不重要位置的特征。
    """
    kernel_size = 7

    if tf.keras.backend.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    if tf.keras.backend.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


'''
Defines the Python-based sea ice forecasting models, such as the IceNet architecture
and the linear trend extrapolation model.
'''

### Custom layers:
# --------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class TemperatureScale(tf.keras.layers.Layer):
    '''
    Implements the temperature scaling layer for probability calibration,
    as introduced in Guo 2017 (http://proceedings.mlr.press/v70/guo17a.html).
    '''
    def __init__(self, **kwargs):

        super(TemperatureScale, self).__init__()

        self.temp = tf.Variable(initial_value=1.0, trainable=False,
                                dtype=tf.float32, name='temp')

    def call(self, inputs):
        ''' Divide the input logits by the T value. '''

        return tf.divide(inputs, self.temp)


    def get_config(self):
        ''' For saving and loading networks with this custom layer. '''
        return {'temp': self.temp.numpy()}


### Network architectures:
# --------------------------------------------------------------------
def unet_batchnorm(input_shape, loss, weighted_metrics, learning_rate=1e-4, filter_size=3,
                   n_filters_factor=1, n_forecast_months=1, use_temp_scaling=False,
                   n_output_classes=3,
                   **kwargs):
    """
    这个函数定义了一个U-Net结构的卷积神经网络，具有注意力机制和批量归一化。
    U-Net是一个全卷积网络，具有对称的编码器和解码器结构，适用于图像分割任务。
    在此实现中，还添加了通道和空间注意力机制，以及可选的温度缩放层。
    """
    inputs = Input(shape=input_shape)

    conv1 = CustomSeparableConv2D(int(64 * n_filters_factor), filter_size, activation='relu', padding='same',
                                  kernel_initializer='he_normal')(inputs)
    conv1 = CustomSeparableConv2D(int(64 * n_filters_factor), filter_size, activation='relu', padding='same',
                                  kernel_initializer='he_normal')(conv1)
    conv1 = channel_attention(conv1)  
    conv1 = spatial_attention(conv1)  
    bn1 = BatchNormalization(axis=-1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = ResidualConv2D(np.int64(128 * n_filters_factor), filter_size)(pool1)
    conv2 = ResidualConv2D(np.int64(128 * n_filters_factor), filter_size)(conv2)
    bn2 = BatchNormalization(axis=-1)(conv2)
    drop2 = Dropout(0.3)(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = ResidualConv2D(np.int64(256 * n_filters_factor), filter_size)(pool2)
    conv3 = ResidualConv2D(np.int64(256 * n_filters_factor), filter_size)(conv3)
    bn3 = BatchNormalization(axis=-1)(conv3)
    drop3 = Dropout(0.3)(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = CustomSeparableConv2D(np.int64(256*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = CustomSeparableConv2D(np.int64(256*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    bn4 = BatchNormalization(axis=-1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = CustomSeparableConv2D(np.int64(512*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = CustomSeparableConv2D(np.int64(512*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    bn5 = BatchNormalization(axis=-1)(conv5)

    up6 = CustomSeparableConv2D(np.int64(256*n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2), interpolation='nearest')(bn5))
    merge6 = concatenate([bn4, up6], axis=3)
    conv6 = CustomSeparableConv2D(np.int64(256*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = CustomSeparableConv2D(np.int64(256*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    bn6 = BatchNormalization(axis=-1)(conv6)

    up7 = CustomSeparableConv2D(np.int64(256*n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2), interpolation='nearest')(bn6))
    merge7 = concatenate([bn3,up7], axis=3)
    conv7 = CustomSeparableConv2D(np.int64(256*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = CustomSeparableConv2D(np.int64(256*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    bn7 = BatchNormalization(axis=-1)(conv7)

    up8 = CustomSeparableConv2D(np.int64(128*n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2), interpolation='nearest')(bn7))
    merge8 = concatenate([bn2,up8], axis=3)
    conv8 = CustomSeparableConv2D(np.int64(128*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = CustomSeparableConv2D(np.int64(128*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    bn8 = BatchNormalization(axis=-1)(conv8)

    up9 = CustomSeparableConv2D(np.int64(64*n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2), interpolation='nearest')(bn8))
    merge9 = concatenate([conv1,up9], axis=3)
    conv9 = CustomSeparableConv2D(np.int64(64*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = CustomSeparableConv2D(np.int64(64*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = CustomSeparableConv2D(np.int64(64*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv9 = channel_attention(conv9)  
    conv9 = spatial_attention(conv9)  

    final_layer_logits = [(CustomSeparableConv2D(n_output_classes, 1, activation='linear')(conv9)) for i in range(n_forecast_months)]
    final_layer_logits = tf.stack(final_layer_logits, axis=-1)

    if use_temp_scaling:
        final_layer_logits_scaled = TemperatureScale()(final_layer_logits)
        final_layer = tf.nn.softmax(final_layer_logits_scaled, axis=-2)
    else:
        final_layer = tf.nn.softmax(final_layer_logits, axis=-2)

    model = Model(inputs, final_layer)

    model.compile(optimizer=Adam(lr=learning_rate), loss=loss, weighted_metrics=weighted_metrics)

    return model


### Benchmark models:
# --------------------------------------------------------------------


def linear_trend_forecast(forecast_month, n_linear_years='all', da=None, dataset='obs'):
    '''
    Returns a simple sea ice forecast based on a gridcell-wise linear extrapolation.

    Parameters:
    forecast_month (datetime.datetime): The month to forecast

    n_linear_years (int or str): Number of past years to use for linear trend
    extrapolation.

    da (xr.DataArray): xarray data array to use instead of observational
    data (used for setting up CMIP6 pre-training linear trend inputs in IceUNetDataPreProcessor).

    dataset (str): 'obs' or 'cmip6'. If 'obs', missing observational SIC months
    will be skipped

    Returns:
    output_map (np.ndarray): The output SIC map predicted
    by fitting a least squares linear trend to the past n_linear_years
    for the month being predicted.

    sie (np.float): The predicted sea ice extend (SIE).
    '''

    if da is None:
        with xr.open_dataset('data/obs/siconca_EASE.nc') as ds:
            da = next(iter(ds.data_vars.values()))

    valid_dates = [pd.Timestamp(date) for date in da.time.values]

    input_dates = [forecast_month - pd.DateOffset(years=1+lag) for lag in range(n_linear_years)]

    # Do not use missing months in the linear trend projection
    input_dates = [date for date in input_dates if date not in config.missing_dates]

    # Chop off input date from before data start
    input_dates = [date for date in input_dates if date in valid_dates]

    input_dates = sorted(input_dates)

    # The actual number of past years used
    actual_n_linear_years = len(input_dates)

    da = da.sel(time=input_dates)

    input_maps = np.array(da.data)

    x = np.arange(actual_n_linear_years)
    y = input_maps.reshape(actual_n_linear_years, -1)

    # Fit the least squares linear coefficients
    r = np.linalg.lstsq(np.c_[x, np.ones_like(x)], y, rcond=None)[0]

    # y = mx + c
    output_map = np.matmul(np.array([actual_n_linear_years, 1]), r).reshape(432, 432)

    land_mask_path = os.path.join(config.mask_data_folder, config.land_mask_filename)
    land_mask = np.load(land_mask_path)
    output_map[land_mask] = 0.

    output_map[output_map < 0] = 0.
    output_map[output_map > 1] = 1.

    sie = np.sum(output_map > 0.15) * 25**2

    return output_map, sie

def loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.reduce_mean(tf.square(y_true - y_pred))


def weighted_metrics(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))
