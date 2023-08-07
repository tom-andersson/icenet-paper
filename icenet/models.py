import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))  # if using jupyter kernel
import config
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, UpSampling2D, \
    concatenate, MaxPooling2D, Input
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
    # 定义一个内部函数，表示残差块
    def layer(input_tensor):
        # 创建一个快捷连接，使用1x1的卷积核和给定的填充和初始化方法
        # 这个快捷连接将用于跳过一些层，创建一个直接连接
        shortcut = Conv2D(filters, (1, 1), padding=padding, kernel_initializer=kernel_initializer)(input_tensor)

        # 对输入张量执行自定义的可分离卷积操作
        # 这将使用深度卷积跟随点卷积的组合
        x = CustomSeparableConv2D(filters, kernel_size, activation=None, padding=padding, kernel_initializer=kernel_initializer)(input_tensor)

        # 使用泄漏的ReLU激活函数，其中alpha参数定义负激活的斜率
        x = LeakyReLU(alpha=0.1)(x)

        # 再次应用自定义的可分离卷积层
        x = CustomSeparableConv2D(filters, kernel_size, activation=None, padding=padding, kernel_initializer=kernel_initializer)(x)

        # 再次应用泄漏的ReLU激活函数
        x = LeakyReLU(alpha=0.1)(x)

        # 将快捷连接与卷积路径合并（相加）
        # 这样可以将输入直接与更深层次的特征组合，从而增加了网络的容量并有助于缓解梯度消失的问题
        x = Add()([shortcut, x])

        return x
    return layer


class CustomSeparableConv2D(tf.keras.layers.Layer):
    """
    此CustomSeparableConv2D类定义了一个自定义的可分离卷积层。
    它首先使用深度卷积在每个通道上单独应用卷积，然后使用1x1卷积（点卷积）来组合这些通道。
    这种方法与常规卷积相比，通常可以减少参数数量并提高效率。
    """
    def __init__(self, filters, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal',
                 **kwargs):
        # 调用父类构造器，允许在创建层时传递自定义参数
        super(CustomSeparableConv2D, self).__init__(**kwargs)

        # 创建一个深度卷积层，它会在每个输入通道上单独应用卷积
        # 这可以有效地减少参数数量，并允许模型学习跨通道的特征
        # depth_multiplier 参数确定每个输入通道的输出通道数
        # 在此情况下，depth_multiplier 设置为1，意味着输入和输出通道数量相同
        self.depthwise_conv2d = DepthwiseConv2D(kernel_size, activation=activation, padding=padding, depth_multiplier=1,
                                                depthwise_initializer=kernel_initializer)

        # 创建一个点卷积层，它具有1x1的卷积核
        # 该层用于组合深度卷积层产生的通道特征
        # 通过这种方式，模型可以学习如何最有效地组合跨通道的信息
        self.pointwise_conv2d = Conv2D(filters, 1, activation=activation, padding=padding,
                                       kernel_initializer=kernel_initializer)

    def call(self, inputs):
        # 通过深度卷积层传递输入
        # 这将在每个通道上单独应用卷积，产生与输入通道数相同的输出通道数
        x = self.depthwise_conv2d(inputs)

        # 通过点卷积层传递深度卷积的输出
        # 该层将组合不同通道的特征，并产生最终的输出通道数，由“filters”参数确定
        x = self.pointwise_conv2d(x)

        return x

def channel_attention(input_feature, ratio=8):
    """
    该函数实现了通道注意力机制，这是一种能够使模型专注于输入特征的重要通道的技术。
    它通过对输入特征执行全局平均池化和全局最大池化，并学习每个通道的权重来实现。
    然后，这些权重与原始输入相乘，从而强调重要通道的贡献并减弱不重要通道的贡献。
    """
    # 确定通道轴的位置，取决于输入数据的格式
    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
    # 获取输入特征的通道数
    channel = input_feature.shape[channel_axis]

    # 定义共享的全连接层，用于学习通道注意力
    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    # 计算全局平均池化，结果将具有与输入通道数相同的特征
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    # 使用共享层进行处理，用于学习每个通道的重要性
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    # 计算全局最大池化，结果将具有与输入通道数相同的特征
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    # 使用与平均池化相同的共享层进行处理
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    # 将平均池化和最大池化的结果结合在一起
    cbam_feature = Add()([avg_pool, max_pool])
    # 通过Sigmoid激活函数将其转换为0和1之间的权重
    cbam_feature = Activation('sigmoid')(cbam_feature)

    # 将原始输入特征与学习到的通道注意力权重相乘
    # 这将突出输入中重要通道的特征，并减小不重要通道的特征
    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    """
    这个函数实现了一种称为空间注意力机制的技术，它使模型能够关注输入特征中的重要空间位置。
    这是通过计算输入特征的平均值和最大值，然后将它们连接并通过一个卷积层来实现的。
    最后，得到的空间注意力图与原始输入相乘，从而强调输入中重要位置的特征，并减弱不重要位置的特征。
    """
    # 设置卷积核的大小
    kernel_size = 7

    # 根据数据格式确定通道轴的位置并相应地调整输入特征
    if tf.keras.backend.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    # 计算沿通道轴的平均值，生成一个空间注意力图
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    # 计算沿通道轴的最大值，生成另一个空间注意力图
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    # 将两个注意力图合并在一起
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    # 对合并后的注意力图应用一个卷积层，得到最终的空间注意力图
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    # 如果数据格式为 "channels_first"，则需要将空间注意力图转置回原来的格式
    if tf.keras.backend.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    # 将原始输入特征与空间注意力图相乘，以实现空间注意力机制
    return multiply([input_feature, cbam_feature])


'''
Defines the Python-based sea ice forecasting models, such as the IceNet architecture
and the linear trend extrapolation model.
'''

### Custom layers:
# --------------------------------------------------------------------

# 使用装饰器注册该类，使其可以在保存和加载Keras模型时序列化
@tf.keras.utils.register_keras_serializable()
class TemperatureScale(tf.keras.layers.Layer):
    '''
    Implements the temperature scaling layer for probability calibration,
    as introduced in Guo 2017 (http://proceedings.mlr.press/v70/guo17a.html).
    '''

    # 构造函数，用于初始化层
    def __init__(self, **kwargs):
        # 调用父类的构造函数
        super(TemperatureScale, self).__init__()
        # 定义一个张量变量，表示温度参数T，初始化为1.0
        # 该参数不可训练，用于缩放输入的logits
        self.temp = tf.Variable(initial_value=1.0, trainable=False,
                                dtype=tf.float32, name='temp')

    # 在前向传播过程中调用该方法，实现层的功能
    def call(self, inputs):
        ''' Divide the input logits by the T value. '''
        # 将输入的logits除以温度参数T，实现概率校准
        return tf.divide(inputs, self.temp)

    # 用于保存和加载具有此自定义层的模型
    def get_config(self):
        ''' For saving and loading networks with this custom layer. '''
        # 将温度参数T保存为配置
        return {'temp': self.temp.numpy()}


### Network architectures:
# --------------------------------------------------------------------
# 定义U-Net模型，具有批量归一化
def unet_batchnorm(input_shape, loss, weighted_metrics, learning_rate=1e-4, filter_size=3,
                   n_filters_factor=1, n_forecast_months=1, use_temp_scaling=False,
                   n_output_classes=3,
                   **kwargs):
    """
    这个函数定义了一个U-Net结构的卷积神经网络，具有注意力机制和批量归一化。
    U-Net是一个全卷积网络，具有对称的编码器和解码器结构，适用于图像分割任务。
    在此实现中，还添加了通道和空间注意力机制，以及可选的温度缩放层。
    """
    # 输入层
    inputs = Input(shape=input_shape)

    # 第一个卷积块，包括两个自定义可分离的卷积层，注意力机制和批量归一化
    conv1 = CustomSeparableConv2D(int(64 * n_filters_factor), filter_size, activation='relu', padding='same',
                                  kernel_initializer='he_normal')(inputs)
    conv1 = CustomSeparableConv2D(int(64 * n_filters_factor), filter_size, activation='relu', padding='same',
                                  kernel_initializer='he_normal')(conv1)
    conv1 = channel_attention(conv1)  # 通道注意力
    conv1 = spatial_attention(conv1)  # 空间注意力
    bn1 = BatchNormalization(axis=-1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    # 第二个卷积块，包括残差卷积、批量归一化和Dropout层
    conv2 = ResidualConv2D(np.int64(128 * n_filters_factor), filter_size)(pool1)
    conv2 = ResidualConv2D(np.int64(128 * n_filters_factor), filter_size)(conv2)
    bn2 = BatchNormalization(axis=-1)(conv2)
    drop2 = Dropout(0.3)(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)

    # 第三个卷积块，同样包括残差卷积、批量归一化和Dropout层
    conv3 = ResidualConv2D(np.int64(256 * n_filters_factor), filter_size)(pool2)
    conv3 = ResidualConv2D(np.int64(256 * n_filters_factor), filter_size)(conv3)
    bn3 = BatchNormalization(axis=-1)(conv3)
    drop3 = Dropout(0.3)(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    # 第四个卷积块，使用自定义可分离的卷积层和批量归一化
    conv4 = CustomSeparableConv2D(np.int64(256*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = CustomSeparableConv2D(np.int64(256*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    bn4 = BatchNormalization(axis=-1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    # 第五个卷积块，使用自定义可分离的卷积层和批量归一化
    conv5 = CustomSeparableConv2D(np.int64(512*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = CustomSeparableConv2D(np.int64(512*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    bn5 = BatchNormalization(axis=-1)(conv5)

    # 下面是上采样部分，包括上采样、合并和卷积操作
    # 通过连接跳跃连接来增强信息流，并逐渐恢复图像分辨率

    # 第一个上采样块
    up6 = CustomSeparableConv2D(np.int64(256*n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2), interpolation='nearest')(bn5))
    merge6 = concatenate([bn4, up6], axis=3)
    conv6 = CustomSeparableConv2D(np.int64(256*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = CustomSeparableConv2D(np.int64(256*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    bn6 = BatchNormalization(axis=-1)(conv6)

    # 第二个上采样块
    up7 = CustomSeparableConv2D(np.int64(256*n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2), interpolation='nearest')(bn6))
    merge7 = concatenate([bn3,up7], axis=3)
    conv7 = CustomSeparableConv2D(np.int64(256*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = CustomSeparableConv2D(np.int64(256*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    bn7 = BatchNormalization(axis=-1)(conv7)

    # 第三个上采样块
    up8 = CustomSeparableConv2D(np.int64(128*n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2), interpolation='nearest')(bn7))
    merge8 = concatenate([bn2,up8], axis=3)
    conv8 = CustomSeparableConv2D(np.int64(128*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = CustomSeparableConv2D(np.int64(128*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    bn8 = BatchNormalization(axis=-1)(conv8)

    # 第四个上采样块
    up9 = CustomSeparableConv2D(np.int64(64*n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2), interpolation='nearest')(bn8))
    merge9 = concatenate([conv1,up9], axis=3)
    conv9 = CustomSeparableConv2D(np.int64(64*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = CustomSeparableConv2D(np.int64(64*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = CustomSeparableConv2D(np.int64(64*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    # 在最后一层添加通道和空间注意力
    conv9 = channel_attention(conv9)  # 通道注意力
    conv9 = spatial_attention(conv9)  # 空间注意力

    # 创建最后的logits层，然后根据是否使用温度缩放来缩放logits
    final_layer_logits = [(CustomSeparableConv2D(n_output_classes, 1, activation='linear')(conv9)) for i in range(n_forecast_months)]
    final_layer_logits = tf.stack(final_layer_logits, axis=-1)

    # 温度缩放的可选应用
    if use_temp_scaling:
        # 温度缩放的logits
        final_layer_logits_scaled = TemperatureScale()(final_layer_logits)
        final_layer = tf.nn.softmax(final_layer_logits_scaled, axis=-2)
    else:
        final_layer = tf.nn.softmax(final_layer_logits, axis=-2)

    # 创建模型
    model = Model(inputs, final_layer)

    # 编译模型，指定优化器、损失函数和加权指标
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
