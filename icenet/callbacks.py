import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))
import config
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ['WANDB_API_KEY'] = config.WANDB_API_KEY
os.environ['WANDB_DIR'] = config.WANDB_DIR
os.environ['WANDB_CONFIG_DIR'] = config.WANDB_CONFIG_DIR
os.environ['WANDB_CACHE_DIR'] = config.WANDB_CACHE_DIR
import wandb


class IceNetPreTrainingEvaluator(tf.keras.callbacks.Callback):
    """
    Custom tf.keras callback to update the `logs` dict used by all other callbacks
    with the validation set metrics. The callback is executed every
    `validation_frequency` batches.

    This can be used in conjuction with the BatchwiseModelCheckpoint callback to
    perform a model checkpoint based on validation data every N batches - ensure
    the `save_frequency` input to BatchwiseModelCheckpoint is also set to
    `validation_frequency`.

    Also ensure that the callbacks list past to Model.fit() contains this
    callback before any other callbacks that need the validation metrics.

    Also use Weights and Biases to log the training and validation metrics.
    """

    def __init__(self, validation_frequency, val_dataloader, sample_at_zero=False):
        # 初始化函数，用于创建类的实例时设置初始值
        self.validation_frequency = validation_frequency  # 设置验证频率，每隔多少个batch进行一次验证
        self.val_dataloader = val_dataloader  # 设置用于验证的数据加载器
        self.sample_at_zero = sample_at_zero  # 是否在第一个batch时进行一次验证

    def on_train_batch_end(self, batch, logs=None):
        # 在每个训练batch结束时调用的方法
        # 判断是否进行验证的时机，根据validation_frequency和sample_at_zero来决定是否进行验证
        if (batch == 0 and self.sample_at_zero) or (batch + 1) % self.validation_frequency == 0:
            # 调用模型的evaluate方法进行验证，并将结果存储在val_logs中
            val_logs = self.model.evaluate(self.val_dataloader, verbose=0, return_dict=True)
            # 将验证结果的键名加上前缀"val_"，并将其加入到logs中
            val_logs = {'val_' + name: val for name, val in val_logs.items()}
            logs.update(val_logs)
            # 打印每个指标的值
            [print('\n' + k + ' {:.2f}'.format(v)) for k, v in logs.items()]
            print('\n')

    class BatchwiseWandbLogger(tf.keras.callbacks.Callback):
        # 批次级Wandb记录器类，用于在每个训练batch结束时记录指标，并可选择记录权重
        def __init__(self, batch_frequency, log_weights=True, sample_at_zero=False):
            # 初始化函数，用于创建类的实例时设置初始值
            self.batch_frequency = batch_frequency  # 设置记录批次的频率
            self.log_weights = log_weights  # 是否记录模型权重
            self.sample_at_zero = sample_at_zero  # 是否在第一个batch时进行一次记录

        def on_train_batch_end(self, batch, logs=None):
            # 在每个训练batch结束时调用的方法
            # 判断是否进行记录的时机，根据batch_frequency和sample_at_zero来决定是否进行记录
            if (batch == 0 and self.sample_at_zero) or (batch + 1) % self.batch_frequency == 0:
                # 使用wandb.log方法记录指标
                wandb.log(logs)

                if self.log_weights:
                    # 记录模型权重信息
                    metrics = {}
                    for layer in self.model.layers:
                        weights = layer.get_weights()
                        if len(weights) == 1:
                            metrics["parameters/" + layer.name + ".weights"] = wandb.Histogram(weights[0])
                        elif len(weights) == 2:
                            metrics["parameters/" + layer.name + ".weights"] = wandb.Histogram(weights[0])
                            metrics["parameters/" + layer.name + ".bias"] = wandb.Histogram(weights[1])
                    # 将记录的权重信息加入到wandb日志中
                    wandb.log(metrics, commit=False)

    class BatchwiseModelCheckpoint(tf.keras.callbacks.Callback):
        # 批次级模型保存类，用于在每个训练batch结束时保存模型
        def __init__(self, save_frequency, model_path, mode, monitor, prev_best=None, sample_at_zero=False):
            # 初始化函数，用于创建类的实例时设置初始值
            self.save_frequency = save_frequency  # 设置保存模型的频率
            self.model_path = model_path  # 设置保存模型的路径
            self.mode = mode  # 设置保存模型的模式（最大值或最小值）
            self.monitor = monitor  # 设置用于判断模型是否改进的指标
            self.sample_at_zero = sample_at_zero  # 是否在第一个batch时进行一次保存

            if prev_best is not None:
                self.best = prev_best
            else:
                if self.mode == 'max':
                    self.best = -np.Inf
                elif self.mode == 'min':
                    self.best = np.Inf

        def on_train_batch_end(self, batch, logs=None):
            # 在每个训练batch结束时调用的方法
            # 判断是否进行保存的时机，根据save_frequency和sample_at_zero来决定是否进行保存
            if (batch == 0 and self.sample_at_zero) or (batch + 1) % self.save_frequency == 0:
                # 根据模式判断是否需要保存模型
                if self.mode == 'max' and logs[self.monitor] > self.best:
                    save = True
                elif self.mode == 'min' and logs[self.monitor] < self.best:
                    save = True
                else:
                    save = False

                if save:
                    # 保存模型并更新最佳指标值
                    print('\n{} improved from {:.3f} to {:.3f}. Saving model to {}.\n'.
                          format(self.monitor, self.best, logs[self.monitor], self.model_path))
                    self.best = logs[self.monitor]
                    self.model.save(self.model_path, overwrite=True)
                else:
                    # 不保存模型，打印提示信息
                    print(
                        '\n{}={:.3f} did not improve from {:.3f}\n'.format(self.monitor, logs[self.monitor], self.best))

    class IceNetCaseStudyCallback(tf.keras.callbacks.Callback):
        # IceNet案例研究回调类，用于在每个epoch结束时生成案例研究图像
        def __init__(self, dataloader, network_folder, seed, include_train=False):
            # 初始化函数，用于创建类的实例时设置初始值
            self.dataloader = dataloader  # 数据加载器
            self.include_train = include_train  # 是否包含训练数据的案例研究

            self.train_date = pd.Timestamp('2010-07-01')  # 训练日期
            self.val_date = pd.Timestamp('2012-07-01')  # 验证日期

            self.n_forecast_months = dataloader.n_forecast_months  # 预测月数

            if include_train:
                # 获取训练数据的输入、输出和掩码
                self.X_train, self.y_train, self.mask_train = dataloader.get_month_of_input_output_data(self.train_date)

                self.class_train = []
                self.class_train_vector = []
                for forecast_month_idx in range(self.n_forecast_months):
                    # 计算训练数据的类别信息
                    class_train = np.argmax(self.y_train[0, :, :, 1:, forecast_month_idx], axis=-1)
                    self.class_train.append(class_train)
                    self.class_train_vector.append(class_train[self.mask_train[forecast_month_idx, :]])

            # 获取验证数据的输入、输出和掩码
            self.X_val, self.y_val, self.mask_val = dataloader.get_month_of_input_output_data(self.val_date)

            self.class_val = []
            self.class_val_vector = []
            for forecast_month_idx in range(self.n_forecast_months):
                # 计算验证数据的类别信息
                class_val = np.argmax(self.y_val[0, :, :, 1:, forecast_month_idx], axis=-1)
                self.class_val.append(class_val)
                self.class_val_vector.append(class_val[self.mask_val[forecast_month_idx, :]])

            # 设置用于保存图像的文件夹路径
            self.folder = os.path.join(network_folder, '{}_training_case_studies'.format(seed))

            if not os.path.isdir(self.folder):
                os.makedirs(self.folder)

        def on_epoch_end(self, epoch, logs=None):
            # 在每个epoch结束时调用的方法，生成案例研究图像
            if self.include_train:
                y_pred_train = self.model.predict(self.X_train)

                self.class_pred_train = []
                self.class_pred_train_vector = []
                for forecast_month_idx in range(self.n_forecast_months):
                    class_pred_train = np.argmax(y_pred_train[0, :, :, :, forecast_month_idx], axis=-1)
                    class_pred_train[~self.mask_train[forecast_month_idx, :]] = 0
                    self.class_pred_train.append(class_pred_train)
                    self.class_pred_train_vector.append(class_pred_train[self.mask_train[forecast_month_idx, :]])

                self.train_acc = []
                for forecast_month_idx in range(self.n_forecast_months):
                    # 计算训练数据的准确率
                    self.train_acc.append(100 * accuracy_score(self.class_train_vector[forecast_month_idx],
                                                               self.class_pred_train_vector[forecast_month_idx]))

            y_pred_val = self.model.predict(self.X_val)

            self.class_pred_val = []
            self.class_pred_val_vector = []
            for forecast_month_idx in range(self.n_forecast_months):
                # 计算验证数据的类别信息和准确率
                class_pred_val = np.argmax(y_pred_val[0, :, :, :, forecast_month_idx], axis=-1)
                class_pred_val[~self.mask_val[forecast_month_idx, :]] = 0
                self.class_pred_val.append(class_pred_val)
                self.class_pred_val_vector.append(class_pred_val[self.mask_val[forecast_month_idx, :]])

            self.val_acc = []
            for forecast_month_idx in range(self.n_forecast_months):
                self.val_acc.append(100 * accuracy_score(self.class_val_vector[forecast_month_idx],
                                                         self.class_pred_val_vector[forecast_month_idx]))

            nrows = 4 if self.include_train else 2
            figsize = (60, 40) if self.include_train else (60, 20)

            fig, axes = plt.subplots(nrows=nrows, ncols=self.n_forecast_months, figsize=figsize)

            for forecast_month_idx in range(self.n_forecast_months):
                if self.include_train:
                    axes[0, forecast_month_idx].imshow(self.class_train[forecast_month_idx], cmap='Blues_r')
                    axes[0, forecast_month_idx].set_title(
                        (self.train_date + relativedelta(months=forecast_month_idx)).strftime('%Y_%m'), fontsize=30)
                    axes[1, forecast_month_idx].imshow(self.class_pred_train[forecast_month_idx], cmap='Blues_r')
                    axes[1, forecast_month_idx].set_xlabel('Acc: {:.2f}'.format(self.train_acc[forecast_month_idx]),
                                                           fontsize=30)

                    if forecast_month_idx == 0:
                        axes[0, forecast_month_idx].set_ylabel('True', fontsize=30)
                        axes[1, forecast_month_idx].set_ylabel('Pred', fontsize=30)

                    axes[2, forecast_month_idx].imshow(self.class_val[forecast_month_idx], cmap='Blues_r')
                    axes[2, forecast_month_idx].set_title(
                        (self.val_date + relativedelta(months=forecast_month_idx)).strftime('%Y_%m'), fontsize=30)
                    axes[3, forecast_month_idx].imshow(self.class_pred_val[forecast_month_idx], cmap='Blues_r')
                    axes[3, forecast_month_idx].set_xlabel('Acc: {:.2f}'.format(self.val_acc[forecast_month_idx]),
                                                           fontsize=30)

                    if forecast_month_idx == 0:
                        axes[2, forecast_month_idx].set_ylabel('True', fontsize=30)
                        axes[2, forecast_month_idx].set_ylabel('Pred', fontsize=30)
                else:
                    axes[0, forecast_month_idx].imshow(self.class_val[forecast_month_idx], cmap='Blues_r')
                    axes[0, forecast_month_idx].set_title(
                        (self.val_date + relativedelta(months=forecast_month_idx)).strftime('%Y_%m'), fontsize=30)
                    axes[1, forecast_month_idx].imshow(self.class_pred_val[forecast_month_idx], cmap='Blues_r')
                    axes[1, forecast_month_idx].set_xlabel('Acc: {:.2f}'.format(self.val_acc[forecast_month_idx]),
                                                           fontsize=30)

                    if forecast_month_idx == 0:
                        axes[0, forecast_month_idx].set_ylabel('True', fontsize=30)
                        axes[1, forecast_month_idx].set_ylabel('Pred', fontsize=30)

            for ax in axes.ravel():
                # 在图像上添加地图的边界
                ax.contourf(np.load(self.dataloader.variable_paths['land']), levels=[0, 1], colors='gray',
                            extend='neither')

            plt.suptitle('Epoch {}'.format(epoch), fontsize=40)

            plt.tight_layout(pad=3.)

            # 将生成的图像保存到文件夹中
            plt.savefig(os.path.join(self.folder, '{}.pdf'.format(epoch)))
            plt.close()

        def on_train_begin(self, logs=None):
            # 在训练开始时调用的方法，用于生成train_begin的案例研究图像
            self.on_epoch_end(epoch='train_begin')

