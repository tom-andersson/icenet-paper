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
        self.validation_frequency = validation_frequency
        self.val_dataloader = val_dataloader
        self.sample_at_zero = sample_at_zero

    def on_train_batch_end(self, batch, logs=None):

        if (batch == 0 and self.sample_at_zero) or (batch + 1) % self.validation_frequency == 0:
            val_logs = self.model.evaluate(self.val_dataloader, verbose=0, return_dict=True)
            val_logs = {'val_' + name: val for name, val in val_logs.items()}
            logs.update(val_logs)
            # wandb.log(logs)
            [print('\n' + k + ' {:.2f}'.format(v)) for k, v in logs.items()]
            print('\n')


class BatchwiseWandbLogger(tf.keras.callbacks.Callback):
    """
    Docstring TODO
    """

    def __init__(self, batch_frequency, log_weights=True, sample_at_zero=False):
        self.batch_frequency = batch_frequency
        self.log_weights = log_weights
        self.sample_at_zero = sample_at_zero

    def on_train_batch_end(self, batch, logs=None):

        if (batch == 0 and self.sample_at_zero) or (batch + 1) % self.batch_frequency == 0:
            wandb.log(logs)

            if self.log_weights:
                # Taken from
                metrics = {}
                for layer in self.model.layers:
                    weights = layer.get_weights()
                    if len(weights) == 1:
                        metrics["parameters/" + layer.name +
                                ".weights"] = wandb.Histogram(weights[0])
                    elif len(weights) == 2:
                        metrics["parameters/" + layer.name +
                                ".weights"] = wandb.Histogram(weights[0])
                        metrics["parameters/" + layer.name +
                                ".bias"] = wandb.Histogram(weights[1])
                wandb.log(metrics, commit=False)


class BatchwiseModelCheckpoint(tf.keras.callbacks.Callback):
    """
    Docstring TODO
    """

    def __init__(self, save_frequency, model_path, mode, monitor, prev_best=None, sample_at_zero=False):
        self.save_frequency = save_frequency
        self.model_path = model_path
        self.mode = mode
        self.monitor = monitor
        self.sample_at_zero = sample_at_zero

        if prev_best is not None:
            self.best = prev_best

        else:
            if self.mode == 'max':
                self.best = -np.Inf
            elif self.mode == 'min':
                self.best = np.Inf

    def on_train_batch_end(self, batch, logs=None):

        if (batch == 0 and self.sample_at_zero) or (batch + 1) % self.save_frequency == 0:
            if self.mode == 'max' and logs[self.monitor] > self.best:
                save = True

            elif self.mode == 'min' and logs[self.monitor] < self.best:
                save = True

            else:
                save = False

            if save:
                print('\n{} improved from {:.3f} to {:.3f}. Saving model to {}.\n'.
                      format(self.monitor, self.best, logs[self.monitor], self.model_path))

                self.best = logs[self.monitor]

                self.model.save(self.model_path, overwrite=True)
            else:
                print('\n{}={:.3f} did not improve from {:.3f}\n'.format(self.monitor, logs[self.monitor], self.best))


class IceNetCaseStudyCallback(tf.keras.callbacks.Callback):

    def __init__(self, dataloader, network_folder, seed, include_train=False):
        self.dataloader = dataloader
        self.include_train = include_train

        self.train_date = pd.Timestamp('2010-07-01')
        self.val_date = pd.Timestamp('2012-07-01')

        self.n_forecast_months = dataloader.n_forecast_months

        if include_train:
            self.X_train, self.y_train, self.mask_train = dataloader.get_month_of_input_output_data(self.train_date)

            self.class_train = []
            self.class_train_vector = []
            for forecast_month_idx in range(self.n_forecast_months):
                class_train = np.argmax(self.y_train[0, :, :, 1:, forecast_month_idx], axis=-1)
                self.class_train.append(class_train)
                self.class_train_vector.append(class_train[self.mask_train[forecast_month_idx, :]])

        self.X_val, self.y_val, self.mask_val = dataloader.get_month_of_input_output_data(self.val_date)

        self.class_val = []
        self.class_val_vector = []
        for forecast_month_idx in range(self.n_forecast_months):
            class_val = np.argmax(self.y_val[0, :, :, 1:, forecast_month_idx], axis=-1)
            self.class_val.append(class_val)
            self.class_val_vector.append(class_val[self.mask_val[forecast_month_idx, :]])

        # self.class_val = np.argmax(self.y_val[0, :, :, 1:], axis=-1)
        # self.class_val_vector = self.class_val[self.mask_val]

        self.folder = os.path.join(network_folder, '{}_training_case_studies'.format(seed))

        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

    def on_epoch_end(self, epoch, logs=None):

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
                self.train_acc.append(100*accuracy_score(self.class_train_vector[forecast_month_idx],
                                                         self.class_pred_train_vector[forecast_month_idx]))

        y_pred_val = self.model.predict(self.X_val)

        self.class_pred_val = []
        self.class_pred_val_vector = []
        for forecast_month_idx in range(self.n_forecast_months):
            class_pred_val = np.argmax(y_pred_val[0, :, :, :, forecast_month_idx], axis=-1)
            class_pred_val[~self.mask_val[forecast_month_idx, :]] = 0
            self.class_pred_val.append(class_pred_val)
            self.class_pred_val_vector.append(class_pred_val[self.mask_val[forecast_month_idx, :]])

        self.val_acc = []
        for forecast_month_idx in range(self.n_forecast_months):
            self.val_acc.append(100*accuracy_score(self.class_val_vector[forecast_month_idx],
                                                   self.class_pred_val_vector[forecast_month_idx]))

        nrows = 4 if self.include_train else 2
        figsize = (60, 40) if self.include_train else (60, 20)

        fig, axes = plt.subplots(nrows=nrows, ncols=self.n_forecast_months, figsize=figsize)

        for forecast_month_idx in range(self.n_forecast_months):
            if self.include_train:
                axes[0, forecast_month_idx].imshow(self.class_train[forecast_month_idx], cmap='Blues_r')
                axes[0, forecast_month_idx].set_title((self.train_date+relativedelta(months=forecast_month_idx)).strftime('%Y_%m'), fontsize=30)
                axes[1, forecast_month_idx].imshow(self.class_pred_train[forecast_month_idx], cmap='Blues_r')
                axes[1, forecast_month_idx].set_xlabel('Acc: {:.2f}'.format(self.train_acc[forecast_month_idx]), fontsize=30)

                if forecast_month_idx == 0:
                    axes[0, forecast_month_idx].set_ylabel('True', fontsize=30)
                    axes[1, forecast_month_idx].set_ylabel('Pred', fontsize=30)

                axes[2, forecast_month_idx].imshow(self.class_val[forecast_month_idx], cmap='Blues_r')
                axes[2, forecast_month_idx].set_title((self.val_date+relativedelta(months=forecast_month_idx)).strftime('%Y_%m'), fontsize=30)
                axes[3, forecast_month_idx].imshow(self.class_pred_val[forecast_month_idx], cmap='Blues_r')
                axes[3, forecast_month_idx].set_xlabel('Acc: {:.2f}'.format(self.val_acc[forecast_month_idx]), fontsize=30)

                if forecast_month_idx == 0:
                    axes[2, forecast_month_idx].set_ylabel('True', fontsize=30)
                    axes[2, forecast_month_idx].set_ylabel('Pred', fontsize=30)
            else:
                axes[0, forecast_month_idx].imshow(self.class_val[forecast_month_idx], cmap='Blues_r')
                axes[0, forecast_month_idx].set_title((self.val_date+relativedelta(months=forecast_month_idx)).strftime('%Y_%m'), fontsize=30)
                axes[1, forecast_month_idx].imshow(self.class_pred_val[forecast_month_idx], cmap='Blues_r')
                axes[1, forecast_month_idx].set_xlabel('Acc: {:.2f}'.format(self.val_acc[forecast_month_idx]), fontsize=30)

                if forecast_month_idx == 0:
                    axes[0, forecast_month_idx].set_ylabel('True', fontsize=30)
                    axes[1, forecast_month_idx].set_ylabel('Pred', fontsize=30)

        for ax in axes.ravel():
            ax.contourf(np.load(self.dataloader.variable_paths['land']),
                        levels=[0, 1], colors='gray', extend='neither')

        plt.suptitle('Epoch {}'.format(epoch), fontsize=40)

        plt.tight_layout(pad=3.)

        plt.savefig(os.path.join(self.folder, '{}.pdf'.format(epoch)))
        plt.close()

    def on_train_begin(self, logs=None):
        self.on_epoch_end(epoch='train_begin')
