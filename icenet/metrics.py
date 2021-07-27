import os
import sys
import tensorflow as tf
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))  # if using jupyter kernel

'''
TensorFlow metrics.
'''


class ConstructLeadtimeAccuracy(tf.keras.metrics.CategoricalAccuracy):

    ''' Computes the network's accuracy over the active grid cell region
    for either a) a specific lead time in months, or b) over all lead times
    at once. '''

    def __init__(self,
                 name='construct_custom_categorical_accuracy',
                 use_all_forecast_months=True,
                 single_forecast_leadtime_idx=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.use_all_forecast_months = use_all_forecast_months
        self.single_forecast_leadtime_idx = single_forecast_leadtime_idx

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.use_all_forecast_months:
            # Make class dimension final dimension for CategoricalAccuracy
            y_true = tf.transpose(y_true, [0, 1, 2, 4, 3])
            y_pred = tf.transpose(y_pred, [0, 1, 2, 4, 3])
            if sample_weight is not None:
                sample_weight = tf.transpose(sample_weight, [0, 1, 2, 4, 3])

            super().update_state(
                y_true, y_pred, sample_weight=sample_weight)

        elif not self.use_all_forecast_months:

            super().update_state(
                y_true[..., self.single_forecast_leadtime_idx],
                y_pred[..., self.single_forecast_leadtime_idx],
                sample_weight=sample_weight[..., self.single_forecast_leadtime_idx]>0)

    def result(self):
        return 100 * super().result()

    def get_config(self):
        ''' For saving and loading networks with this custom metric. '''
        return {
            'single_forecast_leadtime_idx': self.single_forecast_leadtime_idx,
            'use_all_forecast_months': self.use_all_forecast_months,
        }

    @classmethod
    def from_config(cls, config):
        ''' For saving and loading networks with this custom metric. '''
        return cls(**config)
