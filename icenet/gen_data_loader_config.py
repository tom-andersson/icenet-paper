import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))
import config
import pandas as pd
import json

"""

Script to generate a .json file storing data loader configuration settings using
a dictionary. The dictionary settings are:

    dataloader_ID (str): Name for these data loader configuration settings,
    used as the filename for the config .json file.

    dataset_name (str): Name for of the network dataset (generated with
    preproc_icenet_data.py) to load data from.

    input_data (dict): Dictionary of dictionaries dictating which
    variables to include for IceNet2's input 3D volumes and, if appropriate,
    a maximum lag (in days) to grab the data from. The nested dictionaries
    have keys of "include" (a bool for whether to input that variable), and
    "lookbacks" (a list of ints for which past months to input, indexing
    from 0 relative to the most recent month).

        Example:
            'input_data': {
                "siconca":
                    {"abs": {"include": True, 'max_lag': 12},
                     "anom": {"include": False, 'max_lag': 3},
                     "linear_trend": {"include": True}},
                "tas":
                    {"abs": {"include": False, 'max_lag': 3},
                     "anom": {"include": True, 'max_lag': 3}},
                "ta500":
                    {"abs": {"include": False, 'max_lag': 3},
                     "anom": {"include": True, 'max_lag': 3}},
                "tos":
                    {"abs": {"include": False, 'max_lag': 3},
                     "anom": {"include": True, 'max_lag': 3}},
                "psl":
                    {"abs": {"include": False, 'max_lag': 3},
                     "anom": {"include": True, 'max_lag': 3}},
                "zg500":
                    {"abs": {"include": False, 'max_lag': 3},
                     "anom": {"include": True, 'max_lag': 3}},
                "zg250":
                    {"abs": {"include": False, 'max_lag': 3},
                     "anom": {"include": True, 'max_lag': 3}},
                "uas":
                    {"abs": {"include": True, 'max_lag': 1},
                     "anom": {"include": False, 'max_lag': 1}},
                "vas":
                    {"abs": {"include": True, 'max_lag': 1},
                     "anom": {"include": False, 'max_lag': 1}},
                "land":
                    {"metadata": True,
                     "include": True},
                "circmonth":
                    {"metadata": True,
                     "include": True},
            },
    batch_size (int): Number of samples per training batch.

    shuffle (bool): Whether to shuffle the training samples.

    n_forecast_months (int): Total number of days ahead to predict.

    sample_IDs (dict): Dictionary of dictionaries storing the train-val-test
    set splits. Splits are defined in terms of start & end dates for the
    forecast initialisation dates used to define sampled IDs.

        Example:
            'sample_IDs': {
                'obs_train_dates': ('1980-1-1', '2011-6-1'),
                'obs_val_dates': ('2012-1-1', '2017-6-1'),
                'obs_test_dates': ('2018-1-1', '2020-10-1'),
            },

    cmip6_run_dict (dict): Dictionary of tuples storing the CMIP6 runs
    from each climate model (and their valid start and end forecast dates)
    to use for pre-training IceNet.

        Example:
            'cmip6_run_dict': {
                'EC-Earth3': (
                    ('r2i1p1f1', '1851-1-1', '2099-6-1'),
                    ('r7i1p1f1', '1851-1-1', '2099-6-1'),
                    ('r10i1p1f1', '1851-1-1', '2099-6-1'),
                    ('r12i1p1f1', '1851-1-1', '2099-6-1'),
                    ('r14i1p1f1' '1851-1-1', '2099-6-1'),
                ),
                'MRI-ESM2-0': (
                    ('r1i1p1f1', '1851-1-1', '2099-6-1'),
                    ('r2i1p1f1', '1851-1-1', '2029-6-1'),
                    ('r3i1p1f1', '1851-1-1', '2029-6-1'),
                    ('r4i1p1f1', '1851-1-1', '2029-6-1'),
                    ('r5i1p1f1' '1851-1-1', '2029-6-1')
                ),
            },

    raw_data_shape (tuple): Shape of input satellite data as (rows, cols).

    default_seed (int): Default random seed to use for shuffling the order
    of training samples a) before training, and b) after each training epoch.

    loss_weight_months (bool): Whether to weight the loss function for different
    months based on the size of the active grid cell mask.

    verbose_level (int): Controls how much to print. 0: Print nothing.
    1: Print key set-up stages. 2: Print debugging info. 3: Print when an
    output month is skipped due to missing data.
"""

dataloder_config = {
    'dataloader_ID': 'icenet_nature_communications',
    'dataset_name': 'dataset1',
    'input_data': {
        "siconca":
            {"abs": {"include": True, 'max_lag': 12},
             "anom": {"include": False, 'max_lag': 3},
             "linear_trend": {"include": True}},
        "tas":
            {"abs": {"include": False, 'max_lag': 3},
             "anom": {"include": True, 'max_lag': 3}},
        "ta500":
            {"abs": {"include": False, 'max_lag': 3},
             "anom": {"include": True, 'max_lag': 3}},
        "tos":
            {"abs": {"include": False, 'max_lag': 3},
             "anom": {"include": True, 'max_lag': 3}},
        "rsds":
            {"abs": {"include": False, 'max_lag': 3},
             "anom": {"include": True, 'max_lag': 3}},
        "rsus":
            {"abs": {"include": False, 'max_lag': 3},
             "anom": {"include": True, 'max_lag': 3}},
        "psl":
            {"abs": {"include": False, 'max_lag': 3},
             "anom": {"include": True, 'max_lag': 3}},
        "zg500":
            {"abs": {"include": False, 'max_lag': 3},
             "anom": {"include": True, 'max_lag': 3}},
        "zg250":
            {"abs": {"include": False, 'max_lag': 3},
             "anom": {"include": True, 'max_lag': 3}},
        "ua10":
            {"abs": {"include": True, 'max_lag': 3},
             "anom": {"include": False, 'max_lag': 3}},
        "uas":
            {"abs": {"include": True, 'max_lag': 1},
             "anom": {"include": False, 'max_lag': 1}},
        "vas":
            {"abs": {"include": True, 'max_lag': 1},
             "anom": {"include": False, 'max_lag': 1}},
        "land":
            {"metadata": True,
             "include": True},
        "circmonth":
            {"metadata": True,
             "include": True},
    },
    'batch_size': 2,
    'shuffle': True,
    'n_forecast_months': 6,
    'sample_IDs': {
        'obs_train_dates': ('1980-1-1', '2011-6-1'),
        'obs_val_dates': ('2012-1-1', '2017-6-1'),
        'obs_test_dates': ('2018-1-1', '2019-6-1'),
    },
    'cmip6_run_dict': {
        'EC-Earth3': {
            'r2i1p1f1': ('1851-1-1', '2099-6-1'),
            'r7i1p1f1': ('1851-1-1', '2099-6-1'),
            'r10i1p1f1': ('1851-1-1', '2099-6-1'),
            'r12i1p1f1': ('1851-1-1', '2099-6-1'),
            'r14i1p1f1': ('1851-1-1', '2099-6-1'),
        },
        'MRI-ESM2-0': {
            'r1i1p1f1': ('1851-1-1', '2099-6-1'),
            'r2i1p1f1': ('1851-1-1', '2029-6-1'),
            'r3i1p1f1': ('1851-1-1', '2029-6-1'),
            'r4i1p1f1': ('1851-1-1', '2029-6-1'),
            'r5i1p1f1': ('1851-1-1', '2029-6-1'),
        },
    },
    'raw_data_shape': (432, 432),
    'default_seed': 42,
    'loss_weight_months': True,
    'verbose_level': 0,
}

now = pd.Timestamp.now()
fname = now.strftime('%Y_%m_%d_%H%M_{}.json').format(dataloder_config['dataloader_name'])
fpath = os.path.join(config.dataloader_config_folder, fname)
if not os.path.exists(config.dataloader_config_folder):
    os.makedirs(config.dataloader_config_folder)

with open(fpath, 'w') as outfile:
    json.dump(dataloder_config, outfile)

print('Data loader config saved to {}\n'.format(fpath))
print('Data loader name: {}'.format(fname[:-5]))
