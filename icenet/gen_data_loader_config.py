import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))
import config
import pandas as pd
import json

"""
Script to generate a JSON file storing data loader configuration settings.
The JSON file is used to instantiate a IceNetDataLoader objects to generate
IceNet input-output data (e.g. during training or prediction) with specified
input data, number of forecast months, and loss function sample weighting.

A timestamp is prepended to the 'dataloader_name' entry to define a unique
`dataloader_ID` for the configuration settings generated. The JSON file is saved
in the format dataloader_configs/<dataloader_ID>.json. The `dataloader_ID` is
then used in downstream scripts to specify individual dataloader configurations.
It is also used to generate a unique `icenet_ID` for a particular
data/architecture combination: icenet_ID = dataloader_ID + architecture_ID (here
the '+' is conceptual), which allows different IceNet ensembles to be separated
and identified in the project folder hierarchy and results datasets.

The configuration settings are given below.
_______________________________________________________________________________

dataloader_name (str): Name for these data loader configuration settings, used as
the filename for the config .json file.

dataset_name (str): Name of the network dataset (generated with
icenet/preproc_icenet_data.py) to load data from.

input_data (dict): Data structure dictating which variables to include for
IceNet's input data and, if appropriate, a maximum lag (in months) to grab the
data from. The nested dictionaries have keys of "include" (a bool for whether to
input that variable), and "max_lag" (an int for how mnay past months to input).
Metadata variables like the land mask and circular month inputs are identified
with a `'metadata': True` key-val pair.

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

batch_size (int): Number of samples(/forecasts) per training batch.

shuffle (bool): Whether to shuffle the training samples before each epoch.

n_forecast_months (int): Total number of months ahead to predict.

sample_IDs (dict): Data structure storing the train-val-test
set splits. Splits are defined in terms of start & end dates (inclusive) for the
forecast initialisation dates used to define sampled IDs. Forecast initialisation
date is defined as the *first month being forecast*.

    Example:
        'sample_IDs': {
            'obs_train_dates': ('1980-1-1', '2011-6-1'),
            'obs_val_dates': ('2012-1-1', '2017-6-1'),
            'obs_test_dates': ('2018-1-1', '2020-10-1'),
        },

cmip6_run_dict (dict): Data structure storing the CMIP6 runs
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
of training samples before each training epoch.

loss_weight_months (bool): Whether to weight the samples in the loss function
for different target calendar months based on the size of the active grid cell
mask.

verbose_level (int): Controls how much to print. 0: Print nothing.
1: Print key set-up stages. 2: Print debugging info. 3: Print when an
output month is skipped due to missing data.
"""

dataloader_config = {
    'dataloader_name': 'icenet_nature_communications',
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
fname = now.strftime('%Y_%m_%d_%H%M_{}.json').format(dataloader_config['dataloader_name'])
fpath = os.path.join(config.dataloader_config_folder, fname)
if not os.path.exists(config.dataloader_config_folder):
    os.makedirs(config.dataloader_config_folder)

with open(fpath, 'w') as outfile:
    json.dump(dataloader_config, outfile)

print('Data loader config saved to {}\n'.format(fpath))
print('Data loader name: {}'.format(fname[:-5]))
