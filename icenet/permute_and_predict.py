import sys
import os
os.environ["OMP_NUM_THREADS"] = "16"
import numpy as np
import xarray as xr
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))  # if using jupyter kernel
from utils import IceNetDataLoader
import config
import re
import pandas as pd
from tqdm import tqdm
import time
from tensorflow.keras.models import load_model

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
pd.options.display.float_format = '{:.2f}'.format

'''
Runs the permute-and-predict algorithm for IceNet and saves results to
`results/permute_and_predict_results/permute_and_predict_results.csv`.

First, the desired subset of IceNet ensemble members are loaded, the
of unpermuted IceNet input tensor and ground truth outputs are generated from
the data loader, and IceNet's forecasts with unpermuted inputs are generated.

Next, the baseline accuracy for each target date and lead time is computed
by comparing xr.DataArrays of the forecasts with ground truth. Using xarray
results in a `ds_acc_unpermuted` xr.Dataset whose accuracy can conveniently be
queried at any target date and lead time.

Then, the permute-and-predict algorithm is run with an outer loop over random
permutation seeds and an inner loop over IceNet's input variables. The
input tensor is permuted over the time index for each variable, IceNet's forecasts
are regenerated, and the accuracy drop relative to the baseline is computed
and stored in a DataFrame with indexes (Seed, Leadtime, Forecast date, Variable).
The results are checkpointed after each outer seed loop iteration.
'''

################################################################################
#################### User input section
################################################################################

dataloader_ID = '2021_06_15_1854_icenet_nature_communications'
architecture_ID = 'unet_tempscale'

# Number of times to run the PaP method with different permutations (for averaging)
n_runs = 10

# Number of samples per batch for running IceNet predictions over the validation set
#   (reduce for smaller GPU memory)
batch_size = 8

# Which ensemble network seeds to use for the saliency analysis. 'all' or a list
#   of integers.
ensemble_seeds = [38, 42, 43, 44, 46]

temp_scaling_used = True  # Whether or not the network being loaded used temperature scaling

# Note: 2020 data wasn't used at the time of this experiment, but is used
#   in the rest of the paper
heldout_start = '2012-01-01'
heldout_end = '2019-12-01'


################################################################################
#################### Functions
################################################################################


def run_icenet_on_numpy(input_arr, ensemble, batch_size=8):
    '''
    Computes a numpy array of IceNet ensemble-mean ice class forecasts using an
    numpy array of input tensors.

    Inputs:
    input_arr (np.ndarray): Shape (n_forecast_start_dates, n_x, n_y, n_input_vars)
    ensemble: List of TensorFlow networks to compute ensemble mean over
    batch_size (int): Number of samples to predict in parallel on the GPU

    Returns:
    all_icenet_preds (np.ndarray): Shape (n_forecast_start_dates, n_x, n_y,
    n_classes, n_leadtimes) of ice class index prediciont (0, 1, or 2).
    '''

    num_batches = int(np.ceil(input_arr.shape[0] / batch_size))

    all_icenet_preds = []
    for batch_idx in range(num_batches):

        batch_start = batch_idx * batch_size
        batch_end = np.min([(batch_idx + 1) * batch_size, len(init_dates)])
        inputs = input_arr[batch_start:batch_end]

        network_preds = []
        for network in ensemble:
            network_preds.append(network.predict(inputs))

        all_icenet_preds.append(np.array(network_preds))

    all_icenet_preds = np.mean(np.concatenate(all_icenet_preds, axis=1), axis=0)
    all_icenet_preds = all_icenet_preds.argmax(axis=-2).astype(int)

    return all_icenet_preds


def numpy_to_xarray(arr, init_dates, target_dates, n_forecast_months):

    '''
    Converts input NumPy array to an xarray.DataArray with dimensions
    (time, y, x, leadtime). `arr` can either be: a) IceNet forecast ice class
    index predictions (0, 1, or 2), or b) set of ground truth SIC values from
    the data loader.

    Inputs:

    arr: Input NumPy array to convert to xr.DataArray.
    init_dates: List of forecast initialisation dates of `arr`.
    target_dates: List of desired target dates to store in the DataArray.
    n_forecast_months: Maximum lead time of the forecasts.

    Returns:
    ds: xr.DataArray of IceNet forecasts with dimensions (time, y, x, leadtime).
    '''

    leadtimes = np.arange(1, n_forecast_months + 1)

    coords = {
        'time': target_dates,
        'y': range(432),
        'x': range(432),
        'leadtime': np.arange(1, n_forecast_months+1),
    }

    shape = (len(target_dates), 432, 432, n_forecast_months)
    dims = ('time', 'y', 'x', 'leadtime')

    ds = xr.DataArray(
        data=np.zeros(shape, dtype=int),
        coords=coords,
        dims=dims
    )

    for init_date_idx, init_date in enumerate(init_dates):

        forecast_target_dates = pd.date_range(
            start=init_date,
            end=init_date+pd.DateOffset(months=n_forecast_months),
            freq='MS'
        )

        for leadtime_idx, (target_date, leadtime) in \
                enumerate(zip(forecast_target_dates, leadtimes)):

            if target_date in target_dates:
                ds.loc[target_date, :, :, leadtime] = \
                    arr[init_date_idx, ..., leadtime_idx]

    return ds


################################################################################
#################### Set up folder structure
################################################################################

dataloader_config_fpath = os.path.join(config.dataloader_config_folder, dataloader_ID+'.json')
dataloader_ID_folder = os.path.join(config.networks_folder, dataloader_ID)
icenet_folder = os.path.join(dataloader_ID_folder, architecture_ID)
network_h5_files_folder = os.path.join(icenet_folder, 'networks')

pap_results_df_fpath = os.path.join(config.permute_and_predict_results_folder,
                                    'permute_and_predict_results.csv')

if not os.path.exists(config.permute_and_predict_results_folder):
    os.makedirs(config.permute_and_predict_results_folder)

################################################################################
#################### Load data loader
################################################################################

print("\nSetting up the data loader with config file: {}\n\n".format(dataloader_ID))
val_dataloader = IceNetDataLoader(dataloader_config_fpath)
val_dataloader.convert_to_validation_data_loader()
print('\n\nDone.\n')

print("Done.\n\n")
all_ordered_variable_names = val_dataloader.determine_variable_names()
print("Variable names: {}\n\n".format(all_ordered_variable_names))

n_forecast_months = val_dataloader.config['n_forecast_months']
leadtimes = np.arange(1, n_forecast_months + 1)

# Months to compute accuracy drop over
target_dates = pd.date_range(
    start=heldout_start,
    end=heldout_end,
    freq='MS'
)

# Forecast initialisation dates s.t. each target date has a forecast at each lead time
init_dates = pd.date_range(
    start=pd.Timestamp(heldout_start) - pd.DateOffset(months=n_forecast_months-1),
    end=heldout_end,
    freq='MS'
)

print('\nComputing permutation accuracy drops over dates: {}\n\n'.format(target_dates))

# Land mask
land_mask_path = os.path.join(config.mask_data_folder, config.land_mask_filename)
land_mask = np.load(land_mask_path)

months = [pd.Timestamp(date).month for date in target_dates]

mask_fpath_format = os.path.join(config.mask_data_folder, config.active_grid_cell_file_format)
mask_da = xr.DataArray(
    [np.load(mask_fpath_format.format('{:02d}'.format(month))) for month in months],
    dims=('time', 'y', 'x'),
    coords={
        'time': target_dates,
        'y': range(432),
        'x': range(432),
    }
)

################################################################################
#################### Load networks
################################################################################

print("\n\nLoading the IceNet network/s... ", end='', flush=True)

if temp_scaling_used:
    network_regex = re.compile(r'^network_tempscaled_([0-9]*)\.h5$')
else:
    network_regex = re.compile(r'^network_([0-9]*)\.h5$')

filenames = sorted(os.listdir(network_h5_files_folder))
filenames = [filename for filename in filenames if network_regex.match(filename)]
network_paths = [os.path.join(network_h5_files_folder, filename) for filename in filenames]

regex_match_objects = [network_regex.match(filename) for filename in filenames]
ensemble_seeds = [int(match.group(1)) for match in regex_match_objects]
num_ensemble_networks = len(ensemble_seeds)

ensemble = []
for path, network_seed in zip(network_paths, ensemble_seeds):
    if ensemble_seeds != 'all':
        if not np.isin(network_seed, ensemble_seeds):
            continue  # Skip this network - not an ensemble member to use for saliency
    print('Loading network {}... '.format(path), end='', flush=True)
    ensemble.append(load_model(path, compile=False))
    print('Done.')

print("\nLoaded {} networks with seeds: {}\n\n".format(len(ensemble), ensemble_seeds))

################################################################################
#################### Build up all inputs from the held-out data
################################################################################

print('Building up all the baseline inputs and ground truth class maps... ')
unpermuted_inputs_list = []
true_outputs_list = []
for forecast_start_date in init_dates:
    inputs, outputs, _ = val_dataloader.data_generation(forecast_start_date)
    true_outputs_list.append(outputs[0])
    unpermuted_inputs_list.append(inputs[0])
unpermuted_inputs = np.stack(unpermuted_inputs_list, axis=0)
true_outputs = np.stack(true_outputs_list, axis=0)
true_outputs = true_outputs.argmax(axis=-2)

true_ds = numpy_to_xarray(
    true_outputs, init_dates, target_dates, n_forecast_months)
# True SIC doesn't vary with lead time
true_ds = true_ds.isel(leadtime=0).drop('leadtime')
print('Done.\n\n')

################################################################################
#################### Build up all the IceNet predictions with no permutation
#################### and compute unpermuted baseline accuracy
################################################################################

print('Building up all the IceNet predictions...\n')
tic = time.time()
icenet_preds_unpermuted = run_icenet_on_numpy(unpermuted_inputs, ensemble,
                                              batch_size)
icenet_preds_unpermuted_ds = numpy_to_xarray(
    icenet_preds_unpermuted, init_dates, target_dates, n_forecast_months)
dur = time.time() - tic
print("Done in {}m:{:.0f}s\n".format(np.floor(dur / 60), dur % 60))

# Compute percentage of correct classifications over the active
#   grid cell area
correct_da = (icenet_preds_unpermuted_ds == true_ds).astype(np.float32)
correct_weighted_da = correct_da.weighted(mask_da)
ds_acc_unpermuted = (correct_weighted_da.mean(dim=['y', 'x']) * 100)

################################################################################
#################### Run permute-and-predict
################################################################################

num_vars = len(all_ordered_variable_names)
num_samples = unpermuted_inputs.shape[0]

# Random seeds for permuting the inputs
seeds = range(n_runs)

# DataFrame for storing results
multi_index = pd.MultiIndex.from_product(
    [all_ordered_variable_names, seeds, leadtimes, target_dates],
    names=['Seed', 'Leadtime', 'Forecast date', 'Variable'])
pap_results_df = pd.DataFrame(
    index=multi_index, columns=['Accuracy drop (%)'], dtype=np.float32)

print('\nRunning permute and predict (progress bar over seeds below):\n\n')

for seed in tqdm(seeds):

    rng = np.random.default_rng(seed)
    permutation_idxs = rng.choice(num_samples, num_samples, replace=False)

    for var_i, varname in enumerate(tqdm(all_ordered_variable_names, leave=False)):

        permuted_inputs = np.copy(unpermuted_inputs)
        permuted_inputs[:, :, :, var_i] = permuted_inputs[permutation_idxs, :, :, var_i]

        icenet_preds_permuted = run_icenet_on_numpy(permuted_inputs, ensemble,
                                                    batch_size)
        icenet_preds_permuted_ds = numpy_to_xarray(
            icenet_preds_permuted, init_dates, target_dates, n_forecast_months)

        # Compute percentage of correct classifications over the active
        #   grid cell area for each target date and lead time
        correct_da = (icenet_preds_permuted_ds == true_ds).astype(np.float32)
        correct_weighted_da = correct_da.weighted(mask_da)
        ds_acc_permuted = (correct_weighted_da.mean(dim=['y', 'x']) * 100)
        ds_acc_drop = ds_acc_permuted - ds_acc_unpermuted

        for target_date in target_dates:
            for leadtime in leadtimes:
                pap_results_df.loc[seed, leadtime, target_date, varname] = \
                    ds_acc_drop.sel(time=target_date, leadtime=leadtime).data

    print('\nDone.\n')

    # Save permute-and-predict results dataframe _within_ seed loop to checkpoint results
    pap_results_df.to_csv(pap_results_df_fpath)

print('Done')
