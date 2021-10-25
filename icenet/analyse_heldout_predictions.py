import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))
import utils
import config
import itertools
import numpy as np
import distributed
import dask
import pprint
import re
import xarray as xr
import pandas as pd
from time import time

'''
Loads forecast and ground truth NetCDFs with xarray, analyses forecast
performance with dask & distributed, and writes the results to a CSV file using
pandas. The forecast results CSV is saved to `results/forecast_results/` with
filename format `<timestamp>_forecast_results.csv` to avoid overwriting previous
results.

The spatially-averaged metrics are computed in a parallel & distributed fashion
using dask to significantly speed up computation and reduce memory load. This
works by lazily loading the forecast data, building a computation graph, and
iteratively solving the task graph using multiple workers. Optionally turn
off processing with dask and perform the computations in memory by changing the
bools in the user input section.

The xr.DataArray.weighted method is used to compute metrics only over the active
grid cell region.

Optionally pre-load the most recent results CSV file to append new models or
metrics to an existing results dataset. This allows you to include newer
versions of IceNet to compare with the benchmarks and previous IceNet versions
without having to re-analyse all the other models.

The indexes of the `results_df` pd.DataFrame are: ['Model', 'Ensemble member',
'Leadtime', 'Forecast date'].

A dictionary mapping models to lists of metrics to compute (`compute_dict`) is
determined based on the lists of models and metrics in the user input section.
If pre-loading a results CSV, only new models and new metrics not already in the
DataFrame are added to `compute_dict`.

For IceNet, each ensemble member is analysed, as well as the ensemble mean.
This allows the performance of individual ensemble members to be assessed, as
well as the improvement in performance due to ensemble-averaging. For the linear
trend and ensemble-mean SEAS5 model, the 'Ensemble member' entry is filled as 'NA'.

Authors: Tom Andersson, with dask help from James Byrne (BAS).
'''

n_workers = 8
threads_per_worker = 2
temp_dir = '/local/tmp'
dask.config.set(temporary_directory=os.path.expandvars(temp_dir))

####################################################################

if __name__ == "__main__":

    ### User input
    ####################################################################

    compute_in_memory = False
    compute_with_dask = True

    dataloader_ID = '2021_06_15_1854_icenet_nature_communications'
    # dataloader_ID = '2021_06_30_0954_icenet_pretrain_ablation'
    architecture_ID = 'unet_tempscale'

    # Format for storing different IceNet results in one dataframe
    icenet_ID = 'IceNet__{}__{}'.format(dataloader_ID, architecture_ID)

    model_compute_list = [icenet_ID, 'SEAS5', 'Linear trend']
    metric_compute_list = ['Binary accuracy', 'SIE error']

    pre_load_results_df = False

    ### Setup
    ####################################################################

    if np.sum([compute_in_memory, compute_with_dask]) != 1:
        raise ValueError('You must specify a single compute strategy '
                         'at the beginning of the script.')

    dataloader_config_fpath = os.path.join(config.dataloader_config_folder, dataloader_ID+'.json')

    # Data loader
    dataloader = utils.IceNetDataLoader(dataloader_config_fpath)

    n_forecast_months = dataloader.config['n_forecast_months']
    print('\n# of forecast months: {}\n'.format(n_forecast_months))
    leadtimes = np.arange(1, n_forecast_months+1)

    # Load IceNet forecasts (this is done now to obtain `icenet_seeds`, used
    #   for results_df
    if icenet_ID in model_compute_list:

        heldout_forecast_fpath = os.path.join(
            config.forecast_data_folder, 'icenet',
            dataloader_ID, architecture_ID, 'icenet_forecasts.nc'
        )

        chunks = {'seed': 1}
        icenet_forecast_da = xr.open_dataarray(heldout_forecast_fpath, chunks=chunks)
        icenet_seeds = icenet_forecast_da.seed.values

    ### Monthly masks (active grid cell regions to compute metrics over)
    ####################################################################

    mask_fpath_format = os.path.join(config.mask_data_folder, config.active_grid_cell_file_format)

    month_mask_da = xr.DataArray(np.array(
        [np.load(mask_fpath_format.format('{:02d}'.format(month))) for
         month in np.arange(1, 12+1)],
    ))

    ### Initialise results dataframe
    ####################################################################

    heldout_start = pd.Timestamp('2012-01-01')
    heldout_end = pd.Timestamp('2020-12-01')

    all_target_dates = pd.date_range(
        start=heldout_start,
        end=heldout_end,
        freq='MS'
    )

    if not os.path.exists(config.forecast_results_folder):
        os.makedirs(config.forecast_results_folder)

    now = pd.Timestamp.now()
    new_results_df_fname = now.strftime('%Y_%m_%d_%H%M%S_forecast_results.csv')
    new_results_df_fpath = os.path.join(config.forecast_results_folder, new_results_df_fname)

    print('New results will be saved to {}\n\n'.format(new_results_df_fpath))

    if pre_load_results_df:
        results_df_fnames = sorted([f for f in os.listdir(config.forecast_results_folder) if re.compile('.*.csv').match(f)])
        if len(results_df_fnames) >= 1:
            old_results_df_fname = results_df_fnames[-1]
            old_results_df_fpath = os.path.join(config.forecast_results_folder, old_results_df_fname)
            print('\n\nLoading previous results dataset from {}'.format(old_results_df_fpath))

        # Load previous results, do not interpret 'NA' as NaN
        results_df = pd.read_csv(old_results_df_fpath, keep_default_na=False, comment='#')

        # Drop spurious index column if present
        results_df = results_df.drop('Unnamed: 0', axis=1, errors='ignore')
        results_df['Forecast date'] = [pd.Timestamp(date) for date in results_df['Forecast date']]

        existing_models = results_df.Model.unique()
        results_df = results_df.set_index(['Model', 'Ensemble member', 'Leadtime', 'Forecast date'])
        existing_metrics = results_df.columns

        new_models = [model for model in model_compute_list if model not in existing_models]
        new_metrics = [metric for metric in metric_compute_list if metric not in existing_metrics]

        compute_dict = {}
        for new_model in new_models:
            # Compute all metrics for new models
            compute_dict[new_model] = metric_compute_list

        # Add new metrics to the dataframe
        if len(new_metrics) > 0:
            for existing_model in existing_models:
                # Compute new metrics for existing models
                compute_dict[existing_model] = new_metrics

            results_df = pd.concat(
                [results_df, pd.DataFrame(columns=new_metrics)], axis=1)

        # Add new models to the dataframe
        if len(new_models) > 0:
            multi_index = utils.create_results_dataset_index(
                new_models, leadtimes, all_target_dates, icenet_ID, icenet_seeds)
            results_df = results_df.append(pd.DataFrame(index=multi_index)).sort_index()

    else:
        # Instantiate new results dataframe
        multi_index = utils.create_results_dataset_index(
            model_compute_list, leadtimes, all_target_dates, icenet_ID, icenet_seeds)
        results_df = pd.DataFrame(index=multi_index, columns=metric_compute_list, dtype=np.float32)
        results_df = results_df.sort_index()

        compute_dict = {
            model: metric_compute_list for model in model_compute_list
        }

    print('Computations to perform:')
    pprint.pprint(compute_dict)
    print('\n\n')

    ### Load forecasts
    ####################################################################

    heldout_forecasts_dict = {}

    for model in compute_dict.keys():

        # IceNet ID format
        icenet_ID_match = re.compile('^IceNet__(.*)__(.*)$').match(model)

        # IceNet
        if icenet_ID_match:
            forecast_da = icenet_forecast_da

        # Non-IceNet model
        else:
            model_str = model.replace(' ', '_').lower()
            fname = '{}_forecasts.nc'.format(model_str)

            heldout_forecast_fpath = os.path.join(
                config.forecast_data_folder, model_str, fname
            )

            chunks = {}
            forecast_da = xr.open_dataarray(heldout_forecast_fpath, chunks=chunks)
            forecast_da = forecast_da.expand_dims({'seed': ['NA']})

        heldout_forecasts_dict[model] = forecast_da.sel(time=all_target_dates)

        if compute_in_memory:
            heldout_forecasts_dict[model].load()

    ### Ground truth SIC
    ####################################################################

    if len(compute_dict) >= 1:

        true_sic_fpath = os.path.join(config.obs_data_folder, 'siconca_EASE.nc')
        true_sic_da = xr.open_dataarray(true_sic_fpath, chunks={})
        if compute_in_memory:
            true_sic_da = true_sic_da.load()
        true_sic_da = true_sic_da.sel(time=all_target_dates)

        if 'Binary accuracy' in metric_compute_list:
            binary_true_da = true_sic_da > 0.15

    ### Monthwise masks
    ####################################################################

    if len(compute_dict) >= 1:
        months = [pd.Timestamp(date).month - 1 for date in all_target_dates]
        mask_da = xr.DataArray(
            [month_mask_da[month] for month in months],
            dims=('time', 'yc', 'xc'),
            coords={
                'time': true_sic_da.time.values,
                'yc': true_sic_da.yc.values,
                'xc': true_sic_da.xc.values,
            }
        )

    ### Compute
    ####################################################################

    print('Analysing forecasts: \n\n')

    if compute_with_dask:
        client = distributed.Client(n_workers=n_workers, threads_per_worker=threads_per_worker)

    for model, model_metric_compute_list in compute_dict.items():

        print(model)
        print('Computing metrics:')
        print(model_metric_compute_list)
        tic = time()

        # Sea Ice Probability
        icenet_ID_match = re.compile('^IceNet__(.*)__(.*)$').match(model)
        if icenet_ID_match:
            sip_da = heldout_forecasts_dict[model].\
                sel(ice_class=['marginal_ice', 'full_ice']).sum('ice_class')

            binary_forecast_da = sip_da > 0.5

        # Sea Ice Concentration
        else:
            binary_forecast_da = heldout_forecasts_dict[model] > 0.15

        compute_ds = xr.Dataset()
        for metric in model_metric_compute_list:

            if metric == 'Binary accuracy':
                binary_correct_da = (binary_forecast_da == binary_true_da).astype(np.float32)
                binary_correct_weighted_da = binary_correct_da.weighted(mask_da)

                # Mean percentage of correct classifications over the active
                #   grid cell area
                ds_binacc = (binary_correct_weighted_da.mean(dim=['yc', 'xc']) * 100)
                compute_ds[metric] = ds_binacc

            elif metric == 'SIE error':
                binary_forecast_weighted_da = binary_forecast_da.astype(int).weighted(mask_da)
                binary_true_weighted_da = binary_true_da.astype(int).weighted(mask_da)

                ds_sie_error = (
                    binary_forecast_weighted_da.sum(['xc', 'yc']) -
                    binary_true_weighted_da.sum(['xc', 'yc'])
                ) * 25**2

                compute_ds[metric] = ds_sie_error

        if compute_with_dask:
            compute_ds = compute_ds.persist()
            distributed.progress(compute_ds)
            compute_ds = compute_ds.compute()

        print('Writing to results dataset...')
        for compute_da in iter(compute_ds.data_vars.values()):
            metric = compute_da.name

            compute_df_index = results_df.loc[
                pd.IndexSlice[model, :, leadtimes, all_target_dates], metric].\
                droplevel(0).index

            # Ensure indexes are aligned for assigning to results_df
            compute_df = compute_da.to_dataframe().reset_index().\
                set_index(['seed', 'leadtime', 'time']).\
                reindex(index=compute_df_index)

            results_df.loc[pd.IndexSlice[model, :, leadtimes, all_target_dates], metric] = \
                compute_df.values

        dur = time() - tic
        print("Done in {}m:{:.0f}s.\n\n ".format(np.floor(dur / 60), dur % 60))

    if len(compute_dict) > 0:
        print('\nCheckpointing results dataset... ', end='', flush=True)
        results_df.to_csv(new_results_df_fpath)
        print('Done.')

        print('\n\nNEW RESULTS: ')
        print(results_df.head(10))
        print('\n...\n')
        print(results_df.tail(10))
