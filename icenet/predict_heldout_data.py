import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))
import config
import re
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from models import linear_trend_forecast
from utils import IceNetDataLoader
from tensorflow.keras.models import load_model

'''
Produces SIP forecasts from IceNet and SIC forecasts from the linear trend
model. Stores the forecasts in xarray.DataArrays and saves them as NetCDF files
in data/forecasts/<model>/ folders.

The dimensions of the IceNet forecasts are `(time, yc, xc, leadtime, seed,
ice_class)`, where the 'seed' dimension specifies the ensemble member
(identified by the integer random seed value it was trained with) or the
ensemble mean model ('ensemble').

For IceNet, the ensemble-mean SIP forecast is also saved as a separate, smaller
file.

Logic for producing forecasts from other Python-based models could be added to this
script with relative ease.
'''

####################################################################

# List of models to produce forecasts for
models = ['IceNet', 'Linear trend']

# Specifications for the IceNet model to produce forecasts for
dataloader_ID = '2021_06_15_1854_icenet_nature_communications'
architecture_ID = 'unet_tempscale'
tempscaling_used = True  # Whether to load networks with temperature scaling

#### Load network and dataloader
####################################################################

dataloader_config_fpath = os.path.join(config.dataloader_config_folder, dataloader_ID+'.json')
dataloader_ID_folder = os.path.join(config.networks_folder, dataloader_ID)
icenet_folder = os.path.join(config.networks_folder, dataloader_ID, architecture_ID)
network_h5_files_folder = os.path.join(icenet_folder, 'networks')

# Data loader
print("\nSetting up the data loader with config file: {}\n\n".format(dataloader_ID))
dataloader = IceNetDataLoader(dataloader_config_fpath)
print('\n\nDone.\n')

if 'IceNet' in models:

    if tempscaling_used:
        network_regex = re.compile('^network_tempscaled_([0-9]*).h5$')
    else:
        network_regex = re.compile('^network_([0-9]*).h5$')

    network_fpaths = [os.path.join(network_h5_files_folder, f) for f in
                      sorted(os.listdir(network_h5_files_folder)) if network_regex.match(f)]

    ensemble_seeds = [network_regex.match(f)[1] for f in
                      sorted(os.listdir(network_h5_files_folder)) if network_regex.match(f)]
    ensemble_seeds_and_mean = ensemble_seeds.copy()
    ensemble_seeds_and_mean.append('ensemble')

    networks = []
    for network_fpath in network_fpaths:
        print('Loading model from {}... '.format(network_fpath), end='', flush=True)
        networks.append(load_model(network_fpath, compile=False))
        print('Done.')

    print("Temperature scaling factors:")
    for network, seed in zip(networks, ensemble_seeds):
        print('Seed {}: '.format(seed), end='', flush=True)
        for layer in network.layers:
            if re.compile('temperature_scale*').match(layer.name):
                print('{:.3f}'.format(layer.temp.numpy()))

#### Set up forecast folder structure
####################################################################

forecast_folders_dict = {}

for model in models:

    if model == 'IceNet':
        forecast_folders_dict[model] = os.path.join(
            config.forecast_data_folder, 'icenet', dataloader_ID, architecture_ID)

    else:
        model_str = model.replace(' ', '_').lower()
        forecast_folders_dict[model] = os.path.join(
            config.forecast_data_folder, model_str)

    if not os.path.exists(forecast_folders_dict[model]):
        os.makedirs(forecast_folders_dict[model])

#### Load ground truth SIC for statistical models
####################################################################

print('Loading ground truth SIC... ', end='', flush=True)
true_sic_fpath = os.path.join(config.obs_data_folder, 'siconca_EASE.nc')
true_sic_da = xr.open_dataarray(true_sic_fpath)
print('Done.')

#### Set up forecast DataArray dictionary
####################################################################

n_forecast_months = dataloader.config['n_forecast_months']

heldout_start = pd.Timestamp('2012-01-01')
heldout_end = pd.Timestamp('2020-12-01')

all_target_dates = pd.date_range(
    start=heldout_start,
    end=heldout_end,
    freq='MS'
)

all_start_dates = pd.date_range(
    start=heldout_start - pd.DateOffset(months=n_forecast_months-1),
    end=heldout_end,
    freq='MS'
)

leadtimes = np.arange(1, n_forecast_months+1)

model_forecast_dict = {}
for model in models:

    shape = (len(all_target_dates),
             *dataloader.config['raw_data_shape'],
             n_forecast_months)

    coords = {
        'time': all_target_dates,  # To be sliced to target dates
        'yc': true_sic_da.coords['yc'],
        'xc': true_sic_da.coords['xc'],
        'lon': true_sic_da.isel(time=0).coords['lon'],
        'lat': true_sic_da.isel(time=0).coords['lat'],
        'leadtime': leadtimes,
    }

    # Probabilistic SIC class forecasts
    if model == 'IceNet':
        dims = ('seed', 'time', 'yc', 'xc', 'leadtime', 'ice_class')
        coords['seed'] = ensemble_seeds_and_mean
        coords['ice_class'] = ['no_ice', 'marginal_ice', 'full_ice']
        shape = (len(ensemble_seeds_and_mean), *shape, 3)

    # Deterministic SIC forecasts
    else:
        dims = ('time', 'yc', 'xc', 'leadtime')

    model_forecast_dict[model] = xr.DataArray(
        data=np.zeros(shape, dtype=np.float32),
        coords=coords,
        dims=dims
    )

#### Build up forecasts
####################################################################

print('Building up forecast DataArrays...\n\n')

for model in models:

    print(model + ':\n')

    start_date = all_start_dates[0]
    for start_date in tqdm(all_start_dates):

        # Target forecast dates for the forecast beginning at this `start_date`
        target_dates = pd.date_range(
            start=start_date,
            end=start_date + pd.DateOffset(months=n_forecast_months-1),
            freq='MS'
        )

        if model == 'IceNet':

            X, y, sample_weights = dataloader.data_generation([start_date])
            mask = sample_weights > 0
            pred = np.array([network.predict(X)[0] for network in networks])
            pred *= mask  # mask outside active grid cell region to zero
            # concat ensemble mean to the set of network predictions
            ensemble_mean_pred = pred.mean(axis=0, keepdims=True)
            pred = np.concatenate([pred, ensemble_mean_pred], axis=0)

        if model == 'Linear trend':

            # Same forecast for each lead time: interpret start_date as target_date
            #   for efficiency
            pred, _ = linear_trend_forecast(start_date, n_linear_years=35)

        for i, (target_date, leadtime) in enumerate(zip(target_dates, leadtimes)):

            if model == 'Linear trend':

                # Same forecast at each lead time
                if start_date in all_target_dates:
                    model_forecast_dict[model].\
                        loc[start_date, :, :, leadtime] = pred

            if model == 'IceNet':

                if target_date in all_target_dates:
                        model_forecast_dict[model].\
                            loc[:, target_date, :, :, leadtime] = pred[..., i]

    print('Saving forecast NetCDF for {}... '.format(model), end='', flush=True)

    model_str = model.replace(' ', '_').lower()
    forecast_fpath = os.path.join(forecast_folders_dict[model], f'{model_str}_forecasts.nc'.format(model_str))
    if os.path.exists(forecast_fpath):
        os.remove(forecast_fpath)
    model_forecast_dict[model].to_netcdf(forecast_fpath)

    if model == 'IceNet':
        print('Computing ensemble mean SIP... ', end='', flush=True)

        icenet_ensemble_mean_da = model_forecast_dict[model].sel(seed='ensemble')
        icenet_ensemble_mean_sip_da = icenet_ensemble_mean_da.\
            sel(ice_class=['marginal_ice', 'full_ice']).sum('ice_class')

        fpath = os.path.join(forecast_folders_dict[model], 'icenet_sip_forecasts.nc')
        icenet_ensemble_mean_sip_da.to_netcdf(fpath)

    print('Done.')

    del(model_forecast_dict[model])

print('Done.')
