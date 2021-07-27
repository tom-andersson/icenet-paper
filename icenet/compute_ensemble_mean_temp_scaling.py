import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))
from losses import weighted_categorical_crossentropy_single_leadtime
import config
import scipy
import numpy as np
import xarray as xr
import pandas as pd
from utils import IceNetDataLoader
import tensorflow as tf

'''
Computes ensemble-mean temperature scaling parameters for each lead time of
the ensemble-mean IceNet model, using the validation period from 2012-2017.

Loads the raw IceNet ensemble-mean forecasts using the 'ensemble' coordinate of
the 'seed' dimension in `data/forecasts/icenet/<dataloader_ID>/<architecture_ID>/icenet_forecasts.nc`
and uses scipy.optimize.minimize_scalar to obtain the T-value that minimises
the categorical crossentropy at each lead time.

The array of `N` T-values are saved to:
`icenet/trained_networks/<dataloader_ID>/<architecture_ID>/T_ens_opts.npy`

The temperature scaled forecasts are stored in a new 'ensemble_tempscaled'
coordinate of the original `icenet_forecasts.nc` dataset.

The ensemble-mean temperature-scaled sea ice probability (SIP) forecasts
are saved to:
`data/forecasts/icenet/<dataloader_ID>/<architecture_ID>/icenet_sip_forecasts_tempscaled.nc`
'''

#### User input
####################################################################

dataloader_ID = '2021_06_15_1854_icenet_nature_communications'
architecture_ID = 'unet_tempscale'

val_start = '2012-01-01'
val_end = '2017-12-01'

#### Load ensemble mean forecasts and dataloader
####################################################################

dataloader_config_fpath = os.path.join('dataloader_configs', dataloader_ID+'.json')
dataloader_ID_folder = os.path.join(config.networks_folder, dataloader_ID)
icenet_folder = os.path.join(dataloader_ID_folder, architecture_ID)

# Validation observations
true_sic_fpath = os.path.join(config.data_folder, 'obs', 'siconca_EASE.nc')
true_sic_da = xr.open_dataarray(true_sic_fpath).sel(time=slice(val_start, val_end))
true_sic_da = true_sic_da.load()

true_sic_class_da = true_sic_da.expand_dims({'ice_class': 3}, axis=-1)
true_sic_class_da.data = np.zeros(true_sic_class_da.shape)
true_sic_class_da.loc[..., 0] = true_sic_da <= 0.15
true_sic_class_da.loc[..., 1] = (true_sic_da > 0.15) & (true_sic_da < 0.80)
true_sic_class_da.loc[..., 2] = true_sic_da >= 0.80
true_sic_class_da = true_sic_class_da.load()

# IceNet ensemble-mean probabilistic forecasts
heldout_forecast_fpath = os.path.join(
    config.forecast_data_folder, 'icenet',
    dataloader_ID, architecture_ID, 'icenet_forecasts.nc'
)
icenet_forecast_da = xr.open_dataarray(heldout_forecast_fpath).sel(seed='ensemble')
icenet_forecast_da = icenet_forecast_da.sel(time=slice(val_start, val_end))

dataloader = IceNetDataLoader(dataloader_config_fpath)

# Sample weights over the validation dataset
sample_weights = []
for date in pd.date_range(val_start, val_end, freq='MS'):
    _, _, sample_weight = dataloader.data_generation(date)
    # Slice out first lead time
    sample_weight = sample_weight[0, :, :, :, 0]
    sample_weights.append(sample_weight)
sample_weights = np.stack(sample_weights, axis=0)

#### Ensemble-mean temperature scaling
####################################################################


def func(T, ensemble_mean_logits_leadtime, leadtime):

    scaled_ensemble_mean_probs = tf.keras.activations.softmax(
        tf.Variable(ensemble_mean_logits_leadtime / T), axis=-1)

    all_cce = weighted_categorical_crossentropy_single_leadtime(
        true_sic_class_da.data, scaled_ensemble_mean_probs, sample_weights)
    cce = np.mean(all_cce)
    return cce


# Convert from probs to logits using logit = log(p) + c
ensemble_mean_logits = np.log(np.clip(icenet_forecast_da, 1e-9, 1-1e-9))

# Optimal T for each leadtime for the ensemble mean prob
T_ens_opts = []

print('Finding optimal T for each leadtime using Brent optimisation... ', end='', flush=True)

for leadtime in icenet_forecast_da.leadtime.values:
    print('{}, '.format(leadtime), end='', flush=True)

    ensemble_mean_logits_leadtime = ensemble_mean_logits.sel(leadtime=leadtime)

    T_ens_opt = scipy.optimize.minimize_scalar(
        func, args=(ensemble_mean_logits_leadtime, leadtime),
        bracket=(0.1, 2.0), method='brent', tol=1e-6)
    T_ens_opts.append(T_ens_opt.x)

print('Done.')

for T in T_ens_opts:
    print('Optimal T-value at each leadtime: {:.3f}'.format(T))

T_ens_opts = np.array(T_ens_opts)

np.save(os.path.join(os.path.join(icenet_folder, 'T_ens_opts.npy')), T_ens_opts)

#### Include 'ensemble_tempscaled' in forecast dataset
####################################################################

full_icenet_forecast_da = xr.open_dataarray(heldout_forecast_fpath)
ensemble_mean_forecast_da = xr.open_dataarray(heldout_forecast_fpath).sel(seed='ensemble')

ensemble_mean_logits = np.log(np.clip(ensemble_mean_forecast_da, 1e-9, 1-1e-9))
ensemble_mean_logits /= T_ens_opts.reshape(1, 1, 1, 6, 1)

scaled_ensemble_mean_probs = tf.keras.activations.softmax(
    tf.Variable(ensemble_mean_logits), axis=-1)

icenet_forecast_tempscaled_da = ensemble_mean_forecast_da.copy()
icenet_forecast_tempscaled_da.data = scaled_ensemble_mean_probs

# Re-mask outside active grid cell region to zero
mask_fpath_format = os.path.join(config.mask_data_folder, config.active_grid_cell_file_format)
mask_arr = np.tile(np.stack(
    [np.load(mask_fpath_format.format('{:02d}'.format(pd.Timestamp(date).month))) for
     date in icenet_forecast_tempscaled_da.time.values]
)[..., np.newaxis, np.newaxis], [1, 1, 1, 6, 3])
icenet_forecast_tempscaled_da = icenet_forecast_tempscaled_da.where(mask_arr, 0.)

icenet_forecast_tempscaled_da.seed.values = 'ensemble_tempscaled'

if 'ensemble_tempscaled' in full_icenet_forecast_da.seed.values:
    print('Dropping existing ensemble_tempscaled data.')
    full_icenet_forecast_da = \
        full_icenet_forecast_da.where(
            full_icenet_forecast_da.seed != 'ensemble_tempscaled', drop=True)

print('Concatting new ensemble mean forecasts to IceNet forecast dataset... ')
full_icenet_forecast_da = \
    xr.concat([full_icenet_forecast_da, icenet_forecast_tempscaled_da], dim='seed')

print('Overwriting... ')
os.remove(heldout_forecast_fpath)
full_icenet_forecast_da.to_netcdf(heldout_forecast_fpath)

print('Computing new ensemble mean SIP and saving... ', end='', flush=True)
fpath = os.path.join(
    config.forecast_data_folder, 'icenet',
    dataloader_ID, architecture_ID, 'icenet_sip_forecasts_tempscaled.nc'
)
icenet_ensemble_mean_sip_da = icenet_forecast_tempscaled_da.\
    sel(ice_class=['marginal_ice', 'full_ice']).sum('ice_class')
if os.path.exists(fpath):
    os.remove(fpath)
icenet_ensemble_mean_sip_da.to_netcdf(fpath)
print('Done.')
