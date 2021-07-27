import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))
from utils import arr_to_ice_edge_arr
import config
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

'''
Assesses the calibration of IceNet and SEAS5, determines IceNet's 'ice edge region',
and assesses IceNet's ice edge bounding ability. Results are saved to
`results/uncertainty_results/`

Ensemble-mean SIP forecasts are loaded for SEAS5 and IceNet, and the maps are
converted to vectors of grid cell-wise SIP predictions, stored in a
`pd.DataFrame` (`uncertainty_df`), alongside boolean columns for whether the
model made a binary error and whether the ice edge was located at that grid
cell.

`uncertainty_df` is used for producing probability calibration plots,
computing the p'_90% value for the ice edge region over the validation years,
and determining the fraction of observed ice edge bounded by IceNet's ice
edge region over the test years.
'''

### User input
####################################################################

dataloader_ID = '2021_06_15_1854_icenet_nature_communications'
architecture_ID = 'unet_tempscale'

use_tempscaled_ensemble = True

### Paths
####################################################################

dataloader_ID_folder = os.path.join(config.networks_folder, dataloader_ID)
icenet_folder = os.path.join(dataloader_ID_folder, architecture_ID)

if not os.path.exists(config.uncertainty_results_folder):
    os.makedirs(config.uncertainty_results_folder)

ice_edge_region_df_fpath = os.path.join(config.uncertainty_results_folder, 'ice_edge_region_results.csv')
uncertainty_df_fpath = os.path.join(config.uncertainty_results_folder, 'uncertainty_results.csv')
sip_bounding_df_fpath = os.path.join(config.uncertainty_results_folder, 'sip_bounding_results.csv')
sip_prime_90_fpath = os.path.join(icenet_folder, 'sip_prime_90.npy')

### Set up; Load maps
####################################################################

# Validation dates for computing p'_90%
val_start = '2012-01-01'
val_end = '2017-12-01'

### IceNet forecasts
if use_tempscaled_ensemble:
    fname = 'icenet_sip_forecasts_tempscaled.nc'
else:
    fname = 'icenet_sip_forecasts.nc'
fpath = os.path.join(
    config.forecast_data_folder, 'icenet', dataloader_ID, architecture_ID, fname
)
icenet_ensemble_mean_sip_da = xr.open_dataarray(fpath)

### SEAS5 forecasts
fpath = os.path.join(
    config.forecast_data_folder, 'seas5', 'seas5_sip_forecasts.nc'
)
seas5_ensemble_mean_sip_da = xr.open_dataarray(fpath)

### Ground truth SIC
true_sic_fpath = os.path.join(config.data_folder, 'obs', 'siconca_EASE.nc')
true_sic_da = xr.open_dataarray(true_sic_fpath)

### Masks
mask_fpath_format = os.path.join(config.mask_data_folder, config.active_grid_cell_file_format)
land_mask = np.load(os.path.join(config.mask_data_folder, 'land_mask.npy'))
region_mask = np.load(os.path.join(config.mask_data_folder, 'region_mask.npy'))

### Functions
####################################################################


def get_preds(da, date, leadtime=None):
    ''' Get a vector of grid cell values within the active grid cell
    region for a given month '''

    if leadtime is None:
        arr = da.sel(time=date).data
    else:
        arr = da.sel(time=date, leadtime=leadtime).data
    mask = np.load(mask_fpath_format.format('{:02d}'.format(date.month)))
    return arr[mask]


def get_ice_edge_vector(date):
    ''' Get a boolean vector of ice edge locations within the active grid cell
    region for a given month. '''

    arr = true_sic_da.sel(time=date).data
    ice_edge_arr = arr_to_ice_edge_arr(
        arr, 0.15, land_mask, region_mask)
    mask = np.load(mask_fpath_format.format('{:02d}'.format(date.month)))
    return ice_edge_arr[mask]


### Build up uncertainty analysis datasets from the forecast/true NetCDFs
####################################################################

print("Building up dataset of vector predictions for each forecast... ", end='', flush=True)

dates = icenet_ensemble_mean_sip_da.time.values
dates = [pd.Timestamp(date) for date in dates]
leadtimes = icenet_ensemble_mean_sip_da.leadtime.values

# List of vector predictions & ground truth in pandas.DataFrame format
monthly_preds_dfs = []
for date in dates:
    true = get_preds(true_sic_da, date)
    true = true > 0.15  # Convert to binary ice class
    ice_edge = get_ice_edge_vector(date)
    N = true.size
    for leadtime in leadtimes:
        icenet_sip = get_preds(icenet_ensemble_mean_sip_da, date, leadtime)

        # Correct any numerical error causing SIP slightly > 1.
        icenet_sip[icenet_sip > 1.] = 1.

        seas5_sip = get_preds(seas5_ensemble_mean_sip_da, date, leadtime)

        idx = pd.MultiIndex.from_tuples(
            [(leadtime, date)] * N, names=['Leadtime', 'Forecast date'])

        df = pd.DataFrame(
            index=idx,
            columns=['Ground truth', 'Ice edge', 'IceNet SIP', 'SEAS5 SIP'],
            data={
                'Ground truth': true,
                'Ice edge': ice_edge,
                'IceNet SIP': icenet_sip,
                'SEAS5 SIP': seas5_sip,
            },
        )

        monthly_preds_dfs.append(df)

# DataFrame of all grid cell predictions for all months and lead times
preds_df = pd.concat(monthly_preds_dfs)
preds_df = preds_df.rename(columns={'IceNet SIP': 'IceNet', 'SEAS5 SIP': 'SEAS5'})

# Wide form -> Long form
#   Note: It is a waste of memory to store the ground truth and ice edge binary
#   values for each lead time, because they don't vary with lead time. However,
#   the total memory size is small enough here (~ 3 GB) that it's not
#   an issue.
uncertainty_df = pd.melt(
    preds_df.reset_index(), value_name='SIP', var_name='Model',
    value_vars=('IceNet', 'SEAS5'),
    id_vars=('Leadtime', 'Forecast date', 'Ground truth', 'Ice edge'))
uncertainty_df = uncertainty_df.astype({'Model': 'string'})

# Add column with binary variable e = 1 if a binary error was made
uncertainty_df['Error?'] = True

# True negative
uncertainty_df.loc[(uncertainty_df['Ground truth'] == 0) &
                   (uncertainty_df['SIP'] <= .5), 'Error?'] = False
# True positive
uncertainty_df.loc[(uncertainty_df['Ground truth'] == 1) &
                   (uncertainty_df['SIP'] >= .5), 'Error?'] = False

uncertainty_df = uncertainty_df.set_index('Model')
print('Done.\n')

### Sample weights for SIP histograms in Supplementary Information
### For dividing counts by total number of samples in each bin to get error prob
####################################################################

print("Computing weight values for validation SIP error rate histogram... ", end='', flush=True)

bins = {}
# 99 bins betw 0 and 1: ensures no bin edge aligns with SEAS5 discrete SIP values
bins['SEAS5'] = np.linspace(0., 1., 100)  # Note: bin edges
# 100 bins betw 0 and 1: ensures symmetry for IceNet bins
bins['IceNet'] = np.linspace(0., 1., 101)

uncertainty_df['Histogram weight'] = 0.

# Validation data only
uncertainty_val_df = uncertainty_df.reset_index().set_index('Forecast date').\
    sort_index().loc[slice(val_start, val_end)]

uncertainty_val_df = uncertainty_val_df.reset_index().\
    set_index('Model').sort_index()

for model in ['IceNet', 'SEAS5']:
    # Bin counts for SIP histogram, counting only over the validation set
    #   (Supp Fig 2 is plotted for validation data only)
    bin_counts = plt.hist(
        uncertainty_val_df.loc[model].SIP,
        bins=bins[model]
    )[0]
    bin_counts = bin_counts.astype(np.float64)

    # Ignore divide by zero warning due to discrete SEAS5 SIP giving zero-counts
    warnings.simplefilter("ignore", RuntimeWarning)
    weights = 1 / bin_counts  # Vector of weights for normalising each bin

    # Determine which bin index each SIP value lies in, and assign bin weight
    #   Note: Only for validation dates, since the histogram is plotted over
    #   validation dates only.
    uncertainty_df.loc[model, 'Histogram weight'] = \
        weights[pd.cut(uncertainty_df.loc[model, 'SIP'],
                       bins=bins[model],
                       labels=np.arange(len(bins[model])-1),
                       include_lowest=True)]

print('Done.\n')

### Compute SIP for 90% ice edge bounding in validation set (p'_90% in paper)
#############################################################################

print("Computing p'_90% for 90% ice edge bounding:")

sip_primes = np.linspace(0, 0.5, 1001)

# Fraction of various quantities bounded by (p', 1-p')
sip_bounding_df = pd.DataFrame(
    index=pd.MultiIndex.from_product((['IceNet', 'SEAS5'], sip_primes),
                                     names=['Model', "p'"]),
    columns=['frac_all', 'frac_error', 'frac_ice_edge'],
)

for model in ['SEAS5', 'IceNet']:

    sip_all = uncertainty_val_df.loc[model].SIP.values.copy()
    # Make symmetric about p = 0.5. This means finding the fraction of the
    #   shifted SIP greater than p' is equivalent to finding the fraction of
    #   SIP in (p', 1-p')
    sip_all[sip_all > 0.5] = 1 - sip_all[sip_all > 0.5]

    sip_error = uncertainty_val_df.reset_index().\
        set_index(['Model', 'Error?']).sort_index().\
        loc[model, True].SIP.values.copy()
    sip_error[sip_error > 0.5] = 1 - sip_error[sip_error > 0.5]

    sip_ice_edge = uncertainty_val_df.reset_index().\
        set_index(['Model', 'Ice edge']).sort_index().\
        loc[model, True].SIP.values.copy()
    sip_ice_edge[sip_ice_edge > 0.5] = 1 - sip_ice_edge[sip_ice_edge > 0.5]

    norm_all = len(sip_all)
    norm_error = len(sip_error)
    norm_ice_edge = len(sip_ice_edge)

    frac_all = 1 - np.cumsum(np.histogram(sip_all, bins=sip_primes)[0] / norm_all)
    # Account for first bin edge, which integrates to 1
    frac_all = np.insert(frac_all, 0, 1)

    frac_error = 1 - np.cumsum(np.histogram(sip_error, bins=sip_primes)[0] / norm_error)
    # Account for first bin edge, which integrates to 1
    frac_error = np.insert(frac_error, 0, 1)

    frac_ice_edge = 1 - np.cumsum(np.histogram(sip_ice_edge, bins=sip_primes)[0] / norm_ice_edge)
    # Account for first bin edge, which integrates to 1
    frac_ice_edge = np.insert(frac_ice_edge, 0, 1)

    # Convert to %
    frac_all = 100 * np.array(frac_all)
    frac_error = 100 * np.array(frac_error)
    frac_ice_edge = 100 * np.array(frac_ice_edge)

    # Store in DataFrame
    sip_bounding_df.loc[model, 'frac_all'] = frac_all
    sip_bounding_df.loc[model, 'frac_error'] = frac_error
    sip_bounding_df.loc[model, 'frac_ice_edge'] = frac_ice_edge

    if model == 'IceNet':
        sip_prime_95 = sip_primes[np.argwhere(np.diff(np.sign(np.array(
            sip_bounding_df.loc[model, 'frac_ice_edge']) - 95)))].ravel()[0]
        sip_prime_90 = sip_primes[np.argwhere(np.diff(np.sign(np.array(
            sip_bounding_df.loc[model, 'frac_ice_edge']) - 90)))].ravel()[0]

        frac_all_95 = frac_all[np.argwhere(np.diff(np.sign(
            np.array(sip_primes) - sip_prime_95)))].ravel()[0]
        frac_all_90 = frac_all[np.argwhere(np.diff(np.sign(
            np.array(sip_primes) - sip_prime_90)))].ravel()[0]

        print(model)
        print("p' for 95% ice edge bounding: {:.4f}. Fraction of preds in "
              "ice edge regions: {:.1f}".format(sip_prime_95, frac_all_95))
        print("p' for 90% ice edge bounding: {:.4f}. Fraction of preds in "
              "ice edge regions: {:.1f}".format(sip_prime_90, frac_all_90))
        print('\n')

print('Done.\n')

### Analyse ice edge bounding properties of p'_90%
#############################################################################

print("Analysing ice edge bounding properties of p'_90%... ", end='', flush=True)

# Area covered by (p', 1-p') for each month and lead time
area_df = uncertainty_df.loc['IceNet'].groupby(['Leadtime', 'Forecast date']).SIP.\
    apply(lambda p: np.sum((p > sip_prime_90) & (p < 1-sip_prime_90)) * 25**2). \
    to_frame().reset_index()

# Fraction of ice edge grid cells within (p', 1-p') for each month and lead time
ice_edge_region_df = uncertainty_df.reset_index().set_index(['Model', 'Ice edge']).\
    sort_index().loc['IceNet', True].groupby(['Leadtime', 'Forecast date']).SIP.\
    apply(lambda p: 100 * np.sum((p > sip_prime_90) & (p < 1-sip_prime_90)) / len(p)). \
    to_frame().reset_index()
ice_edge_region_df = ice_edge_region_df.rename({'SIP': 'Coverage'}, axis=1)
ice_edge_region_df['Area'] = area_df['SIP']  # Grab pandas.Series

month_names = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])
forecast_month_names = month_names[ice_edge_region_df['Forecast date'].dt.month.values - 1]
ice_edge_region_df['Calendar month'] = forecast_month_names

print('Done.\n')

### Save
#############################################################################

print("Saving datasets... ", end='', flush=True)

uncertainty_df.to_csv(uncertainty_df_fpath)
sip_bounding_df.to_csv(sip_bounding_df_fpath)
ice_edge_region_df.to_csv(ice_edge_region_df_fpath)
np.save(sip_prime_90_fpath, sip_prime_90)

print('Done.\n')
