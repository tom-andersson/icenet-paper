import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))  # if using jupyter kernel
import config
import dask
import xarray as xr
import pandas as pd
import time
import numpy as np

'''
Script to compute SEAS5 bias correction fields; bias correct the ensemble-mean
sea ice concentration (SIC) forecasts and the sea ice probability (SIP) forecasts;
and save in the SEAS5 forecast folder.

The bias-corrected forecasts are saved as NetCDFs in `data/forecasts/seas5/`
with filenames `seas5_forecasts.nc` and `seas5_sip_forecasts.nc` with
dimensions `(time, yc, xc, leadtime)`.
'''

biascorrection_dates = pd.date_range(
    start='2002-01-01',
    end='2011-12-01',
    freq='MS'
)

#### Load data
###############################################################################

seas5_folder = os.path.join(config.forecast_data_folder, 'seas5')
seas5_EASE_folder = os.path.join(seas5_folder, 'EASE')

leadtimes = np.arange(1, 7)

# Raw SEAS5 forecasts
fpaths = sorted([os.path.join(seas5_EASE_folder, f) for f in os.listdir(seas5_EASE_folder)])
seas5_forecast_da_list = []
for leadtime, fpath in zip(leadtimes, fpaths):
    seas5_leadtime_da = xr.open_dataset(fpath, chunks={'time': 12})['siconc']
    seas5_leadtime_da = seas5_leadtime_da.assign_coords({'leadtime': leadtime})

    # Convert spatial coords to km
    seas5_leadtime_da = seas5_leadtime_da.assign_coords(
        dict(xc=seas5_leadtime_da.xc/1e3, yc=seas5_leadtime_da.yc/1e3))

    seas5_forecast_da_list.append(seas5_leadtime_da)

# Load the monthly active grid cell masks
active_grid_cell_masks = {}
for month in np.arange(1, 13):
    month_str = '{:02d}'.format(month)
    active_grid_cell_masks[month_str] = np.load(
        os.path.join(config.mask_data_folder,
                     config.active_grid_cell_file_format.format(month_str)))

### Ground truth SIC
true_sic_fpath = os.path.join(config.obs_data_folder, 'siconca_EASE.nc')
true_sic_da = xr.open_dataarray(true_sic_fpath, chunks={'time': 12})

#### Bias correct ensemble mean SIC
###############################################################################
seas5_mean_forecast_da = xr.concat(
    [da.mean('number') for da in seas5_forecast_da_list], 'leadtime')

# Compute bias correction fields for each calendar month and lead time
err_da = (seas5_mean_forecast_da - true_sic_da).sel(time=biascorrection_dates)
biascorrection_da = err_da.groupby('time.month').mean('time')

biascorrected_da = seas5_mean_forecast_da.groupby('time.month') - biascorrection_da
biascorrected_da = biascorrected_da.drop('month')

print('Bias correcting ensemble-mean SIC... ', end='', flush=True)
tic = time.time()
biascorrected_da.compute()
dur = time.time() - tic
print("Done in {}m:{:.0f}s.\n\n ".format(np.floor(dur / 60), dur % 60))

biascorrected_da.data[biascorrected_da.data >= 1.] = 1.
biascorrected_da.data[biascorrected_da.data <= 0.] = 0.

print('Saving... ', end='', flush=True)
fpath = os.path.join(seas5_folder, 'seas5_forecasts.nc')
biascorrected_da.to_netcdf(fpath)
print("Done.")

#### Bias correct SIP
###############################################################################
seas5_forecast_da = xr.concat(seas5_forecast_da_list, 'leadtime')

# Bias correct each ensemble member with ensemble mean biascorrection fields
biascorrected_da = seas5_forecast_da.groupby('time.month') - biascorrection_da
biascorrected_da = biascorrected_da.drop('month')

# Compute SIP
sip_da = (biascorrected_da > .15).sum('number') / seas5_forecast_da.number.size

print('Bias correcting SIP... ', end='', flush=True)
tic = time.time()
sip_da.compute()
dur = time.time() - tic
print("Done in {}m:{:.0f}s.\n\n ".format(np.floor(dur / 60), dur % 60))

print('Saving... ', end='', flush=True)
fpath = os.path.join(seas5_folder, 'seas5_sip_forecasts.nc')
sip_da.to_netcdf(fpath)
print("Done.")
