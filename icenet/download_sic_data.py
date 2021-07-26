import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))
import config
from tqdm import tqdm
import pandas as pd
import xarray as xr
import time
import numpy as np
from scipy import interpolate

'''
Downloads OSI-SAF SIC data from 1979-present using OpenDAP.
The dataset comprises OSI-450 (1979-2015) and OSI-430-b (2016-ownards)
Monthly averages are-computed on the server-side.
This script can take about an hour to run.

The query URLs were obtained from the following sites:
    - OSI-450 (1979-2016): https://thredds.met.no/thredds/dodsC/osisaf/met.no/reprocessed/ice/conc_v2p0_nh_agg.html
    - OSI-430-b (2016-present): https://thredds.met.no/thredds/dodsC/osisaf/met.no/reprocessed/ice/conc_crb_nh_agg.html
'''

if not os.path.exists(config.obs_data_folder):
    os.makedirs(config.obs_data_folder)

print('Downloading OSI-450 monthly averages... ', end='', flush=True)
tic = time.time()
da_osi450 = xr.open_dataarray('https://thredds.met.no/thredds/dodsC/osisaf/met.no/reprocessed/ice/conc_v2p0_nh_agg?xc[0:1:431],yc[0:1:431],lat[0:1:431][0:1:431],lon[0:1:431][0:1:431],time[0:1:11800],ice_conc[0:1:11800][0:1:431][0:1:431]')
da_osi450 = da_osi450.resample(time='1MS').mean()
dur = time.time() - tic
print("Done in {}m:{:.0f}s.\n\n ".format(np.floor(dur / 60), dur % 60))

print('Downloading OSI-430-b monthly averages... ', end='', flush=True)
tic = time.time()
da_osi430b = xr.open_dataarray('https://thredds.met.no/thredds/dodsC/osisaf/met.no/reprocessed/ice/conc_crb_nh_agg?xc[0:1:431],yc[0:1:431],lat[0:1:431][0:1:431],lon[0:1:431][0:1:431],time[0:1:2008],ice_conc[0:1:2008][0:1:431][0:1:431]')
da_osi430b = da_osi430b.resample(time='1MS').mean()
da_osi430b.lat.values = da_osi450.lat.values  # Fix inconsistent latitude values
dur = time.time() - tic
print("Done in {}m:{:.0f}s.\n\n ".format(np.floor(dur / 60), dur % 60))

print('Concatenating... ', end='', flush=True)
tic = time.time()
da = xr.concat([da_osi450, da_osi430b], dim='time')
dur = time.time() - tic
print("Done in {}m:{:.0f}s.\n\n ".format(np.floor(dur / 60), dur % 60))

### Preprocess the data
da /= 100.  # Convert from SIC % to fraction

mask_fpath_format = os.path.join(config.mask_data_folder,
                                 config.active_grid_cell_file_format)

# Load grid cell mask
mask_dict = {
    month: np.load(mask_fpath_format.format('{:02d}'.format(month))) for
    month in np.arange(1, 12+1)
}

dates = [pd.Timestamp(date) for date in da.time.values]

print('Preprocessing the SIC data:')
tic = time.time()
x = da['xc'].data
y = da['yc'].data
for date in tqdm(dates):

    # Grab mask
    mask = mask_dict[date.month]

    # Set outside mask to zero
    da.loc[date].data[~mask] = 0.

    # Grab polar hole
    skip_interp = False
    if date <= config.polarhole1_final_date:
        polarhole_mask = np.load(os.path.join(config.mask_data_folder, config.polarhole1_fname))
    elif date <= config.polarhole2_final_date:
        polarhole_mask = np.load(os.path.join(config.mask_data_folder, config.polarhole2_fname))
    elif date <= config.polarhole3_final_date and config.use_polarhole3:
        polarhole_mask = np.load(os.path.join(config.mask_data_folder, config.polarhole3_fname))
    else:
        skip_interp = True

    # Interpolate polar hole
    if not skip_interp:

        xx, yy = np.meshgrid(np.arange(432), np.arange(432))

        valid = ~polarhole_mask

        x = xx[valid]
        y = yy[valid]

        x_interp = xx[polarhole_mask]
        y_interp = yy[polarhole_mask]

        values = da.loc[date].data[valid]

        interp_vals = interpolate.griddata((x, y), values, (x_interp, y_interp), method='linear')
        interpolated_array = da.loc[date].data.copy()
        interpolated_array[polarhole_mask] = interp_vals
        da.loc[date].data = interpolated_array

dur = time.time() - tic
print("Done in {}m:{:.0f}s.\n\n ".format(np.floor(dur / 60), dur % 60))

fpath = os.path.join(config.obs_data_folder, 'siconca_EASE.nc')
if os.path.exists(fpath):
    os.remove(fpath)
da.to_netcdf(fpath)
