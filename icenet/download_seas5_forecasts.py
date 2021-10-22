import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))
import config
import utils
import pandas as pd
import iris
import warnings
import time
from ecmwfapi import ECMWFService
import numpy as np
import argparse

'''
Script to download 2001-2020 SEAS5 sea ice concentration (SIC) forecast data
from ECMWF and regrid to the EASE grid using iris. All 25 ensemble members are
downloaded. To obtain manageable download sizes, a single monthly forecast lead
time is downloaded at a time (controlled by the --leadtime command line input).

This script requires you to have created and ECMWF account and emailed the
Computing Representative to upgrade your account to access ECMWF MARS Catalogue
data. See the README.

The download_seas5_forecasts_in_parallel.sh bash script runs this script in
parallel to download each lead time in [1, ..., 6] simultaneously.
'''

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--leadtime', default=1)
args = parser.parse_args()

leadtime = args.leadtime

print(f"Leadtime: {leadtime} month/s\n\n")

# USER INPUT SECTION
################################################################################

download_folder = os.path.join(config.forecast_data_folder, 'seas5', 'latlon')
EASE_folder = os.path.join(config.forecast_data_folder, 'seas5', 'EASE')
for folder in [download_folder, EASE_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

do_download = True  # Download the ECMWF SEAS5 historical SIC forecasts
overwrite = True  # Whether to overwite downloaded/regridded data
do_regrid = True  # Regrid from lat/lon to 25km NH EASE grid
delete_after_regridding = True  # Delete large lat-lon data after regridding

init_dates = pd.date_range(
    start='2001-01-01',
    end='2020-12-01',
    freq='MS',
)

init_dates = [date.strftime('%Y-%m-%d') for date in init_dates]

# ECMWF API FOR HIGH RES DATA
################################################################################

server = ECMWFService("mars", url="https://api.ecmwf.int/v1",
                      key=config.ECMWF_API_KEY,
                      email=config.ECMWF_API_EMAIL)

request_dict = {
    'class': 'od',  # For hi-res SEAS5 data (as opposed to 'c3')
    'date': init_dates,
    'expver': 1,
    # 'fcmonth': list(map(int, np.arange(1, 7))),
    'fcmonth': leadtime,
    'number': list(range(25)),
    'levtype': "sfc",
    'method': 1,
    'origin': "ecmf",
    'param': "31.128",
    'stream': "msmm",
    'system': 5,
    'time': "00:00:00",
    'type': "fcmean",
    'grid': '0.25/0.25',
    'format': 'netcdf',
    'area': '90/-180/0/180'
}

# Load SIC data for EASE grid
################################################################################

sic_day_fpath = os.path.join(config.obs_data_folder, 'ice_conc_nh_ease2-250_cdr-v2p0_197901021200.nc')

if not os.path.exists(sic_day_fpath):
    print("Downloading single daily SIC netCDF file for regridding ERA5 data to EASE grid...\n\n")

    # Ignore "Missing CF-netCDF ancially data variable 'status_flag'" warning
    warnings.simplefilter("ignore", UserWarning)

    retrieve_sic_day_cmd = 'wget -m -nH --cut-dirs=6 -P {} ' \
        'ftp://osisaf.met.no/reprocessed/ice/conc/v2p0/1979/01/ice_conc_nh_ease2-250_cdr-v2p0_197901021200.nc'
    os.system(retrieve_sic_day_cmd.format(config.obs_data_folder))

    print('Done.')

# Load a single SIC map to obtain the EASE grid for regridding ERA data
sic_EASE_cube = iris.load_cube(sic_day_fpath, 'sea_ice_area_fraction')

# Convert EASE coord units to metres for regridding
sic_EASE_cube.coord('projection_x_coordinate').convert_units('meters')
sic_EASE_cube.coord('projection_y_coordinate').convert_units('meters')

################################################################################

if not os.path.exists(download_folder):
    os.makedirs(download_folder)

download_filename = f'seas5_leadtime{leadtime}_latlon.nc'
download_path = os.path.join(download_folder, download_filename)

if do_download:
    print('Beginning download...', end='', flush=True)
    tic = time.time()

    if os.path.exists(download_path) and not overwrite:
        print('File exists - skipping.')
        do_download = False #change do_download so that we do not attempt to download
    elif os.path.exists(download_path) and overwrite:
        print('Deleting existing file.')
        os.remove(download_path)
        do_download = True #for clarity

    # File doesn't exist (or will be overwritten)
    if do_download:
        server.execute(request_dict, download_path)

        dur = time.time() - tic
        print("Done in {}m:{:.0f}s.\n\n ".format(np.floor(dur / 60), dur % 60))

if do_regrid:

    EASE_filename = f'seas5_leadtime{leadtime}_EASE.nc'
    EASE_path = os.path.join(EASE_folder, EASE_filename)

    print("Regridding to EASE... ", end='', flush=True)

    if os.path.exists(EASE_path) and not overwrite:
        print('File exists - skipping.')

    elif os.path.exists(EASE_path) and overwrite:
        print('Deleting existing file.')
        os.remove(EASE_path)

    # File doesn't exist (or was just deleted)
    if not os.path.exists(EASE_path):

        ### Regrid forecast data
        #####################################################################

        cube = iris.load_cube(download_path)
        cube = utils.assignLatLonCoordSystem(cube)

        cube = cube.regrid(sic_EASE_cube, iris.analysis.Linear())

        # Save the regridded cube in order to open in Xarray
        if os.path.exists(EASE_path):
            os.remove(EASE_path)
        iris.save(cube, EASE_path)

        if delete_after_regridding:
            os.remove(download_path)

        print("Done")

print('Done.')
