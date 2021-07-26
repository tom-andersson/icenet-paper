import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))  # if using jupyter kernel
import config
import iris
import xarray as xr
import time
import warnings
import numpy as np
import utils
import argparse

"""
Script to rotate ERA5 and CMIP6 wind vector data to the EASE grid. Replaces the
uas_EASE and vas_EASE files with their rotated counterparts.

The rotate_wind_data_in_parallel.sh bash scipt runs this script in parallel for
the ERA5 and all climate simulation data in parallel. Note this can have heavy
CPU demand!
"""

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--cmip6', default=False, help='Rotate CMIP6 wind rather than ERA5',
                    action='store_true')
parser.add_argument('--source_id', default='EC-Earth3', type=str)
parser.add_argument('--member_id', default='r2i1p1f1', type=str)
commandline_args = parser.parse_args()

cmip6 = commandline_args.cmip6
source_id = commandline_args.source_id
member_id = commandline_args.member_id

overwrite = True
gen_video = True

# Check wind magnitude before and after is the same
verify_wind_magnitude = True

################################################################################

if not cmip6:
    wind_data_folder = config.obs_data_folder
    fname_suffix = '_EASE.nc'
elif cmip6:
    wind_data_folder = os.path.join(config.cmip6_data_folder, source_id, member_id)
    fname_suffix = '_EASE_cmpr.nc'

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

land_mask = np.load(os.path.join(config.mask_data_folder, config.land_mask_filename))

# get the gridcell angles
angles = utils.gridcell_angles_from_dim_coords(sic_EASE_cube)

# invert the angles
utils.invert_gridcell_angles(angles)

# Rotate, regrid, and save
################################################################################

tic = time.time()

print(f'\nRotating wind data in {wind_data_folder}')
wind_cubes = {}
for var in ['uas', 'vas']:
    EASE_path = os.path.join(wind_data_folder, f'{var}{fname_suffix}')
    wind_cubes[var] = iris.load_cube(EASE_path)

# rotate the winds using the angles
wind_cubes_r = {}
wind_cubes_r['uas'], wind_cubes_r['vas'] = utils.rotate_grid_vectors(
    wind_cubes['uas'], wind_cubes['vas'], angles)

# save the new cube
for var, cube_ease_r in wind_cubes_r.items():
    EASE_path = os.path.join(wind_data_folder, f'{var}{fname_suffix}')

    if os.path.exists(EASE_path) and overwrite:
        print("Removing existing file: {}". format(EASE_path))
        os.remove(EASE_path)
    elif os.path.exists(EASE_path) and not overwrite:
        print("Skipping due to existing file: {}". format(EASE_path))
        sys.exit()

    iris.save(cube_ease_r, EASE_path)

if gen_video:
    for var in ['uas', 'vas']:
        print(f'generating video for {var}')
        EASE_path = os.path.join(wind_data_folder, f'{var}{fname_suffix}')

        video_folder = os.path.join('videos', 'wind')
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
        fname = '{}_{}.mp4'.format(wind_data_folder.replace('/', '_'), var)
        video_path = os.path.join(video_folder, fname)

        utils.xarray_to_video(
            da=next(iter(xr.open_dataset(EASE_path).data_vars.values())),
            video_path=video_path,
            fps=6,
            mask=land_mask,
            figsize=7,
            dpi=100,
        )

toc = time.time()
print("Done in {:.3f}s.".format(toc - tic))
