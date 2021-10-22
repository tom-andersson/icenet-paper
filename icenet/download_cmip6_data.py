import os
import numpy as np
import sys
import iris
import time
import warnings
import xarray as xr
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))
import config
import argparse
import utils

'''
Script to download monthly-averaged CMIP6 climate simulation runs from the Earth
System Grid Federation (ESFG):
https://esgf-node.llnl.gov/search/cmip6/.
The simulations are regridded from latitude/longitude to the same EASE grid as
the OSI-SAF sea ice data.

The --source_id and --member_id command line inputs control which climate model
and model run to download.

The `download_dict` dictates which variables to download from each climate
model. Entries within the variable dictionaries of `variable_dict` provide
further specification for the variable to download - e.g. whether it is on an
ocean grid and whether to download data at a specified pressure level. All this
information is used to create the `query` dictionary that is passed to
utils.esgf_search to find download links for the variable. The script loops
through each variable, downloading those for which 'include' is True in the
variable dictionary.

Variable files are saved to cmip6/<source_id>/<member_id>/ in <var>_EASE_cmpr.nc
format.

See download_cmip6_data_in_parallel.sh to download and regrid multiple climate
simulations in parallel using this script.
'''

#### COMMAND LINE INPUT
# --------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--source_id', default='EC-Earth3', type=str)
parser.add_argument('--member_id', default='r2i1p1f1', type=str)
commandline_args = parser.parse_args()

source_id = commandline_args.source_id
member_id = commandline_args.member_id

print('\n\nDownloading data for {}, {}\n'.format(source_id, member_id))

####### User download options
################################################################################

overwrite = False
delete_latlon_data = True  # Delete lat-lon intermediate files is use_xarray is True
compress = True

do_download = True
do_regrid = True
gen_video = True

download_dict = {
    'MRI-ESM2-0': {
        'experiment_ids': ['historical', 'ssp245'],
        'data_nodes': ['esgf-data2.diasjp.net'],
        'frequency': 'mon',
        'variable_dict': {
            'siconca': {
                'include': True,
                'table_id': 'SImon',
                'plevels': None
            },
            'tas': {
                'include': True,
                'table_id': 'Amon',
                'plevels': None
            },
            'ta': {
                'include': True,
                'table_id': 'Amon',
                'plevels': [500_00]
            },
            'tos': {
                'include': True,
                'table_id': 'Omon',
                'plevels': None,
                'ocean_variable': True
            },
            'psl': {
                'include': True,
                'table_id': 'Amon',
                'plevels': None
            },
            'rsus': {
                'include': True,
                'table_id': 'Amon',
                'plevels': None
            },
            'rsds': {
                'include': True,
                'table_id': 'Amon',
                'plevels': None
            },
            'zg': {
                'include': True,
                'table_id': 'Amon',
                'plevels': [500_00, 250_00]
            },
            'uas': {
                'include': True,
                'table_id': 'Amon',
                'plevels': None
            },
            'vas': {
                'include': True,
                'table_id': 'Amon',
                'plevels': None
            },
            'ua': {
                'include': True,
                'table_id': 'Amon',
                'plevels': [10_00]
            },
        }
    },
    'EC-Earth3': {
        'experiment_ids': ['historical', 'ssp245'],
        'data_nodes': ['esgf.bsc.es'],
        'frequency': 'mon',
        'variable_dict': {
            'siconca': {
                'include': True,
                'table_id': 'SImon',
                'plevels': None
            },
            'tas': {
                'include': True,
                'table_id': 'Amon',
                'plevels': None
            },
            'ta': {
                'include': True,
                'table_id': 'Amon',
                'plevels': [500_00]
            },
            'tos': {
                'include': True,
                'table_id': 'Omon',
                'plevels': None,
                'ocean_variable': True
            },
            'psl': {
                'include': True,
                'table_id': 'Amon',
                'plevels': None
            },
            'rsus': {
                'include': True,
                'table_id': 'Amon',
                'plevels': None
            },
            'rsds': {
                'include': True,
                'table_id': 'Amon',
                'plevels': None
            },
            'zg': {
                'include': True,
                'table_id': 'Amon',
                'plevels': [500_00, 250_00]
            },
            'uas': {
                'include': True,
                'table_id': 'Amon',
                'plevels': None
            },
            'vas': {
                'include': True,
                'table_id': 'Amon',
                'plevels': None
            },
            'ua': {
                'include': True,
                'table_id': 'Amon',
                'plevels': [10_00]
            },
        }
    }
}

download_folder = os.path.join(config.cmip6_data_folder, source_id, member_id)
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

##### Load SIC data for EASE grid and import the land mask
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

##### Download
################################################################################

# Ignore "Missing CF-netCDF variable" warnings from download
warnings.simplefilter("ignore", UserWarning)

tic = time.time()

source_id_dict = download_dict[source_id]

query = {
    'source_id': source_id,
    'member_id': member_id,
    'frequency': source_id_dict['frequency'],
}

for variable_id, variable_id_dict in source_id_dict['variable_dict'].items():

    # variable_id = 'vas'
    # variable_id_dict = source_id_dict['variable_dict'][variable_id]

    if variable_id_dict['include'] is False:
        continue

    query['variable_id'] = variable_id
    query['table_id'] = variable_id_dict['table_id']

    if 'ocean_variable' in variable_id_dict.keys() or source_id == 'EC-Earth3':
        query['grid_label'] = 'gr'
    else:
        query['grid_label'] = 'gn'

    print('\n\n{}: '.format(variable_id), end='', flush=True)

    video_folder = os.path.join(config.video_folder, 'cmip6', source_id, member_id)

    # Paths for each plevel (None if surface variable)
    fpaths_EASE = {}
    fpaths_latlon = {}
    video_fpaths = {}

    if variable_id_dict['plevels'] is None:
        variable_id_dict['plevels'] = [None]

    skip = {}  # Whether to skip each plevel variable
    existing_EASE_fpaths = []

    for plevel in variable_id_dict['plevels']:

        fname = variable_id
        if plevel is not None:
            # suffix for the pressure level in hPa
            fname += '{:.0f}'.format(plevel / 100)

        # Intermediate lat-lon file before iris regridding
        fpaths_latlon[plevel] = os.path.join(download_folder, fname + '_latlon.nc')

        fname += '_EASE'
        video_fpaths[plevel] = os.path.join(video_folder, fname + '.mp4')
        if compress:
            fname += '_cmpr'
        fpaths_EASE[plevel] = os.path.join(download_folder, fname + '.nc')

        if os.path.exists(fpaths_EASE[plevel]):
            if overwrite:
                print('removing existing file... ', end='', flush=True)
                os.remove(fpaths_EASE[plevel])
                skip[plevel] = False
            else:
                skip[plevel] = True
                existing_EASE_fpaths.append(fpaths_EASE[plevel])
        else:
            skip[plevel] = False

    skipall = all([skip_bool for skip_bool in skip.values()])

    if skipall:
        print('skipping due to existing files {}'.format(existing_EASE_fpaths), end='', flush=True)
        continue

    if do_download:
        print('searching ESGF... ', end='', flush=True)
        results = []
        for experiment_id in source_id_dict['experiment_ids']:
            query['experiment_id'] = experiment_id

            experiment_id_results = []
            for data_node in source_id_dict['data_nodes']:
                query['data_node'] = data_node

                experiment_id_results.extend(utils.esgf_search(**query))

                # Keep looping over possible data nodes until the experiment data is found
                if len(experiment_id_results) > 0:
                    print('found {}, '.format(experiment_id), end='', flush=True)
                    results.extend(experiment_id_results)
                    break  # Break out of the loop over data nodes

        results = list(set(results))
        print('found {} files. '.format(len(results)), end='', flush=True)

        for plevel in variable_id_dict['plevels']:
            if plevel is not None:
                print('{} hPa, '.format(plevel / 100), end='', flush=True)

            if skip[plevel]:
                print('skipping this plevel due to existing file {}'.format(fpaths_EASE[plevel]), end='', flush=True)
                continue

            print('loading metadata... ', end='', flush=True)

            # Avoid 500MB DAP request limit
            cmip6_da = xr.open_mfdataset(results, combine='by_coords', chunks={'time': '499MB'})[variable_id]

            if plevel is not None:
                cmip6_da = cmip6_da.sel(plev=plevel)

            print('downloading with xarray... ', end='', flush=True)
            cmip6_da.compute()

            print('saving to regrid in iris... ', end='', flush=True)
            cmip6_da.to_netcdf(fpaths_latlon[plevel])

    if do_regrid:
        for plevel in variable_id_dict['plevels']:

            if skip[plevel]:
                print('skipping this plevel due to existing file {}'.format(fpaths_EASE[plevel]), end='', flush=True)
                continue

            cmip6_cube = iris.load_cube(fpaths_latlon[plevel])
            cmip6_ease = utils.regrid_cmip6(cmip6_cube, sic_EASE_cube, verbose=True)

            # Preprocessing
            if variable_id == 'siconca':
                cmip6_ease.data[cmip6_ease.data > 500] = 0.
                cmip6_ease.data[:, land_mask] = 0.
                if source_id == 'MRI-ESM2-0':
                    cmip6_ease.data = cmip6_ease.data / 100.
            elif variable_id == 'tos':
                cmip6_ease.data[cmip6_ease.data > 500] = 0.
                cmip6_ease.data[:, land_mask] = 0.

            if cmip6_ease.data.dtype != np.float32:
                cmip6_ease.data = cmip6_ease.data.astype(np.float32)

            fpaths_EASE[plevel]
            utils.save_cmip6(cmip6_ease, fpaths_EASE[plevel], compress, verbose=True)

            if delete_latlon_data:
                os.remove(fpaths_latlon[plevel])

    if gen_video:
        if (source_id, member_id) == ('MRI-ESM2-0', 'r2i1p1f1') or \
                (source_id, member_id) == ('EC-Earth3', 'r2i1p1f1'):
            print('\nGenerating video... ')
            utils.xarray_to_video(
                da=next(iter(xr.open_dataset(fpaths_EASE[plevel]).data_vars.values())),
                video_path=video_fpaths[plevel],
                fps=30,
                mask=land_mask,
                figsize=7,
                dpi=150,
            )

    print('Done.\n\n')

dur = time.time() - tic
print("\n\nTOTAL DURATION: {:.0f}m:{:.0f}s\n".format(np.floor(dur / 60), dur % 60))
