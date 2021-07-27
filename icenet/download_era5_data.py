import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))  # if using jupyter kernel
import config
import cdsapi
import xarray as xr
import argparse
import iris
import time
import warnings
import sys
import os
import numpy as np
from utils import assignLatLonCoordSystem, fix_near_real_time_era5_func, \
    xarray_to_video

"""
Script to download monthly-averaged ERA5 reanalysis variables from the Climate
Data Store (CDS) and regrid them from latitude/longitude to the same EASE grid
as the OSI-SAF sea ice data.

Single-level (surface) variables:
https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview

Pressure-level variables:
https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels-monthly-means?tab=overview

The --var command line input controls which variable to download (using CMIP6
variable naming convention:
https://docs.google.com/spreadsheets/d/1UUtoz6Ofyjlpx5LdqhKcwHFz2SGoTQV2_yekHyMfL9Y/edit#gid=1221485271.
The `variables` dict maps from CMIP6 variable names to the CDS naming scheme.

See download_era5_data_in_parallel.sh to download and regrid multiple
variables in parallel using this script.

Files are saved to data/obs/ in <var>_EASE.nc format.

You need to have set up your ~/.cdsapirc file before running this - see the
README.
"""


################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--var', default='tas')
args = parser.parse_args()

variable = args.var

# User input
################################################################################

# Whether to skip variables that have already been downloaded or regridded
overwrite = True

do_download = True
do_regrid = True
gen_video = False

area = [90, -180, 0, 180]  # Latitude/longitude boundaries to download

# Which years to download
years = ['1979', '1980', '1981', '1982', '1983', '1984',
         '1985', '1986', '1987', '1988', '1989', '1990',
         '1991', '1992', '1993', '1994', '1995', '1996',
         '1997', '1998', '1999', '2000', '2001', '2002',
         '2003', '2004', '2005', '2006', '2007', '2008',
         '2009', '2010', '2011', '2012', '2013', '2014',
         '2015', '2016', '2017', '2018', '2019', '2020',
         '2021']

months = ['01', '02', '03', '04', '05', '06',
          '07', '08', '09', '10', '11', '12']

# Near-real-time data contains ERA5T with 'expver' coord -- remove 'expver'
#   dim and concatenate into one array
fix_near_real_time_era5_coords = True

# Variable information
################################################################################

# To add more variables, go to the following dataset sites, fill in the download
#   form for the desired variable, and check the variable's CDI name by clicking
#   'show API request':
#   - Surface vars: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=form
#   - Pressure level vars: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels-monthly-means?tab=overview
variables = {
    'tas': {
        'cdi_name': '2m_temperature',
    },
    'ta500': {
        'plevel': '500',
        'cdi_name': 'temperature',
    },
    'tos': {
        'cdi_name': 'sea_surface_temperature',
    },
    # RSUS needs to be computed using net solar radiation and downwards radiation,
    # and need to convert from J/m^2 to W/m^2 by dividing by the number of seconds
    # in a day (60*60*25)
    'rsds_and_rsus': {
        'rss_cdi_name': 'surface_net_solar_radiation',
        'rsds_cdi_name': 'surface_solar_radiation_downwards',
    },
    'psl': {
        'cdi_name': 'mean_sea_level_pressure',
    },
    'zg500': {
        'plevel': '500',
        'cdi_name': 'geopotential',
    },
    'zg250': {
        'plevel': '250',
        'cdi_name': 'geopotential',
    },
    'ua10': {
        'plevel': '10',
        'cdi_name': 'u_component_of_wind',
    },
    # Note: the surface wind variables are not regridded here; a separate script
    #   is used to rotate and regrid them.
    'uas': {
        'cdi_name': '10m_u_component_of_wind',
    },
    'vas': {
        'cdi_name': '10m_v_component_of_wind',
    },
}

var_dict = variables[variable]

if not os.path.isdir(config.obs_data_folder):
    os.makedirs(config.obs_data_folder)

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

# Download and regridding functions
################################################################################


def retrieve_CDS_data(var_cdi_name, latlon_path, plevel=None):

    print("\nDownloading data for {}...\n".format(var_cdi_name))
    tic = time.time()

    if os.path.exists(latlon_path) and overwrite:
        print("Removing existing file: {}".format(latlon_path))
        os.remove(latlon_path)
    elif os.path.exists(latlon_path) and not overwrite:
        print("Skipping download due to existing file: {}". format(latlon_path))
        return 0

    cds_dict = {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': var_cdi_name,
        'year': years,
        'month': months,
        'time': '00:00',
        'format': 'netcdf',
        'area': area
    }

    if plevel is not None:
        dataset = 'reanalysis-era5-pressure-levels-monthly-means'
        cds_dict['pressure_level'] = plevel
    else:
        dataset = 'reanalysis-era5-single-levels-monthly-means'

    cds.retrieve(dataset, cds_dict, latlon_path)

    toc = time.time()
    print("Done in {:.3f}s.".format(toc - tic))


def regrid_var(variable, EASE_path):

    # Load the monthly averaged ERA5 reanalysis data as iris cubes
    print("\nRegridding and saving {} reanalysis data... ".format(variable), end='', flush=True)
    tic = time.time()

    latlon_path = os.path.join(config.obs_data_folder, '{}_latlon.nc'.format(variable))

    if os.path.exists(EASE_path) and overwrite:
        print("Removing existing file: {}". format(EASE_path))
        os.remove(EASE_path)
    if os.path.exists(EASE_path) and not overwrite:
        print("Skipping regrid due to existing file: {}". format(EASE_path))
        return 0

    if fix_near_real_time_era5_coords:
        fix_near_real_time_era5_func(latlon_path)

    cube = iris.load_cube(latlon_path)
    cube = assignLatLonCoordSystem(cube)

    # Regrid onto the EASE grid
    cube_ease = cube.regrid(sic_EASE_cube, iris.analysis.Linear())

    # Further processing needed
    if variable in ['tos', 'zg500', 'zg250']:
        if variable == 'tos':
            # Overwrite maksed values with zeros
            cube_ease.data[cube_ease.data > 500.] = 0.
            cube_ease.data[cube_ease.data < 0.] = 0.

            cube_ease.data[:, land_mask] = 0.

            cube_ease.data = cube_ease.data.data  # Remove mask from masked array
        elif variable in ['zg500', 'zg250']:
            # Convert from geopotential to geopotential height
            cube_ease /= 9.80665

    # Save the new cube
    iris.save(cube_ease, EASE_path)

    toc = time.time()
    print("Done in {:.3f}s.".format(toc - tic))


# Download and regrid
################################################################################

cds = cdsapi.Client()

if variable != 'rsds_and_rsus':

    latlon_path = os.path.join(config.obs_data_folder, '{}_latlon.nc'.format(variable))

    if 'plevel' not in var_dict.keys():
        plevel = None
    else:
        plevel = var_dict['plevel']

    if do_download:
        retrieve_CDS_data(var_dict['cdi_name'], latlon_path, plevel)

    if do_regrid:
        # Regrid to EASE
        EASE_path = os.path.join(config.obs_data_folder, '{}_EASE.nc'.format(variable))
        regrid_var(variable, EASE_path)

        # Delete lat-lon data
        os.remove(latlon_path)

    if gen_video:
        EASE_path = os.path.join(config.obs_data_folder, '{}_EASE.nc'.format(variable))
        video_folder = os.path.join('videos', 'era5')
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
        video_path = os.path.join(video_folder, f'{variable}.mp4')
        xarray_to_video(
            da=next(iter(xr.open_dataset(EASE_path).data_vars.values())),
            video_path=video_path,
            fps=6,
            mask=land_mask,
            figsize=10,
            dpi=150,
        )

elif variable == 'rsds_and_rsus':

    da_dict = {}
    download_paths_dict = {}
    for radiation_variable in ('rss', 'rsds'):

        latlon_path = os.path.join(config.obs_data_folder, '{}_latlon.nc'.format(radiation_variable))

        retrieve_CDS_data(var_dict['{}_cdi_name'.format(radiation_variable)], latlon_path)

        download_paths_dict[radiation_variable] = latlon_path
        da_dict[radiation_variable] = xr.open_dataarray(latlon_path)

    # Compute upwelling solar, convert to W/m^2
    rsus_da = (da_dict['rsds'] - da_dict['rss']) / (60 * 60 * 24)
    rsus_da = rsus_da.rename('rsus')

    rsds_da = da_dict['rsds'] / (60 * 60 * 24)

    # Delete downloaded data
    for path in download_paths_dict.values():
        os.remove(path)

    # Save new, processed data
    rsds_da.to_netcdf(os.path.join(config.obs_data_folder, 'rsds_latlon.nc'))
    rsus_da.to_netcdf(os.path.join(config.obs_data_folder, 'rsus_latlon.nc'))

    for radiation_variable in ('rsus', 'rsds'):
        # Regrid to EASE
        EASE_path = os.path.join(config.obs_data_folder, '{}_EASE.nc'.format(radiation_variable))
        regrid_var(radiation_variable, EASE_path)

        if gen_video:
            video_folder = os.path.join('videos', 'era5')
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            video_path = os.path.join(video_folder, f'{radiation_variable}.mp4')
            xarray_to_video(
                da=next(iter(xr.open_dataset(EASE_path).data_vars.values())),
                video_path=video_path,
                fps=6,
                mask=land_mask,
                figsize=10,
                dpi=150,
            )

        # Delete lat-lon data
        latlon_path = os.path.join(config.obs_data_folder, '{}_latlon.nc'.format(radiation_variable))
        os.remove(latlon_path)
