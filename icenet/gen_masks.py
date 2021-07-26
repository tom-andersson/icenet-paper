import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))  # if using jupyter kernel
import numpy as np
import xarray as xr
import os
import shutil
import config
import iris
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

'''
Obtains masks for land, the polar holes, OSI-SAF monthly maximum ice extent (the 'active
grid cell region'), and the Arctic regions & coastline. Figures of the
masks are saved in the figures/ folder.

The polar hole radii were determined from Sections 2.1, 2.2, and 2.3 of
http://osisaf.met.no/docs/osisaf_cdop3_ss2_pum_sea-ice-conc-climate-data-record_v2p0.pdf

Region mask: https://nsidc.org/data/polar-stereo/tools_masks.html#region_masks
'''

###############################################################################

save_active_grid_cell_masks = True  # Save the monthly land-lake-ocean masks for active gridcells
save_land_mask = True  # Save the land mask (constant across months)
save_arctic_region_mask = True  # Save Arctic region mask from NSIDC, with coastline cells
save_polarhole_masks = True  # Save the polarhole masks
save_figures = True  # Figures of the max extent/region masks

temp_ice_data_folder = os.path.join(config.mask_data_folder, 'temp')

if not os.path.isdir(config.mask_data_folder):
    os.makedirs(config.mask_data_folder)

retrieve_cmd_template_osi450 = 'wget --quiet -m -nH --cut-dirs=4 -P ' + temp_ice_data_folder + \
    ' ftp://osisaf.met.no/reprocessed/ice/conc/v2p0/{:04d}/{:02d}/' + '{}'
filename_template_osi450 = 'ice_conc_nh_ease2-250_cdr-v2p0_{:04d}{:02d}021200.nc'

#### Generate the land-lake-sea mask using the second day from each month of
#### the year 2000 (chosen arbitrarily as the mask is fixed within a calendar month)
###############################################################################

print("Generating active grid cell region & and masks\n")

year = 2000

if save_figures:
    fig_folder = os.path.join(config.figure_folder, 'max_extent_masks')
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

#### Active grid cell masks and land mask
###############################################################################

for month in range(1, 13):

    # Download the data if not already downloaded
    filename_osi450 = filename_template_osi450.format(year, month)
    os.system(retrieve_cmd_template_osi450.format(year, month, filename_osi450))

    year_str = '{:04d}'.format(year)
    month_str = '{:02d}'.format(month)
    month_path = os.path.join(temp_ice_data_folder, year_str, month_str)

    # More than just '.listing' was downloaded
    if len(os.listdir(month_path)) > 1:
        day_path = os.path.join(month_path, filename_osi450)

        with xr.open_dataset(day_path) as ds:
            status_flag = ds['status_flag']
            status_flag = np.array(status_flag.data).astype(np.uint8)
            status_flag = status_flag.reshape(432, 432)

            # See status flag definition in Table 4 of OSI-SAF documention:
            #   http://osisaf.met.no/docs/osisaf_cdop3_ss2_pum_sea-ice-conc-climate-data-record_v2p0.pdf
            #   Note 'Bit Nr' corresponds to leftmost bit in the byte (e.g. Bit Nr 0
            #   is index 7)
            binary = np.unpackbits(status_flag, axis=1).reshape(432, 432, 8)

            # Mask out: land, lake, and 'outside max climatology' (i.e. open sea)
            max_extent_mask = np.sum(binary[:, :, [7, 6, 0]], axis=2).reshape(432, 432) >= 1
            max_extent_mask = ~max_extent_mask  # False outside of max extent
            max_extent_mask[325:386, 317:380] = False  # Remove Caspian and Black seas

    if save_active_grid_cell_masks:
        mask_filename = config.active_grid_cell_file_format.format(month_str)
        mask_path = os.path.join(config.mask_data_folder, mask_filename)
        np.save(mask_path, max_extent_mask)

    if save_land_mask and month == 1:
        land_mask = np.sum(binary[:, :, [7, 6]], axis=2).reshape(432, 432) >= 1
        land_mask_path = os.path.join(config.mask_data_folder, config.land_mask_filename)
        np.save(land_mask_path, land_mask)

    if save_figures:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(max_extent_mask, cmap='Blues_r')
        ax.contour(land_mask, colors='k', linewidths=0.3)
        plt.savefig(os.path.join(fig_folder, month_str + '.png'))
        plt.close()

    # Delete the downloaded daily data
    shutil.rmtree(temp_ice_data_folder)

#### Arctic region mask
###############################################################################

if save_arctic_region_mask:
    ### Authors: Tony Phillips (BAS), Tom Andersson (BAS)

    print("Generating NSIDC Arctic sea region array\n")

    # Download the Arctic region mask
    retrieve_arctic_region_mask_cmd = 'wget --quiet -m -nH --cut-dirs=6 -P {} ' \
        'ftp://sidads.colorado.edu/pub/DATASETS/seaice/polar-stereo/tools/region_n.msk'
    os.system(retrieve_arctic_region_mask_cmd.format(os.path.join(config.mask_data_folder)))

    # Download SIC data for regridding
    sic_day_fpath = os.path.join(config.mask_data_folder, 'ice_conc_nh_ease2-250_cdr-v2p0_197901021200.nc')

    if not os.path.exists(sic_day_fpath):

        # Ignore "Missing CF-netCDF ancially data variable 'status_flag'" warning
        warnings.simplefilter("ignore", UserWarning)

        retrieve_sic_day_cmd = 'wget --quiet -m -nH --cut-dirs=6 -P {} ' \
            'ftp://osisaf.met.no/reprocessed/ice/conc/v2p0/1979/01/ice_conc_nh_ease2-250_cdr-v2p0_197901021200.nc'
        os.system(retrieve_sic_day_cmd.format(config.mask_data_folder))

    sic_EASE_cube = iris.load_cube(sic_day_fpath, 'sea_ice_area_fraction')
    sic_EASE_cube.coord('projection_x_coordinate').convert_units('meters')
    sic_EASE_cube.coord('projection_y_coordinate').convert_units('meters')

    # set ellipsoid, coordinate system, grid metadata and polar stereo coordinates for
    # NSIDC 25km polar stereographic sea ice data

    # ellipsoid -- see http://nsidc.org/data/polar-stereo/ps_grids.html
    a = 6378273.0
    e = 0.081816153
    b = a * ((1.0 - e*e) ** 0.5)
    ellipsoid = iris.coord_systems.GeogCS(semi_major_axis=a, semi_minor_axis=b)

    # coordinate system -- see ftp://sidads.colorado.edu/pub/tools/mapx/nsidc_maps/Nps.mpp
    nps_cs = iris.coord_systems.Stereographic(90, -45,
                                              false_easting=0.0, false_northing=0.0,
                                              true_scale_lat=70,
                                              ellipsoid=ellipsoid)

    # grid definition -- see ftp://sidads.colorado.edu/pub/tools/mapx/nsidc_maps/N3B.gpd
    grid_length = 25000   # in m
    nx = 304
    ny = 448
    cx = 153.5
    cy = 233.5

    # derive X and Y coordinates of pixel centres -- Y reversed so it starts at the bottom-left
    x = np.linspace(-cx, (nx-1)-cx, num=nx) * grid_length
    y = np.linspace(cy-(ny-1), cy, num=ny) * grid_length

    # read region data
    region_file_path = os.path.join(config.mask_data_folder, 'region_n.msk')
    region_data = np.fromfile(region_file_path, dtype='b', offset=300)

    # reshape and flip the data in the Y-direction
    region_data = region_data.reshape((ny, nx))[::-1]

    # Shift up by 1 so that class 0 means 'undefined'
    region_data += 1

    # convert to a cube
    x_coord = iris.coords.DimCoord(x, 'projection_x_coordinate', units='m', coord_system=nps_cs)
    y_coord = iris.coords.DimCoord(y, 'projection_y_coordinate', units='m', coord_system=nps_cs)

    regions = iris.cube.Cube(region_data, dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])

    regions_ease = regions.regrid(sic_EASE_cube, iris.analysis.Nearest(extrapolation_mode='mask'))

    # Save the mask as a Numpy array and remove the temporary files
    arctic_region_mask = regions_ease.data.data

    ###### Extract the coastline properly using numpy
    land_mask_path = os.path.join(config.mask_data_folder, config.land_mask_filename)
    land_mask = np.load(land_mask_path)

    # Masked elements over sea
    sea_mask = np.ma.masked_array(np.full((432, 432), 0.))
    sea_mask[~land_mask] = np.ma.masked

    coast_arrays = {}
    for direction in ('horiz', 'vertic'):

        # C-style indexing for horizontal raveling; F-style for vertical raveling
        if direction == 'horiz':
            order = 'C'  # Scan columns fastest
        elif direction == 'vertic':
            order = 'F'  # Scan rows fastest

        # Tuples with starts and ends indexes of masked element chunks
        slice_ends = np.ma.clump_masked(sea_mask.ravel(order=order))

        coast_idxs = []
        coast_idxs.extend([s.start for s in slice_ends])
        coast_idxs.extend([s.stop - 1 for s in slice_ends])

        coastline_i = np.array(np.full((432, 432), False), order=order)
        coastline_i.ravel(order=order)[coast_idxs] = True
        coast_arrays[direction] = coastline_i

    coastline = coast_arrays['horiz'] + coast_arrays['vertic']

    # Remove artefacts along edge of the grid
    coastline[:, 0] = coastline[0, :] = coastline[:, -1] = coastline[-1, :] = False

    #  Paste over the land mask
    arctic_region_mask[land_mask] = 12
    # Paste over new coastline grid cells
    arctic_region_mask[coastline] = 13

    # Make 'Undefined' cells outside of region equal to 'Open sea'
    arctic_region_mask[arctic_region_mask == 0] = 2

    # Save the Arctic mask array
    np.save(os.path.join(config.mask_data_folder, config.region_mask_filename), arctic_region_mask)

    # Make pan-Arctic region plot
    land_map = np.zeros((432, 432))
    land_map[land_mask] = 1.

    if save_figures:
        fig_folder = os.path.join(config.figure_folder, 'regions')
        if not os.path.exists(fig_folder):
            os.makedirs(fig_folder)

        fig, axes = plt.subplots(figsize=(40, 20), nrows=1, ncols=2)
        im = axes[0].imshow(regions_ease.data.data, cmap=plt.get_cmap('tab20c', 14), clim=(-0.5, 13.5))
        axes[0].contour(land_map, levels=[0.5], colors='k', linewidths=0.5)
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        region_names = ['Undefined', 'Lakes', 'Open sea', 'Sea of Okhotsk and Japan',
                        'Bering Sea', 'Hudson Bay', 'Baffin Bay - Davis Strait - Labrador Sea',
                        'Greenland Sea', 'Barents and Kara Seas', 'Arctic Ocean',
                        'Canadian Archipelago', 'Gulf of St. Lawrence', 'Land', 'Coast']
        formatter = mpl.ticker.FuncFormatter(lambda val, loc: region_names[loc])
        plt.colorbar(im, cax, ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], format=formatter)
        axes[0].set_title('Arctic region segmentation', fontsize=20)
        getattr(mpl.cm, 'Blues_r').set_bad(color='black')
        axes[1].imshow(sic_EASE_cube.data[0, :], cmap='Blues_r')
        axes[1].contour(land_map, levels=[0.5], colors='k', linewidths=0.5)
        axes[1].contour(regions_ease.data.data, levels=np.arange(0,13,1), colors='r', linewidths=2)
        axes[1].set_title('SIC day map with Arctic region segmentation', fontsize=20)
        plt.savefig(os.path.join(fig_folder, 'all_regions.pdf'))
        plt.savefig(os.path.join(fig_folder, 'all_regions.png'))
        plt.close()

        for i, region_str in enumerate(region_names):
            fig, ax = plt.subplots(figsize=(10, 10))

            ax.contour(land_map, levels=[0.5], colors='k', linewidths=0.5)
            ax.imshow(sic_EASE_cube.data[0, :], cmap='Blues_r')
            ax.contour(regions_ease.data.data == i, levels=np.arange(0,13,1), colors='r', linewidths=2)

            plt.savefig(os.path.join(fig_folder, region_str + '.pdf'))
            plt.savefig(os.path.join(fig_folder, region_str + '.png'))
            plt.close()

    os.remove(sic_day_fpath)
    os.remove(region_file_path)

#### Polar hole masks
###############################################################################

if save_polarhole_masks:

    print("Generating polar hole masks\n")

    #### Generate the polar hole masks
    x = np.tile(np.arange(0, 432).reshape(432, 1), (1, 432)).astype(np.float32) - 215.5
    y = np.tile(np.arange(0, 432).reshape(1, 432), (432, 1)).astype(np.float32) - 215.5
    squaresum = np.square(x) + np.square(y)

    # Jan 1979 - June 1987
    polarhole1 = np.full((432, 432), False)
    polarhole1[squaresum < config.polarhole1_radius**2] = True
    np.save(os.path.join(config.mask_data_folder, config.polarhole1_fname), polarhole1)

    # July 1987 - Oct 2005
    polarhole2 = np.full((432, 432), False)
    polarhole2[squaresum < config.polarhole2_radius**2] = True
    np.save(os.path.join(config.mask_data_folder, config.polarhole2_fname), polarhole2)

    # Nov 2005 - Dec 2015
    polarhole3 = np.full((432, 432), False)
    polarhole3[squaresum < config.polarhole3_radius**2] = True
    np.save(os.path.join(config.mask_data_folder, config.polarhole3_fname), polarhole3)

print("\nDone.")
