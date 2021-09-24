import os
import sys
import numpy as np
import tensorflow as tf
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))  # if using jupyter kernel
from models import linear_trend_forecast
import config
import itertools
import requests
import json
import time
import re
import xarray as xr
import pandas as pd
from dateutil.relativedelta import relativedelta
import iris
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio
from tqdm import tqdm


###############################################################################
############### DATA PROCESSING & LOADING
###############################################################################


class IceNetDataPreProcessor(object):
    """
    Normalises IceNet input data and saves the normalised monthly averages
    as .npy files. If preprocessing climate model data for transfer learning,
    the observational normalisation is repeated for the climate model data in order
    to preserve the mapping from raw values to normalised values.

    Data is stored in the following form with observations separated from climate
    model transfer learning data:
     - data/network_datasets/<dataset_name>/obs/tas/2006_04.npy
     - data/network_datasets/<dataset_name>/transfer/MRI-ESM2-0/r1i1p1f1/tas/2056_09.npy

    Normalisation parameters computed over the observational training data are
    stored in a JSON file at data/network_datasets/<dataset_name>/norm_params.json
    so that they are only computed once. Similarly, monthly climatology fields
    used for computing anomaly fields are saved next to the raw NetCDF files so that
    climatologies are only computed once for each variable.
    """

    def __init__(self, dataloader_config_fpath, preproc_vars,
                 n_linear_years, minmax, verbose_level,
                 preproc_obs_data=True,
                 preproc_transfer_data=False, cmip_transfer_data={}):
        """
        Parameters:

        dataloader_config_fpath (str): Path to the data loader configuration
            settings JSON file, defining IceNet's input-output data configuration.
            This also defines the dataset name, used as the folder name to
            store the preprocessed data within data/network_datasets/.

        preproc_vars (dict): Which variables to preprocess. Example:

                preproc_vars = {
                    'siconca': {'anom': True, 'abs': True},
                    'tas': {'anom': True, 'abs': False},
                    'tos': {'anom': True, 'abs': False},
                    'rsds': {'anom': True, 'abs': False},
                    'rsus': {'anom': True, 'abs': False},
                    'psl': {'anom': False, 'abs': True},
                    'zg500': {'anom': False, 'abs': True},
                    'zg250': {'anom': False, 'abs': True},
                    'ua10': {'anom': False, 'abs': True},
                    'uas': {'anom': False, 'abs': True},
                    'vas': {'anom': False, 'abs': True},
                    'sfcWind': {'anom': False, 'abs': True},
                    'land': {'metadata': True, 'include': True},
                    'circmonth': {'metadata': True, 'include': True}
                }

        n_linear_years (int): Number of past years to used in the linear trend
        projections.

        minmax (bool): Whether to use min-max normalisation to (-1, 1) or normalise
        the mean and standard deviation to 0 and 1.

        verbose_level (int): Controls how much to print. 0: Print nothing.
        1: Print key set-up stages. 2: Print debugging info.

        preproc_obs_data (bool): Whether to preprocess observational data
        (default True).

        preproc_transfer_data (bool): Whether to also preprocess CMIP6 data for each variable
        (default False).

        cmip_transfer_data (dict): Which CMIP6 models & model runs to
        preprocess for transfer learning. Example:

                cmip_transfer_data = {
                    'MRI-ESM2-0': ('r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1',
                                   'r4i1p1f1', 'r5i1p1f1')
                }

        """

        with open(dataloader_config_fpath, 'r') as readfile:
            self.config = json.load(readfile)

        self.preproc_vars = preproc_vars
        self.n_linear_years = n_linear_years
        self.minmax = minmax
        self.verbose_level = verbose_level
        self.preproc_obs_data = preproc_obs_data
        self.preproc_transfer_data = preproc_transfer_data
        self.cmip_transfer_data = cmip_transfer_data

        self.load_or_instantiate_norm_params_dict()
        self.set_obs_train_dates()
        self.set_up_folder_hierarchy()

        if self.verbose_level >= 1:
            print("Loading and normalising the raw input maps.\n")
            tic = time.time()

        self.preproc_and_save_icenet_data()

        if self.verbose_level >= 1:
            print("\nPreprocessing completed in {:.0f}s.\n".format(time.time() - tic))

    def load_or_instantiate_norm_params_dict(self):

        # Path to JSON file storing normalisation parameters for each variable
        self.norm_params_fpath = os.path.join(
            config.network_dataset_folder, self.config['dataset_name'], 'norm_params.json')

        if not os.path.exists(self.norm_params_fpath):
            self.norm_params = {}

        else:
            with open(self.norm_params_fpath, 'r') as readfile:
                self.norm_params = json.load(readfile)

    def set_obs_train_dates(self):

        forecast_start_date_ends = self.config['sample_IDs']['obs_train_dates']

        if forecast_start_date_ends is not None:

            # Convert to Pandas Timestamps
            forecast_start_date_ends = [
                pd.Timestamp(date).to_pydatetime() for date in forecast_start_date_ends
            ]

            self.obs_train_dates = list(pd.date_range(
                forecast_start_date_ends[0],
                forecast_start_date_ends[1],
                freq='MS',
                closed='right',
            ))

    def set_up_folder_hierarchy(self):

        """
        Initialise the folders to store the datasets.
        """

        if self.verbose_level >= 1:
            print('Setting up the folder hierarchy for {}... '.format(self.config['dataset_name']),
                  end='', flush=True)

        # Parent folder for this dataset
        self.dataset_path = os.path.join(config.data_folder, 'network_datasets', self.config['dataset_name'])

        # Dictionary data structure to store folder paths
        self.paths = {}

        # Set up the folder hierarchy
        self.paths['obs'] = {}

        for varname, vardict in self.preproc_vars.items():

            if 'metadata' not in vardict.keys():
                self.paths['obs'][varname] = {}

                for data_format in vardict.keys():

                    if vardict[data_format] is True:
                        path = os.path.join(self.dataset_path, 'obs',
                                            varname, data_format)

                        self.paths['obs'][varname][data_format] = path

                        if not os.path.exists(path):
                            os.makedirs(path)

        self.paths['transfer'] = {}

        for model_name, member_ids in self.cmip_transfer_data.items():
            self.paths['transfer'][model_name] = {}
            for member_id in member_ids:
                self.paths['transfer'][model_name][member_id] = {}

                for varname, vardict in self.preproc_vars.items():

                    if 'metadata' not in vardict.keys():
                        self.paths['transfer'][model_name][member_id][varname] = {}

                        for data_format in vardict.keys():

                            if vardict[data_format] is True:
                                path = os.path.join(self.dataset_path, 'transfer',
                                                    model_name, member_id,
                                                    varname, data_format)

                                self.paths['transfer'][model_name][member_id][varname][data_format] = path

                                if not os.path.exists(path):
                                    os.makedirs(path)

        for varname, vardict in self.preproc_vars.items():
            if 'metadata' in vardict.keys():

                if vardict['include'] is True:
                    path = os.path.join(self.dataset_path, 'meta')

                    self.paths['meta'] = path

                    if not os.path.exists(path):
                        os.makedirs(path)

        if self.verbose_level >= 1:
            print('Done.')

    @staticmethod
    def standardise_cmip6_time_coord(da):

        """
        Convert the cmip6 xarray time dimension to use day=1, hour=0 convention
        used in the rest of the project.
        """

        standardised_dates = []
        for datetime64 in da.time.values:
            date = pd.Timestamp(datetime64, unit='s')
            date = date.replace(day=1, hour=0)
            standardised_dates.append(date)
        da = da.assign_coords({'time': standardised_dates})

        return da

    @staticmethod
    def mean_and_std(list, verbose_level=2):

        # Must use float64s to be JSON serialisable
        mean = np.nanmean(list, dtype=np.float64)
        std = np.nanstd(list, dtype=np.float64)

        return mean, std

    def normalise_array_using_all_training_months(self, da, minmax=False,
                                                  mean=None, std=None,
                                                  min=None, max=None):

        """
        Using the *training* months only, compute the mean and
        standard deviation of the input raw satellite DataArray (`da`)
        and return a normalised version. If minmax=True,
        instead normalise to lie between min and max of the elements of `array`.

        If min, max, mean, or std are given values other than None,
        those values are used rather than being computed from the training months.

        Returns:
        new_da (xarray.DataArray): Normalised array.

        mean, std (float): Pre-computed mean and standard deviation for the
        normalisation.

        min, max (float): Pre-computed min and max for the normalisation.
        """

        if (min is not None and max is not None) or (mean is not None and std is not None):
            # Function has been passed precomputed normalisation parameters
            pass
        else:
            # Function will be computing new normalisation parameters
            training_samples = da.sel(time=self.obs_train_dates).data
            training_samples = training_samples.ravel()

        if not minmax:
            if mean is None and std is None:
                # Compute mean and std
                mean, std = IceNetDataPreProcessor.mean_and_std(
                    training_samples, self.verbose_level)
            elif mean is not None and std is None:
                # Compute std only
                _, std = IceNetDataPreProcessor.mean_and_std(
                    training_samples, self.verbose_level)
            elif mean is None and std is not None:
                # Compute mean only
                mean, _ = IceNetDataPreProcessor.mean_and_std(
                    training_samples, self.verbose_level)

            new_da = (da - mean) / std

        elif minmax:
            if min is None:
                # Compute min
                min = np.nanmin(training_samples).astype(np.float64)
            if max is None:
                # Compute max
                max = np.nanmax(training_samples).astype(np.float64)

            new_da = (da - min) / (max - min)

        if minmax:
            return new_da, min, max
        elif not minmax:
            return new_da, mean, std

    def save_xarray_in_monthly_averages(self, da, dataset_type, varname, data_format,
                                        model_name=None, member_id=None):

        """
        Saves an xarray DataArray as monthly averaged .npy files using the
        self.paths data structure.

        Parameters:
        da (xarray.DataArray): The DataArray to save.

        dataset_type (str): Either 'obs' or 'transfer' (for CMIP6 data) - the type
        of dataset being saved.

        varname (str): Variable name being saved.

        data_format (str): Either 'abs' or 'anom' - the format of the data
        being saved.
        """

        if self.verbose_level >= 2:
            print('Saving {} {} monthly averages... '.format(data_format, varname), end='', flush=True)

        # Allow for datasets without a time dimension (a single time slice)
        dates = da.time.values
        if hasattr(dates, '__iter__'):
            pass  # Dataset has 'time' dimension; dates already iterable
        else:
            dates = [dates]  # Convert single time value to iterable
            da = da.expand_dims({'time': dates})

        for date in dates:
            slice = da.sel(time=date).data
            date = pd.Timestamp(date)
            year_str = '{:04d}'.format(date.year)
            month_str = '{:02d}'.format(date.month)
            fname = '{}_{}.npy'.format(year_str, month_str)

            if dataset_type == 'obs':
                np.save(os.path.join(self.paths[dataset_type][varname][data_format], fname),
                        slice)

            if dataset_type == 'transfer':
                np.save(os.path.join(self.paths[dataset_type][model_name][member_id][varname][data_format], fname),
                        slice)

        if self.verbose_level >= 2:
            print('Done.')

    def build_linear_trend_da(self, input_da, dataset):

        """
        Construct a DataArray `linea_trend_da` containing the linear trend SIC
        forecasts based on the input DataArray `input_da`.

        `linear_trend_da` will be saved in monthly averages using
        the `save_xarray_in_monthly_averages` method.

        Parameters:
        `input_da` (xarray.DataArray): Input DataArray to produce linear SIC
        forecasts for.

        `dataset` (str): 'obs' or 'cmip6' (dictates whether to skip missing
        observational months in the linear trend extrapolation)

        Returns:
        `linear_trend_da` (xarray.DataArray): DataArray whose time slices
        correspond to the linear trend SIC projection for that month.
        """

        linear_trend_da = input_da.copy(data=np.zeros(input_da.shape, dtype=np.float32))

        # No prediction possible for the first year of data
        forecast_dates = input_da.time.values[12:]

        # Convert from datetime64 to pd.Timestamp
        forecast_dates = [pd.Timestamp(date) for date in forecast_dates]

        # Add on the future year
        last_year = forecast_dates[-12:]
        forecast_dates.extend([date + pd.DateOffset(years=1) for date in last_year])

        linear_trend_da = linear_trend_da.assign_coords({'time': forecast_dates})

        for forecast_date in forecast_dates:
            linear_trend_da.loc[dict(time=forecast_date)] = \
                linear_trend_forecast(forecast_date, self.n_linear_years, da=input_da, dataset=dataset)[0]

        return linear_trend_da

    def check_if_params_precomputed(self, varname, data_format):
        ''' Searches self.norm_params for normalisation parameters
        for a given variable name and data format. '''

        if varname == 'siconca':
            # No normalisation for SIC
            return True

        # Grab existing parameters if stored in norm_params JSON file
        precomputed_params_exists = False
        if varname in self.norm_params.keys():
            if data_format in self.norm_params[varname].keys():
                params = self.norm_params[varname][data_format]
                if self.minmax:
                    if 'min' in params.keys() and 'max' in params.keys():
                        precomputed_params_exists = True
                elif not self.minmax:
                    if 'mean' in params.keys() and 'std' in params.keys():
                        precomputed_params_exists = True

        return precomputed_params_exists

    def save_variable(self, varname, data_format, dates=None):

        """
        Save a normalised 3-dimensional satellite/reanalysis dataset as monthly
        averages (either the absolute values or the monthly anomalies
        computed with xarray).

        This method assumes there is only one variable stored in the NetCDF files.

        Parameters:
        varname (str): Name of the variable to load & save

        data_format (str): 'abs' for absolute values, or 'anom' to compute the
        anomalies, or 'linear_trend' for SIC linear trend projections.

        dates (list of dates): Months to use to compute the monthly
        climatologies (defaults to the months used for training).
        """

        if data_format == 'anom':
            if dates is None:
                dates = self.obs_train_dates

        ########################################################################
        ################# Observational variable
        ########################################################################

        if self.preproc_obs_data:
            if self.verbose_level >= 2:
                print("Preprocessing {} data for {}...  ".format(data_format, varname), end='', flush=True)
                tic = time.time()

            fpath = os.path.join(config.obs_data_folder, '{}_EASE.nc'.format(varname))
            with xr.open_dataset(fpath) as ds:
                da = next(iter(ds.data_vars.values()))

            if data_format == 'anom':

                # Check if climatology already computed
                train_start = self.obs_train_dates[0].strftime('%Y')
                train_end = self.obs_train_dates[-1].strftime('%Y')

                climatology_fpath = os.path.join(
                    config.obs_data_folder,
                    '{}_climatology_{}_{}.nc'.format(varname, train_start, train_end))

                if os.path.exists(climatology_fpath):
                    with xr.open_dataset(climatology_fpath) as ds:
                        climatology = next(iter(ds.data_vars.values()))
                else:
                    climatology = da.sel(time=dates). \
                        groupby("time.month", restore_coord_dims=True).mean("time")
                    climatology.to_netcdf(climatology_fpath)

                da = da.groupby("time.month", restore_coord_dims=True) - climatology

            elif data_format == 'linear_trend':
                da = self.build_linear_trend_da(da, dataset='obs')

            # Realise the array
            da.data = np.asarray(da.data, dtype=np.float32)

            # Normalise the array
            if varname == 'siconca':
                # Don't normalise SIC - already betw 0 and 1
                mean, std = None, None
                min, max = None, None

            elif varname != 'siconca':
                precomputed_params_exists = self.check_if_params_precomputed(varname, data_format)

                if precomputed_params_exists:
                    if self.minmax:
                        min = self.norm_params[varname][data_format]['min']
                        max = self.norm_params[varname][data_format]['max']
                        if self.verbose_level >= 2:
                            print("Using precomputed min/max: {}/{}...  ".format(min, max),
                                  end='', flush=True)
                    elif not self.minmax:
                        mean = self.norm_params[varname][data_format]['mean']
                        std = self.norm_params[varname][data_format]['std']
                        if self.verbose_level >= 2:
                            print("Using precomputed mean/std: {}/{}...  ".format(mean, std),
                                  end='', flush=True)
                elif not precomputed_params_exists:
                    mean, std = None, None
                    min, max = None, None
                    self.norm_params[varname] = {}
                    self.norm_params[varname][data_format] = {}

                if self.minmax:
                    da, min, max = self.normalise_array_using_all_training_months(
                        da, self.minmax, min=min, max=max)
                    if not precomputed_params_exists:
                        if self.verbose_level >= 2:
                            print("Newly computed min/max: {}/{}...  ".format(min, max),
                                  end='', flush=True)
                        self.norm_params[varname][data_format]['min'] = min
                        self.norm_params[varname][data_format]['max'] = max
                elif not self.minmax:
                    da, mean, std = self.normalise_array_using_all_training_months(
                        da, self.minmax, mean=mean, std=std)
                    if not precomputed_params_exists:
                        if self.verbose_level >= 2:
                            print("Newly computed mean/std: {}/{}...  ".format(mean, std),
                                  end='', flush=True)
                        self.norm_params[varname][data_format]['mean'] = mean
                        self.norm_params[varname][data_format]['std'] = std

            da.data[np.isnan(da.data)] = 0.  # Convert any NaNs to zeros

            self.save_xarray_in_monthly_averages(da, 'obs', varname, data_format)

            if self.verbose_level >= 2:
                print("Done in {:.0f}s.\n".format(time.time() - tic))

        ########################################################################
        ################# Transfer variable
        ########################################################################

        if self.preproc_transfer_data:
            if self.verbose_level >= 2:
                print("Preprocessing CMIP6 {} data for {}...  ".format(data_format, varname), end='', flush=True)
                tic = time.time()

            if not self.check_if_params_precomputed(varname, data_format):
                raise ValueError('Normalisation parameters must be computed '
                                 'from observational data before preprocessing '
                                 'CMIP6 data.')

            elif varname != 'siconca' and self.minmax:
                min = self.norm_params[varname][data_format]['min']
                max = self.norm_params[varname][data_format]['max']
                if self.verbose_level >= 2:
                    print("Using precomputed min/max: {}/{}...  ".format(min, max),
                          end='', flush=True)

            elif varname != 'siconca' and not self.minmax:
                mean = self.norm_params[varname][data_format]['mean']
                std = self.norm_params[varname][data_format]['std']
                if self.verbose_level >= 2:
                    print("Using precomputed mean/std: {}/{}...  ".format(mean, std),
                          end='', flush=True)

            for model_name, member_ids in self.cmip_transfer_data.items():
                print('{}: '.format(model_name), end='', flush=True)

                for member_id in member_ids:
                    print('{}, '.format(member_id), end='', flush=True)

                    fname = '{}_EASE_cmpr.nc'.format(varname)
                    fpath = os.path.join(config.cmip6_data_folder, model_name, member_id, fname)

                    with xr.open_dataset(fpath) as ds:
                        da = next(iter(ds.data_vars.values()))

                    # Convert to my month convention of day=1 and time=00:00
                    da = IceNetDataPreProcessor.standardise_cmip6_time_coord(da)

                    # Realise the array
                    da.data = np.asarray(da.data, dtype=np.float32)

                    if data_format == 'anom':

                        climatology = da.sel(time=dates). \
                            groupby("time.month", restore_coord_dims=True).mean("time")
                        da = da.groupby("time.month", restore_coord_dims=True) - climatology

                    elif data_format == 'linear_trend':
                        da = self.build_linear_trend_da(da, dataset='cmip6')

                    # Normalise the array
                    if varname != 'siconca':
                        if self.minmax:
                            da, _, _ = self.normalise_array_using_all_training_months(
                                da, self.minmax, min=min, max=max)
                        elif not self.minmax:
                            da, _, _ = self.normalise_array_using_all_training_months(
                                da, self.minmax, mean=mean, std=std)

                    self.save_xarray_in_monthly_averages(da, 'transfer', varname, data_format,
                                                         model_name, member_id)

            if self.verbose_level >= 2:
                print("Done in {:.0f}s.\n".format(time.time() - tic))

    def preproc_and_save_icenet_data(self):

        '''
        Loop through each variable, preprocessing and saving.
        '''

        for varname, vardict in self.preproc_vars.items():

            if 'metadata' not in vardict.keys():

                for data_format in vardict.keys():

                    if vardict[data_format] is True:

                        self.save_variable(varname, data_format)

            elif 'metadata' in vardict.keys():

                if vardict['include']:
                    if varname == 'land':
                        if self.verbose_level >= 2:
                            print("Setting up the land map: ", end='', flush=True)

                        land_mask = np.load(os.path.join(config.mask_data_folder, config.land_mask_filename))
                        land_map = np.ones(self.config['raw_data_shape'], np.float32)
                        land_map[~land_mask] = -1.

                        np.save(os.path.join(self.paths['meta'], 'land.npy'), land_map)

                        print('\n')

                    elif varname == 'circmonth':
                        if self.verbose_level >= 2:
                            print("Computing circular month values... ", end='', flush=True)
                            tic = time.time()

                        for month in np.arange(1, 13):
                            cos_month = np.cos(2 * np.pi * month / 12, dtype='float32')
                            sin_month = np.sin(2 * np.pi * month / 12, dtype='float32')

                            np.save(os.path.join(self.paths['meta'], 'cos_month_{:02d}.npy'.format(month)), cos_month)
                            np.save(os.path.join(self.paths['meta'], 'sin_month_{:02d}.npy'.format(month)), sin_month)

                        if self.verbose_level >= 2:
                            print("Done in {:.0f}s.\n".format(time.time() - tic))

        with open(self.norm_params_fpath, 'w') as outfile:
            json.dump(self.norm_params, outfile)


class IceNetDataLoader(tf.keras.utils.Sequence):
    """
    Custom data loader class for generating batches of input-output tensors for
    training IceNet. Inherits from  keras.utils.Sequence, which ensures each the
    network trains once on each  sample per epoch. Must implement a __len__
    method that returns the  number of batches and a __getitem__ method that
    returns a batch of data. The  on_epoch_end method is called after each
    epoch.
    See: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

    """

    def __init__(self, dataloader_config_fpath, seed=None):

        '''
        Params:
        dataloader_config_fpath (str): Path to the data loader configuration
            settings JSON file, defining IceNet's input-output data configuration.

        seed (int): Random seed used for shuffling the training samples before
            each epoch.
        '''

        with open(dataloader_config_fpath, 'r') as readfile:
            self.config = json.load(readfile)

        if seed is None:
            self.set_seed(self.config['default_seed'])
        else:
            self.set_seed(seed)

        self.do_transfer_learning = False

        self.set_obs_forecast_IDs(dataset='train')
        self.set_transfer_forecast_IDs()
        self.all_forecast_IDs = self.obs_forecast_IDs
        self.remove_missing_dates()
        self.set_variable_path_formats()
        self.set_number_of_input_channels_for_each_input_variable()
        self.load_polarholes()
        self.determine_tot_num_channels()
        self.on_epoch_end()

        if self.config['verbose_level'] >= 1:
            print("Setup complete.\n")

    def set_obs_forecast_IDs(self, dataset='train'):
        """
        Build up a list of forecast initialisation dates for the train, val, or
        test sets based on the configuration JSON file start & end points for
        each dataset.
        """

        forecast_start_date_ends = self.config['sample_IDs']['obs_{}_dates'.format(dataset)]

        if forecast_start_date_ends is not None:

            # Convert to Pandas Timestamps
            forecast_start_date_ends = [
                pd.Timestamp(date).to_pydatetime() for date in forecast_start_date_ends
            ]

            self.obs_forecast_IDs = list(pd.date_range(
                forecast_start_date_ends[0],
                forecast_start_date_ends[1],
                freq='MS',
                closed='right',
            ))

    def set_transfer_forecast_IDs(self):

        '''
        Use self.cmip6_transfer_train_dict to set up a list array of
        3-tuples of the form:
            (cmip6_model_name, member_id, forecast_start_date)

        This list is used as IDs into the transfer data hierarchy
        to train on all cmip6 models and and their runs simultaneously.
        '''

        self.transfer_forecast_IDs = []
        for cmip6_model_name, member_id_dict in self.config['cmip6_run_dict'].items():
            for member_id, (start_date, end_date) in member_id_dict.items():

                member_id_dates = list(pd.date_range(
                    start_date,
                    end_date,
                    freq='MS',
                    closed='right',
                ))

                self.transfer_forecast_IDs.extend(
                    itertools.product([cmip6_model_name], [member_id], member_id_dates)
                )

    def set_variable_path_formats(self):

        """
        Initialise the paths to the .npy files of each variable based on
        `self.config['input_data']`.
        """

        if self.config['verbose_level'] >= 1:
            print('Setting up the variable paths for {}... '.format(self.config['dataset_name']),
                  end='', flush=True)

        # Parent folder for this dataset
        self.dataset_path = os.path.join(config.network_dataset_folder, self.config['dataset_name'])

        # Dictionary data structure to store image variable paths
        self.variable_paths = {}

        for varname, vardict in self.config['input_data'].items():

            if 'metadata' not in vardict.keys():
                self.variable_paths[varname] = {}

                for data_format in vardict.keys():

                    if vardict[data_format]['include'] is True:

                        if not self.do_transfer_learning:
                            path = os.path.join(
                                self.dataset_path, 'obs',
                                varname, data_format, '{:04d}_{:02d}.npy'
                            )
                        elif self.do_transfer_learning:
                            path = os.path.join(
                                self.dataset_path, 'transfer', '{}', '{}',
                                varname, data_format, '{:04d}_{:02d}.npy'
                            )

                        self.variable_paths[varname][data_format] = path

            elif 'metadata' in vardict.keys():

                if vardict['include'] is True:

                    if varname == 'land':
                        path = os.path.join(self.dataset_path, 'meta', 'land.npy')
                        self.variable_paths['land'] = path

                    elif varname == 'circmonth':
                        path = os.path.join(self.dataset_path, 'meta',
                                            '{}_month_{:02d}.npy')
                        self.variable_paths['circmonth'] = path

        if self.config['verbose_level'] >= 1:
            print('Done.')

    def set_seed(self, seed):
        """
        Set the seed used by the random generator (used to randomly shuffle
        the ordering of training samples after each epoch).
        """
        if self.config['verbose_level'] >= 1:
            print("Setting the data generator's random seed to {}".format(seed))
        self.rng = np.random.default_rng(seed)

    def determine_variable_names(self):
        """
        Set up a list of strings for the names of each input variable (in the
        correct order) by looping over the `input_data` dictionary.
        """
        variable_names = []

        for varname, vardict in self.config['input_data'].items():
            # Input variables that span time
            if 'metadata' not in vardict.keys():
                for data_format in vardict.keys():
                    if vardict[data_format]['include']:
                        if data_format != 'linear_trend':
                            for lag in np.arange(1, vardict[data_format]['max_lag']+1):
                                variable_names.append(varname+'_{}_{}'.format(data_format, lag))
                        elif data_format == 'linear_trend':
                            for leadtime in np.arange(1, self.config['n_forecast_months']+1):
                                variable_names.append(varname+'_{}_{}'.format(data_format, leadtime))

            # Metadata input variables that don't span time
            elif 'metadata' in vardict.keys() and vardict['include']:
                if varname == 'land':
                    variable_names.append(varname)

                elif varname == 'circmonth':
                    variable_names.append('cos(month)')
                    variable_names.append('sin(month)')

        return variable_names

    def set_number_of_input_channels_for_each_input_variable(self):
        """
        Build up the dict `self.num_input_channels_dict` to store the number of input
        channels spanned by each input variable.
        """

        if self.config['verbose_level'] >= 1:
            print("Setting the number of input months for each input variable.")

        self.num_input_channels_dict = {}

        for varname, vardict in self.config['input_data'].items():
            if 'metadata' not in vardict.keys():
                # Variables that span time
                for data_format in vardict.keys():
                    if vardict[data_format]['include']:
                        varname_format = varname+'_{}'.format(data_format)
                        if data_format != 'linear_trend':
                            self.num_input_channels_dict[varname_format] = vardict[data_format]['max_lag']
                        elif data_format == 'linear_trend':
                            self.num_input_channels_dict[varname_format] = self.config['n_forecast_months']

            # Metadata input variables that don't span time
            elif 'metadata' in vardict.keys() and vardict['include']:
                if varname == 'land':
                    self.num_input_channels_dict[varname] = 1

                if varname == 'circmonth':
                    self.num_input_channels_dict[varname] = 2

    def determine_tot_num_channels(self):
        """
        Determine the number of channels for the input 3D volumes.
        """

        self.tot_num_channels = 0
        for varname, num_channels in self.num_input_channels_dict.items():
            self.tot_num_channels += num_channels

    def all_sic_input_dates_from_forecast_start_date(self, forecast_start_date):
        """
        Return a list of all the SIC dates used as input for a particular forecast
        date, based on the "max_lag" options of self.config['input_data'].
        """

        # Find all SIC lags
        max_lags = []
        if self.config['input_data']['siconca']['abs']['include']:
            max_lags.append(self.config['input_data']['siconca']['abs']['max_lag'])
        if self.config['input_data']['siconca']['anom']['include']:
            max_lags.append(self.config['input_data']['siconca']['anom']['max_lag'])
        max_lag = np.max(max_lags)

        input_dates = [
            forecast_start_date - pd.DateOffset(months=int(lag)) for lag in np.arange(1, max_lag+1)
        ]

        return input_dates

    def check_for_missing_date_dependence(self, forecast_start_date):
        """
        Check a forecast ID and return a bool for whether any of the input SIC maps
        are missing. Used to remove forecast IDs that depend on missing SIC data.

        Note: If one of the _forecast_ dates are missing but not _input_ dates,
        the sample weight matrix for that date will be all zeroes so that the
        samples for that date do not appear in the loss function.
        """
        contains_missing_date = False

        # Check SIC input dates
        input_dates = self.all_sic_input_dates_from_forecast_start_date(forecast_start_date)

        for input_date in input_dates:
            if any([input_date == missing_date for missing_date in config.missing_dates]):
                contains_missing_date = True
                break

        return contains_missing_date

    def remove_missing_dates(self):

        '''
        Remove dates from self.obs_forecast_IDs that depend on a missing
        observation of SIC.
        '''

        if self.config['verbose_level'] >= 2:
            print('Checking forecast start dates for missing SIC dates... ', end='', flush=True)

        new_obs_forecast_IDs = []
        for forecast_start_date in self.obs_forecast_IDs:
            if self.check_for_missing_date_dependence(forecast_start_date):
                if self.config['verbose_level'] >= 3:
                    print('Removing {}, '.format(
                        forecast_start_date.strftime('%Y_%m_%d')), end='', flush=True)

            else:
                new_obs_forecast_IDs.append(forecast_start_date)

        self.obs_forecast_IDs = new_obs_forecast_IDs

    def load_polarholes(self):
        """
        Loads each of the polar holes.
        """

        if self.config['verbose_level'] >= 1:
            tic = time.time()
            print("Loading and augmenting the polar holes... ", end='', flush=True)

        polarhole_path = os.path.join(config.mask_data_folder, config.polarhole1_fname)
        self.polarhole1_mask = np.load(polarhole_path)

        polarhole_path = os.path.join(config.mask_data_folder, config.polarhole2_fname)
        self.polarhole2_mask = np.load(polarhole_path)

        if config.use_polarhole3:
            polarhole_path = os.path.join(config.mask_data_folder, config.polarhole3_fname)
            self.polarhole3_mask = np.load(polarhole_path)

        self.nopolarhole_mask = np.full((432, 432), False)

        if self.config['verbose_level'] >= 1:
            print("Done in {:.0f}s.\n".format(time.time() - tic))

    def determine_polar_hole_mask(self, forecast_start_date):
        """
        Determine which polar hole mask to use (if any) by finding the oldest SIC
        input month based on the current output month. The polar hole active for
        the oldest input month is used (because the polar hole size decreases
        monotonically over time, and we wish to use the largest polar hole for
        the input data).

        Parameters:
        forecast_start_date (pd.Timestamp): Timepoint for the forecast initialialisation.

        Returns:
        polarhole_mask: Mask array with NaNs on polar hole grid cells and 1s
        elsewhere.
        """

        oldest_input_date = min(self.all_sic_input_dates_from_forecast_start_date(forecast_start_date))

        if oldest_input_date <= config.polarhole1_final_date:
            polarhole_mask = self.polarhole1_mask
            if self.config['verbose_level'] >= 3:
                print("Forecast start date: {}, polar hole: {}".format(
                    forecast_start_date.strftime("%Y_%m"), 1))

        elif oldest_input_date <= config.polarhole2_final_date:
            polarhole_mask = self.polarhole2_mask
            if self.config['verbose_level'] >= 3:
                print("Forecast start date: {}, polar hole: {}".format(
                    forecast_start_date.strftime("%Y_%m"), 2))

        else:
            polarhole_mask = self.nopolarhole_mask
            if self.config['verbose_level'] >= 3:
                print("Forecast start date: {}, polar hole: {}".format(
                    forecast_start_date.strftime("%Y_%m"), "none"))

        return polarhole_mask

    def determine_active_grid_cell_mask(self, forecast_date):
        """
        Determine which active grid cell mask to use (a boolean array with
        True on active cells and False on inactive cells). The cells with 'True'
        are where predictions are to be made with IceNet. The active grid cell
        mask for a particular month is determined by the sum of the land cells,
        the ocean cells (for that month), and the missing polar hole.

        The mask is used for removing 'inactive' cells (such as land or polar
        hole cells) from the loss function in self.data_generation.
        """

        output_month_str = '{:02d}'.format(forecast_date.month)
        output_active_grid_cell_mask_fname = config.active_grid_cell_file_format. \
            format(output_month_str)
        output_active_grid_cell_mask_path = os.path.join(
            config.mask_data_folder, output_active_grid_cell_mask_fname)
        output_active_grid_cell_mask = np.load(output_active_grid_cell_mask_path)

        # Only use the polar hole mask if predicting observational data
        if not self.do_transfer_learning:
            polarhole_mask = self.determine_polar_hole_mask(forecast_date)

            # Add the polar hole mask to that land/ocean mask for the current month
            output_active_grid_cell_mask[polarhole_mask] = False

        return output_active_grid_cell_mask

    def turn_on_transfer_learning(self):

        '''
        Converts the data loader to use CMIP6 pre-training data
        for transfer learning.
        '''

        self.do_transfer_learning = True
        self.all_forecast_IDs = self.transfer_forecast_IDs
        self.on_epoch_end()  # Shuffle transfer training indexes
        self.set_variable_path_formats()

    def turn_off_transfer_learning(self):

        '''
        Converts the data loader back to using ERA5/OSI-SAF observational
        training data.
        '''

        self.do_transfer_learning = False
        self.all_forecast_IDs = self.obs_forecast_IDs
        self.on_epoch_end()  # Shuffle transfer training indexes
        self.set_variable_path_formats()

    def convert_to_validation_data_loader(self):

        """
        Resets the `all_forecast_IDs` array to correspond to the validation
        months defined by the data loader configuration file.
        """

        self.set_obs_forecast_IDs(dataset='val')
        self.remove_missing_dates()
        self.all_forecast_IDs = self.obs_forecast_IDs

    def convert_to_test_data_loader(self):

        """
        As above but for the testing months.
        """

        self.set_obs_forecast_IDs(dataset='test')
        self.remove_missing_dates()
        self.all_forecast_IDs = self.obs_forecast_IDs

    def data_generation(self, forecast_IDs):
        """
        Generate input-output data for IceNet for a given forecast ID.

        Parameters:
        forecast_IDs (list):
            If self.do_transfer_learning is False, a list of pd.Timestamp objects
            corresponding to the forecast initialisation dates (first month
            being forecast) for the batch of X-y data to load.

            If self.do_transfer_learning is True, a list of tuples
            of the form (cmip6_model_name, member_id, forecast_start_date).

        Returns:
        X (ndarray): Batch of input 3D volumes.

        y (ndarray): Batch of ground truth output SIC class maps

        sample_weight (ndarray): Batch of pixelwise weights for weighting the
            loss function (masking outside the active grid cell region and
            up-weighting summer months).
        """

        # Allow non-list input for single forecasts
        forecast_IDs = pd.Timestamp(forecast_IDs) if isinstance(forecast_IDs, str) else forecast_IDs
        forecast_IDs = [forecast_IDs] if not isinstance(forecast_IDs, list) else forecast_IDs

        current_batch_size = len(forecast_IDs)

        if self.do_transfer_learning:
            cmip6_model_names = [forecast_ID[0] for forecast_ID in forecast_IDs]
            cmip6_member_ids = [forecast_ID[1] for forecast_ID in forecast_IDs]
            forecast_start_dates = [forecast_ID[2] for forecast_ID in forecast_IDs]
        else:
            forecast_start_dates = forecast_IDs

        ########################################################################
        # OUTPUT LABELS
        ########################################################################

        # Build up the set of N_samps output SIC time-series
        #   (each n_forecast_months long in the time dimension)

        # To become array of shape (N_samps, *raw_data_shape, n_forecast_months)
        batch_sic_list = []

        # True = forecasts months corresponding to no data
        missing_month_dict = {}

        for sample_idx, forecast_date in enumerate(forecast_start_dates):

            # To become array of shape (*raw_data_shape, n_forecast_months)
            sample_sic_list = []

            # List of forecast indexes with missing data
            missing_month_dict[sample_idx] = []

            for forecast_leadtime_idx in range(self.config['n_forecast_months']):

                forecast_date = forecast_start_dates[sample_idx] + pd.DateOffset(months=forecast_leadtime_idx)

                if self.do_transfer_learning:
                    sample_sic_list.append(
                        np.load(self.variable_paths['siconca']['abs'].format(
                            cmip6_model_names[sample_idx], cmip6_member_ids[sample_idx],
                            forecast_date.year, forecast_date.month))
                    )

                elif not self.do_transfer_learning:
                    if any([forecast_date == missing_date for missing_date in config.missing_dates]):
                        # Output file does not exist
                        sample_sic_list.append(np.zeros(self.config['raw_data_shape']))

                    else:
                        fpath = self.variable_paths['siconca']['abs'].format(
                            forecast_date.year, forecast_date.month)
                        if os.path.exists(fpath):
                            sample_sic_list.append(np.load(fpath))
                        else:
                            # Ground truth data doesn't exist: fill with NaNs
                            sample_sic_list.append(
                                np.full(self.config['raw_data_shape'], np.nan, dtype=np.float32))

            batch_sic_list.append(np.stack(sample_sic_list, axis=2))

        batch_sic = np.stack(batch_sic_list, axis=0)

        no_ice_gridcells = batch_sic <= 0.15
        ice_gridcells = batch_sic >= 0.80
        marginal_ice_gridcells = ~((no_ice_gridcells) | (ice_gridcells))

        # Categorical representation with channel dimension for class probs
        y = np.zeros((
            current_batch_size,
            *self.config['raw_data_shape'],
            self.config['n_forecast_months'],
            3
        ), dtype=np.float32)

        y[no_ice_gridcells, 0] = 1
        y[marginal_ice_gridcells, 1] = 1
        y[ice_gridcells, 2] = 1

        # Move lead time to final axis
        y = np.moveaxis(y, source=3, destination=4)

        # Missing months
        for sample_idx, forecast_leadtime_idx_list in missing_month_dict.items():
            if len(forecast_leadtime_idx_list) > 0:
                y[sample_idx, :, :, :, forecast_leadtime_idx_list] = 0

        ########################################################################
        # PIXELWISE LOSS FUNCTION WEIGHTING
        ########################################################################

        sample_weight = np.zeros((
            current_batch_size,
            *self.config['raw_data_shape'],
            1,  # Broadcastable class dimension
            self.config['n_forecast_months']
        ), dtype=np.float32)
        for sample_idx, forecast_date in enumerate(forecast_start_dates):

            for forecast_leadtime_idx in range(self.config['n_forecast_months']):

                forecast_date = forecast_start_dates[sample_idx] + pd.DateOffset(months=forecast_leadtime_idx)

                if any([forecast_date == missing_date for missing_date in config.missing_dates]):
                    # Leave sample weighting as all-zeros
                    pass

                else:
                    # Zero loss outside of 'active grid cells'
                    sample_weight_ij = self.determine_active_grid_cell_mask(forecast_date)
                    sample_weight_ij = sample_weight_ij.astype(np.float32)

                    # Scale the loss for each month s.t. March is
                    #   scaled by 1 and Sept is scaled by 1.77
                    if self.config['loss_weight_months']:
                        sample_weight_ij *= 33928. / np.sum(sample_weight_ij)

                    sample_weight[sample_idx, :, :, 0, forecast_leadtime_idx] = \
                        sample_weight_ij

        ########################################################################
        # INPUT FEATURES
        ########################################################################

        # Batch tensor
        X = np.zeros((
            current_batch_size,
            *self.config['raw_data_shape'],
            self.tot_num_channels
        ), dtype=np.float32)

        # Build up the batch of inputs
        for sample_idx, forecast_start_date in enumerate(forecast_start_dates):

            present_date = forecast_start_date - relativedelta(months=1)

            # Initialise variable indexes used to fill the input tensor `X`
            variable_idx1 = 0
            variable_idx2 = 0

            for varname, vardict in self.config['input_data'].items():

                if 'metadata' not in vardict.keys():

                    for data_format in vardict.keys():

                        if vardict[data_format]['include']:

                            varname_format = '{}_{}'.format(varname, data_format)

                            if data_format != 'linear_trend':
                                lbs = range(vardict[data_format]['max_lag'])
                                input_months = [present_date - relativedelta(months=lb) for lb in lbs]
                            elif data_format == 'linear_trend':
                                input_months = [present_date + relativedelta(months=forecast_leadtime)
                                                for forecast_leadtime in np.arange(1, self.config['n_forecast_months']+1)]

                            variable_idx2 += self.num_input_channels_dict[varname_format]

                            if not self.do_transfer_learning:
                                X[sample_idx, :, :, variable_idx1:variable_idx2] = \
                                    np.stack([np.load(self.variable_paths[varname][data_format].format(
                                              date.year, date.month))
                                              for date in input_months], axis=-1)
                            elif self.do_transfer_learning:
                                cmip6_model_name = cmip6_model_names[sample_idx]
                                cmip6_member_id = cmip6_member_ids[sample_idx]

                                X[sample_idx, :, :, variable_idx1:variable_idx2] = \
                                    np.stack([np.load(self.variable_paths[varname][data_format].format(
                                              cmip6_model_name, cmip6_member_id, date.year, date.month))
                                              for date in input_months], axis=-1)

                            variable_idx1 += self.num_input_channels_dict[varname_format]

                elif 'metadata' in vardict.keys() and vardict['include']:

                    variable_idx2 += self.num_input_channels_dict[varname]

                    if varname == 'land':
                        X[sample_idx, :, :, variable_idx1] = np.load(self.variable_paths['land'])

                    elif varname == 'circmonth':
                        X[sample_idx, :, :, variable_idx1] = \
                            np.load(self.variable_paths['circmonth'].format('cos', forecast_start_date.month))
                        X[sample_idx, :, :, variable_idx1 + 1] = \
                            np.load(self.variable_paths['circmonth'].format('sin', forecast_start_date.month))

                    variable_idx1 += self.num_input_channels_dict[varname]

        return X, y, sample_weight

    def __getitem__(self, batch_idx):
        '''
        Generate one batch of data of size `batch_size` at batch index `batch_idx`
        into the set of batches in the epoch.
        '''

        batch_start = batch_idx * self.config['batch_size']
        batch_end = np.min([(batch_idx + 1) * self.config['batch_size'], len(self.all_forecast_IDs)])

        sample_idxs = np.arange(batch_start, batch_end)
        batch_IDs = [self.all_forecast_IDs[sample_idx] for sample_idx in sample_idxs]

        return self.data_generation(batch_IDs)

    def __len__(self):
        ''' Returns the number of batches per training epoch. '''
        return int(np.ceil(len(self.all_forecast_IDs) / self.config['batch_size']))

    def on_epoch_end(self):
        """ Randomly shuffles training samples after each epoch. """

        if self.config['verbose_level'] >= 2:
            print("on_epoch_end called")

        # Randomly shuffle the forecast IDs in-place
        self.rng.shuffle(self.all_forecast_IDs)


################## MISC FUNCTIONS
################################################################################


def create_results_dataset_index(model_compute_list, leadtimes,
                                 all_target_dates, icenet_ID,
                                 icenet_seeds):

    '''
    Returns a pandas.MultiIndex object of results dataset indexes for a
    given list of models to compute metrics for. For IceNet, the 'Ensemble
    member' column delineates the performance of each IceNet ensemble
    member (identified by the integer random seed value it was trained
    with) and the ensemble mean models ('ensemble' or 'ensemble_tempscaled').
    '''

    multi_index = pd.MultiIndex.from_product(
        [model_compute_list, leadtimes, all_target_dates])

    idxs = []
    for row in multi_index:
        model = row[0]
        row = [[item] for item in row]
        if model == icenet_ID:
            idxs.extend(list(itertools.product(*row, icenet_seeds)))
        else:
            idxs.extend(list(itertools.product(*row, ['NA'])))

    multi_index = pd.MultiIndex.from_tuples(
        idxs, names=['Model', 'Leadtime', 'Forecast date', 'Ensemble member']).\
        reorder_levels(['Model', 'Ensemble member', 'Leadtime', 'Forecast date'])

    return multi_index


def make_varname_verbose(varname, leadtime, fc_month_idx):

    '''
    Takes IceNet short variable name (e.g. siconca_abs_3) and converts it to a
    long name for a given forecast calendar month and lead time (e.g.
    'Feb SIC').

    Inputs:
    varname: Short variable name.
    leadtime: Lead time of the forecast.
    fc_month_index: Mod-12 calendar month index for the month being forecast
        (e.g. 8 for September)

    Returns:
    verbose_varname: Long variable name.
    '''

    month_names = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])

    varname_regex = re.compile('^(.*)_(abs|anom|linear_trend)_([0-9]+)$')

    var_lookup_table = {
        'siconca': 'SIC',
        'tas': '2m air temperature',
        'ta500': '500 hPa air temperature',
        'tos': 'sea surface temperature',
        'rsds': 'downwelling solar radiation',
        'rsus': 'upwelling solar radiation',
        'psl': 'sea level pressure',
        'zg500': '500 hPa geopotential height',
        'zg250': '250 hPa geopotential height',
        'ua10': '10 hPa zonal wind speed',
        'uas': 'x-direction wind',
        'vas': 'y-direction wind'
    }

    initialisation_month_idx = (fc_month_idx - leadtime) % 12

    varname_match = varname_regex.match(varname)

    field = varname_match[1]
    data_format = varname_match[2]
    lead_or_lag = int(varname_match[3])

    verbose_varname = ''

    month_suffix = ' '
    month_prefix = ''
    if data_format != 'linear_trend':
        # Read back from initialisation month to get input lag month
        lag = lead_or_lag  # In no of months
        input_month_name = month_names[(initialisation_month_idx - lag + 1) % 12]

        if (initialisation_month_idx - lag + 1) // 12 == -1:
            # Previous calendar year
            month_prefix = 'Previous '

    elif data_format == 'linear_trend':
        # Read forward from initialisation month to get linear trend forecast month
        lead = lead_or_lag  # In no of months
        input_month_name = month_names[(initialisation_month_idx + lead) % 12]

        if (initialisation_month_idx + lead) // 12 == 1:
            # Next calendar year
            month_prefix = 'Next '

    # Month the input corresponds to
    verbose_varname += month_prefix + input_month_name + month_suffix

    # verbose variable name
    if data_format != 'linear_trend':
        verbose_varname += var_lookup_table[field]
        if data_format == 'anom':
            verbose_varname += ' anomaly'
    elif data_format == 'linear_trend':
        verbose_varname += 'linear trend SIC forecast'

    return verbose_varname


def make_varname_verbose_any_leadtime(varname):

    ''' As above, but agnostic to what the target month or lead time is. E.g.
    "SIC (1)" for sea ice concentration at a lag of 1 month. '''

    varname_regex = re.compile('^(.*)_(abs|anom|linear_trend)_([0-9]+)$')

    var_lookup_table = {
        'siconca': 'SIC',
        'tas': '2m air temperature',
        'ta500': '500 hPa air temperature',
        'tos': 'sea surface temperature',
        'rsds': 'downwelling solar radiation',
        'rsus': 'upwelling solar radiation',
        'psl': 'sea level pressure',
        'zg500': '500 hPa geopotential height',
        'zg250': '250 hPa geopotential height',
        'ua10': '10 hPa zonal wind speed',
        'uas': 'x-direction wind',
        'vas': 'y-direction wind',
        'land': 'land mask',
        'cos(month)': 'cos(init month)',
        'sin(month)': 'sin(init month)',
    }

    exception_vars = ['cos(month)', 'sin(month)', 'land']

    if varname in exception_vars:
        return var_lookup_table[varname]
    else:
        varname_match = varname_regex.match(varname)

        field = varname_match[1]
        data_format = varname_match[2]
        lead_or_lag = int(varname_match[3])

        # verbose variable name
        if data_format != 'linear_trend':
            verbose_varname = var_lookup_table[field]
            if data_format == 'anom':
                verbose_varname += ' anomaly'
        elif data_format == 'linear_trend':
            verbose_varname = 'Linear trend SIC forecast'

        verbose_varname += ' ({:.0f})'.format(lead_or_lag)

        return verbose_varname


################################################################################
################## FUNCTIONS
################################################################################


def assignLatLonCoordSystem(cube):
    ''' Assign coordinate system to iris cube to allow regridding. '''

    cube.coord('latitude').coord_system = iris.coord_systems.GeogCS(6367470.0)
    cube.coord('longitude').coord_system = iris.coord_systems.GeogCS(6367470.0)

    return cube


def fix_near_real_time_era5_func(latlon_path):

    '''
    Near-real-time ERA5 data is classed as a different dataset called 'ERA5T'.
    This results in a spurious 'expver' dimension. This method detects
    whether that dim is present and removes it, concatenating into one array
    '''

    ds = xr.open_dataarray(latlon_path)

    if len(ds.data.shape) == 4:
        print('Fixing spurious ERA5 "expver dimension for {}.'.format(latlon_path))

        arr = xr.open_dataarray(latlon_path).data
        arr = ds.data
        # Expver 1 (ERA5)
        era5_months = ~np.isnan(arr[:, 0, :, :]).all(axis=(1, 2))

        # Expver 2 (ERA5T - near real time)
        era5t_months = ~np.isnan(arr[:, 1, :, :]).all(axis=(1, 2))

        ds = xr.concat((ds[era5_months, 0, :], ds[era5t_months, 1, :]), dim='time')

        ds = ds.reset_coords('expver', drop=True)

        os.remove(latlon_path)
        ds.load().to_netcdf(latlon_path)


###############################################################################
############### LEARNING RATE SCHEDULER
###############################################################################


def make_exp_decay_lr_schedule(rate, start_epoch=1, end_epoch=np.inf, verbose=False):

    ''' Returns an exponential learning rate function that multiplies by
    exp(-rate) each epoch after `start_epoch`. '''

    def lr_scheduler_exp_decay(epoch, lr):
        ''' Learning rate scheduler for fine tuning.
        Exponential decrease after start_epoch until end_epoch. '''

        if epoch >= start_epoch and epoch < end_epoch:
            lr = lr * np.math.exp(-rate)

        if verbose:
            print('\nSetting learning rate to: {}\n'.format(lr))

        return lr

    return lr_scheduler_exp_decay


###############################################################################
############### REGRIDDING VECTOR DATA
###############################################################################


def rotate_grid_vectors(u_cube, v_cube, angles):
    """
    Author: Tony Phillips (BAS)

    Wrapper for :func:`~iris.analysis.cartography.rotate_grid_vectors`
    that can rotate multiple masked spatial fields in one go by iterating
    over the horizontal spatial axes in slices
    """
    # lists to hold slices of rotated vectors
    u_r_all = iris.cube.CubeList()
    v_r_all = iris.cube.CubeList()

    # get the X and Y dimension coordinates for each source cube
    u_xy_coords = [u_cube.coord(axis='x', dim_coords=True),
                   u_cube.coord(axis='y', dim_coords=True)]
    v_xy_coords = [v_cube.coord(axis='x', dim_coords=True),
                   v_cube.coord(axis='y', dim_coords=True)]

    # iterate over X, Y slices of the source cubes, rotating each in turn
    for u, v in zip(u_cube.slices(u_xy_coords, ordered=False),
                    v_cube.slices(v_xy_coords, ordered=False)):
        u_r, v_r = iris.analysis.cartography.rotate_grid_vectors(u, v, angles)
        u_r_all.append(u_r)
        v_r_all.append(v_r)

    # return the slices, merged back together into a pair of cubes
    return (u_r_all.merge_cube(), v_r_all.merge_cube())


def gridcell_angles_from_dim_coords(cube):
    """
    Author: Tony Phillips (BAS)

    Wrapper for :func:`~iris.analysis.cartography.gridcell_angles`
    that derives the 2D X and Y lon/lat coordinates from 1D X and Y
    coordinates identifiable as 'x' and 'y' axes

    The provided cube must have a coordinate system so that its
    X and Y coordinate bounds (which are derived if necessary)
    can be converted to lons and lats
    """

    # get the X and Y dimension coordinates for the cube
    x_coord = cube.coord(axis='x', dim_coords=True)
    y_coord = cube.coord(axis='y', dim_coords=True)

    # add bounds if necessary
    if not x_coord.has_bounds():
        x_coord = x_coord.copy()
        x_coord.guess_bounds()
    if not y_coord.has_bounds():
        y_coord = y_coord.copy()
        y_coord.guess_bounds()

    # get the grid cell bounds
    x_bounds = x_coord.bounds
    y_bounds = y_coord.bounds
    nx = x_bounds.shape[0]
    ny = y_bounds.shape[0]

    # make arrays to hold the ordered X and Y bound coordinates
    x = np.zeros((ny, nx, 4))
    y = np.zeros((ny, nx, 4))

    # iterate over the bounds (in order BL, BR, TL, TR), mesh them and
    # put them into the X and Y bound coordinates (in order BL, BR, TR, TL)
    c = [0, 1, 3, 2]
    cind = 0
    for yi in [0, 1]:
        for xi in [0, 1]:
            xy = np.meshgrid(x_bounds[:, xi], y_bounds[:, yi])
            x[:,:,c[cind]] = xy[0]
            y[:,:,c[cind]] = xy[1]
            cind += 1

    # convert the X and Y coordinates to longitudes and latitudes
    source_crs = cube.coord_system().as_cartopy_crs()
    target_crs = ccrs.PlateCarree()
    pts = target_crs.transform_points(source_crs, x.flatten(), y.flatten())
    lons = pts[:, 0].reshape(x.shape)
    lats = pts[:, 1].reshape(x.shape)

    # get the angles
    angles = iris.analysis.cartography.gridcell_angles(lons, lats)

    # add the X and Y dimension coordinates from the cube to the angles cube
    angles.add_dim_coord(y_coord, 0)
    angles.add_dim_coord(x_coord, 1)

    # if the cube's X dimension preceeds its Y dimension
    # transpose the angles to match
    if cube.coord_dims(x_coord)[0] < cube.coord_dims(y_coord)[0]:
        angles.transpose()

    return angles


def invert_gridcell_angles(angles):
    """
    Author: Tony Phillips (BAS)

    Negate a cube of gridcell angles in place, transforming
    gridcell_angle_from_true_east <--> true_east_from_gridcell_angle
    """
    angles.data *= -1

    names = ['true_east_from_gridcell_angle', 'gridcell_angle_from_true_east']
    name = angles.name()
    if name in names:
        angles.rename(names[1 - names.index(name)])


###############################################################################
############### CMIP6
###############################################################################


# Below taken from https://hub.binder.pangeo.io/user/pangeo-data-pan--cmip6-examples-ro965nih/lab
def esgf_search(server="https://esgf-node.llnl.gov/esg-search/search",
                files_type="OPENDAP", local_node=False, latest=True, project="CMIP6",
                verbose1=False, verbose2=False, format="application%2Fsolr%2Bjson",
                use_csrf=False, **search):
    client = requests.session()
    payload = search
    payload["project"] = project
    payload["type"]= "File"
    if latest:
        payload["latest"] = "true"
    if local_node:
        payload["distrib"] = "false"
    if use_csrf:
        client.get(server)
        if 'csrftoken' in client.cookies:
            # Django 1.6 and up
            csrftoken = client.cookies['csrftoken']
        else:
            # older versions
            csrftoken = client.cookies['csrf']
        payload["csrfmiddlewaretoken"] = csrftoken

    payload["format"] = format

    offset = 0
    numFound = 10000
    all_files = []
    files_type = files_type.upper()
    while offset < numFound:
        payload["offset"] = offset
        url_keys = []
        for k in payload:
            url_keys += ["{}={}".format(k, payload[k])]

        url = "{}/?{}".format(server, "&".join(url_keys))
        if verbose1:
            print(url)
        r = client.get(url)
        r.raise_for_status()
        resp = r.json()["response"]
        numFound = int(resp["numFound"])
        resp = resp["docs"]
        offset += len(resp)
        for d in resp:
            if verbose2:
                for k in d:
                    print("{}: {}".format(k,d[k]))
            url = d["url"]
            for f in d["url"]:
                sp = f.split("|")
                if sp[-1] == files_type:
                    all_files.append(sp[0].split(".html")[0])
    return sorted(all_files)


def regrid_cmip6(cmip6_cube, grid_cube, verbose=False):

    if verbose:
        tic = time.time()
        print("regridding... ", end='', flush=True)

    cs = grid_cube.coord_system().ellipsoid

    for coord in ['longitude', 'latitude']:
        cmip6_cube.coord(coord).coord_system = cs

    cmip6_ease = cmip6_cube.regrid(grid_cube, iris.analysis.Linear())

    if verbose:
        dur = time.time() - tic
        print("done in {}m:{:.0f}s... ".format(np.floor(dur / 60), dur % 60), end='', flush=True)

    return cmip6_ease


def save_cmip6(cmip6_ease, fpath, compress=True, verbose=False):
    tic = time.time()

    if compress:
        if verbose:
            print('compressing & saving... ', end='', flush=True)
        iris.fileformats.netcdf.save(cmip6_ease, fpath, complevel=7, zlib=True)
    else:
        if verbose:
            print('saving uncompressed... ', end='', flush=True)
        iris.save(cmip6_ease, fpath)

    if verbose:
        dur = time.time() - tic
        print("done in {}m:{:.0f}s... ".format(np.floor(dur / 60), dur % 60), end='', flush=True)


###############################################################################
############### PLOTTING
###############################################################################


def compute_heatmap(results_df, model, seed='NA', metric='Binary accuracy'):
    '''
    Returns a binary accuracy heatmap of lead time vs. calendar month
    for a given model.
    '''

    month_names = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])

    # Mean over calendar month
    mean_df = results_df.loc[model, seed].reset_index().\
        groupby(['Calendar month', 'Leadtime']).mean()

    # Pivot
    heatmap_df = mean_df.reset_index().\
        pivot('Calendar month', 'Leadtime', metric).reindex(month_names)

    return heatmap_df


def arr_to_ice_edge_arr(arr, thresh, land_mask, region_mask):

    '''
    Compute a boolean mask with True over ice edge contour grid cells using
    matplotlib.pyplot.contour and an input threshold to define the ice edge
    (e.g. 0.15 for the 15% SIC ice edge or 0.5 for SIP forecasts). The contour
    along the coastline is removed using the region mask.
    '''

    X, Y = np.meshgrid(np.arange(arr.shape[0]), np.arange(arr.shape[1]))
    X = X.T
    Y = Y.T

    cs = plt.contour(X, Y, arr, [thresh], alpha=0)  # Do not plot on any axes
    x = []
    y = []
    for p in cs.collections[0].get_paths():
        x_i, y_i = p.vertices.T
        x.extend(np.round(x_i))
        y.extend(np.round(y_i))
    x = np.array(x, int)
    y = np.array(y, int)
    ice_edge_arr = np.zeros(arr.shape, dtype=bool)
    ice_edge_arr[x, y] = True
    # Mask out ice edge contour that hugs the coastline
    ice_edge_arr[land_mask] = False
    ice_edge_arr[region_mask == 13] = False

    return ice_edge_arr


def arr_to_ice_edge_rgba_arr(arr, thresh, land_mask, region_mask, rgb):

    ice_edge_arr = arr_to_ice_edge_arr(arr, thresh, land_mask, region_mask)

    # Contour pixels -> alpha=1, alpha=0 elsewhere
    ice_edge_rgba_arr = np.zeros((*arr.shape, 4))
    ice_edge_rgba_arr[:, :, 3] = ice_edge_arr
    ice_edge_rgba_arr[:, :, :3] = rgb

    return ice_edge_rgba_arr


###############################################################################
############### VIDEOS
###############################################################################


def xarray_to_video(da, video_path, fps, mask=None, mask_type='contour', clim=None,
                    crop=None, data_type='abs', video_dates=None, cmap='viridis',
                    figsize=15, dpi=300):

    '''
    Generate video of an xarray.DataArray. Optionally input a list of
    `video_dates` to show, otherwise the full set of time coordiantes
    of the dataset is used.

    Parameters:
    da (xr.DataArray): Dataset to create video of.

    video_path (str): Path to save the video to.

    fps (int): Frames per second of the video.

    mask (np.ndarray): Boolean mask with True over masked elements to overlay
    as a contour or filled contour. Defaults to None (no mask plotting).

    mask_type (str): 'contour' or 'contourf' dictating whether the mask is overlaid
    as a contour line or a filled contour.

    data_type (str): 'abs' or 'anom' describing whether the data is in absolute
    or anomaly format. If anomaly, the colorbar is centred on 0.

    video_dates (list): List of Pandas Timestamps or datetime.datetime objects
    to plot video from the dataset.

    crop (list): [(a, b), (c, d)] to crop the video from a:b and c:d

    clim (list): Colormap limits. Default is None, in which case the min and max values
    of the array are used.

    cmap (str): Matplotlib colormap.

    figsize (int or float): Figure size in inches.

    dpi (int): Figure DPI.
    '''

    if clim is not None:
        min = clim[0]
        max = clim[1]
    elif clim is None:
        max = da.max().values
        min = da.min().values

        if data_type == 'anom':
            if np.abs(max) > np.abs(min):
                min = -max
            elif np.abs(min) > np.abs(max):
                max = -min

    def make_frame(date):
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        fig.set_dpi(dpi)
        im = ax.imshow(da.sel(time=date), cmap=cmap, clim=(min, max))
        if mask is not None:
            if mask_type == 'contour':
                ax.contour(mask, levels=[.5, 1], colors='k')
            elif mask_type == 'contourf':
                ax.contourf(mask, levels=[.5, 1], colors='k')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        ax.set_title('{:04d}/{:02d}/{:02d}'.format(date.year, date.month, date.day), fontsize=figsize*4)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax)

        # TEMP crop to image
        # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()
        return image

    if video_dates is None:
        video_dates = [pd.Timestamp(date).to_pydatetime() for date in da.time.values]

    if crop is not None:
        a = crop[0][0]
        b = crop[0][1]
        c = crop[1][0]
        d = crop[1][1]
        da = da.isel(xc=np.arange(a, b), yc=np.arange(c, d))
        if mask is not None:
            mask = mask[a:b, c:d]

    imageio.mimsave(video_path,
                    [make_frame(date) for date in tqdm(video_dates)],
                    fps=fps)
