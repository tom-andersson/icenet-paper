import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))  # if using jupyter kernel
from utils import IceNetDataPreProcessor

'''
Use the IceNetDataPreProcessor class to normalise and save data in NumPy format
for training IceNet. This can take a minute or so to run.
'''

dataloader_config_fpath = 'dataloader_configs/2021_06_15_1854_icenet_nature_communications.json'

preproc_vars = {
    'siconca': {'anom': False, 'abs': True, 'linear_trend': True},
    'tas': {'anom': True, 'abs': False},
    'ta500': {'anom': True, 'abs': False},
    'tos': {'anom': True, 'abs': False},
    'rsds': {'anom': True, 'abs': False},
    'rsus': {'anom': True, 'abs': False},
    'psl': {'anom': True, 'abs': False},
    'zg500': {'anom': True, 'abs': False},
    'zg250': {'anom': True, 'abs': False},
    'ua10': {'anom': False, 'abs': True},
    'uas': {'anom': False, 'abs': True},
    'vas': {'anom': False, 'abs': True},
    'land': {'metadata': True, 'include': True},
    'circmonth': {'metadata': True, 'include': True}
}

n_linear_years = 35  # Number of past years to used in the linear trend projections

minmax = False

verbose_level = 2

raw_data_shape = (432, 432)

preproc_obs_data = False

# If True, normalisation parameters must have been computed for each variable
#   by running this with `preproc_obs_data` equal to True
preproc_transfer_data = True

cmip_transfer_data = {
    'EC-Earth3': (
        'r2i1p1f1',
        'r7i1p1f1',
        'r10i1p1f1',
        'r12i1p1f1',
        'r14i1p1f1',
    ),
    'MRI-ESM2-0': (
        'r1i1p1f1',
        'r2i1p1f1',
        'r3i1p1f1',
        'r4i1p1f1',
        'r5i1p1f1',
    ),
}

dpp = IceNetDataPreProcessor(
    dataloader_config_fpath=dataloader_config_fpath,
    preproc_vars=preproc_vars,
    n_linear_years=n_linear_years,
    minmax=minmax,
    verbose_level=verbose_level,
    raw_data_shape=raw_data_shape,
    preproc_obs_data=preproc_obs_data,
    preproc_transfer_data=preproc_transfer_data,
    cmip_transfer_data=cmip_transfer_data)
