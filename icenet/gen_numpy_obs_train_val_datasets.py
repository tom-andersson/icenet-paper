import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))  # if using jupyter kernel
import config
from utils import IceNetDataLoader

'''
Using a given dataloader_ID, loads monthly .npy files from
data/network_datasets/<dataset_name>/ and stores the entire training/validation
input-output datasets as large NumPy arrays. The results are saved to
trained_networks/<dataloader_ID>/obs_train_val_data/numpy/.

'''

#### User input
####################################################################

dataloader_ID = '2021_06_15_1854_icenet_nature_communications'

#### Data loaders; set up paths
###############################################################################

dataloader_config_fpath = os.path.join(config.dataloader_config_folder, dataloader_ID +'.json')
dataloader_ID_folder = os.path.join(config.networks_folder, dataloader_ID, 'obs_train_val_data', 'numpy')

if not os.path.exists(dataloader_ID_folder):
    os.makedirs(dataloader_ID_folder)

# Load the training and validation data loader objects from the pickle file
print("\nSetting up the training and validation data"
      "loaders with config file: {}\n\n".format(dataloader_ID))
dataloader = IceNetDataLoader(dataloader_config_fpath)
val_dataloader = IceNetDataLoader(dataloader_config_fpath)
val_dataloader.convert_to_validation_data_loader()
print('\n\nDone.\n')

#### Build up numpy dataset
###############################################################################

print('\nBuilding up the training dataset... ', end='', flush=True)
X_train = []
y_train = []
sample_weight_train = []
for date in dataloader.all_forecast_IDs:
    X, y, sample_weight = dataloader.data_generation(date)
    X_train.append(X)
    y_train.append(y)
    sample_weight_train.append(sample_weight)

X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)
sample_weight_train = np.concatenate(sample_weight_train, axis=0)
print('Done. X_train {:.2f} GB'.format(X_train.nbytes/1e9))

print('Building up the validation dataset... ', end='', flush=True)
X_val = []
y_val = []
sample_weight_val = []
for date in val_dataloader.all_forecast_IDs:
    X, y, sample_weight = dataloader.data_generation(date)
    X_val.append(X)
    y_val.append(y)
    sample_weight_val.append(sample_weight)

X_val = np.concatenate(X_val, axis=0)
y_val = np.concatenate(y_val, axis=0)
sample_weight_val = np.concatenate(sample_weight_val, axis=0)
print('Done. X_val {:.2f} GB'.format(X_val.nbytes/1e9))

print('\nSaving... ', end='', flush=True)
np.save(os.path.join(dataloader_ID_folder, 'X_train.npy'), X_train)
np.save(os.path.join(dataloader_ID_folder, 'y_train.npy'), y_train)
np.save(os.path.join(dataloader_ID_folder, 'sample_weight_train.npy'), sample_weight_train)
np.save(os.path.join(dataloader_ID_folder, 'X_val.npy'), X_val)
np.save(os.path.join(dataloader_ID_folder, 'y_val.npy'), y_val)
np.save(os.path.join(dataloader_ID_folder, 'sample_weight_val.npy'), sample_weight_val)
print('Done.')
