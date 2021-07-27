import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))  # if using jupyter kernel
import config
import argparse
import logging
import re

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tensorflow as tf

'''
Author: James Byrne (BAS). Modified by Tom Andersson.

Using a given dataloader_ID, loads the NumPy arrays in
trained_networks/<dataloader_ID>/obs_train_val_data/numpy/ generated from
icenet/gen_numpy_obs_train_val_data.py and converts them to TfRecords datasets
for faster training with TensorFlow. Saves the results in
trained_networks/<dataloader_ID>/obs_train_val_data/tfrecords/.
'''

#### User input
####################################################################

dataloader_ID = '2021_06_15_1854_icenet_nature_communications'

#### Set up paths
####################################################################

dataloader_ID_folder = os.path.join(config.networks_folder, dataloader_ID, 'obs_train_val_data')

#### Functions
####################################################################


def get_args():
    a = argparse.ArgumentParser()
    a.add_argument("-s", "--batch-size", type=int, default=2)
    a.add_argument("-v", "--verbose", default=False, action="store_true")
    a.add_argument("-w", "--workers", type=int, default=4)
    return a.parse_args()


def convert_batch(tf_path, data):
    logging.info("{} with {} samples".format(tf_path, data[0].shape[0]))

    with tf.io.TFRecordWriter(tf_path) as writer:
        logging.info("Processing batch file {}".format(tf_path))

        for i in range(data[0].shape[0]):
            logging.info("Processing input {}/{}".format(i, data[0].shape[0]))

            (x, y, w) = (data[0][i,...], data[1][i, ...], data[2][i, ...])

            logging.info("x shape {}, y shape {}, w shape {}".format(x.shape, y.shape, w.shape))

            record_data = tf.train.Example(features=tf.train.Features(feature={
                "x": tf.train.Feature(float_list=tf.train.FloatList(value=x.reshape(-1))),
                "y": tf.train.Feature(float_list=tf.train.FloatList(value=y.reshape(-1))),
                "w": tf.train.Feature(float_list=tf.train.FloatList(value=w.reshape(-1))),
            })).SerializeToString()

            writer.write(record_data)


def convert_data(x_data, y_data, sample_weight_data, output_dir,
                 workers=4, batch_size=1, wildcard="*"):
    x_data = np.load(x_data)
    y_data = np.load(y_data)
    sample_weight_data = np.load(sample_weight_data)

    options = tf.io.TFRecordOptions()
    batch_i = 0

    os.makedirs(output_dir)

    def batch(x, y, w, batch_size):
        i = 0
        while i < x.shape[0]:
            yield x[i:i+batch_size], y[i:i+batch_size], w[i:i+batch_size]
            i += batch_size

    tasks = []

    for data in batch(x_data, y_data, sample_weight_data, batch_size):
        tf_path = os.path.join(output_dir, "{:05}.tfrecord".format(batch_i))
        tasks.append((tf_path, data))
        batch_i += 1

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for args in tasks:
            executor.submit(convert_batch, *args)


#### Main script
####################################################################
if __name__ == "__main__":
    args = get_args()
    log_state = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(log_state)

    for data in [('X_train', 'y_train', 'sample_weight_train'),
                 ('X_val', 'y_val', 'sample_weight_val')]:
        x_data = os.path.join(dataloader_ID_folder, 'numpy', "{}.npy".format(data[0]))
        y_data = os.path.join(dataloader_ID_folder, 'numpy', "{}.npy".format(data[1]))
        sample_weight_data = os.path.join(dataloader_ID_folder, 'numpy', "{}.npy".format(data[2]))

        if not os.path.exists(x_data) or not os.path.exists(y_data):
            logging.warning("Skipping {}, one does not exist.".format(data))
            continue

        set_type = re.search(r"(train|test|val)", data[0], re.IGNORECASE).group(1)
        output_path = os.path.join(dataloader_ID_folder, 'tfrecords', set_type)
        convert_data(x_data, y_data, sample_weight_data, output_path,
                     workers=args.workers, batch_size=args.batch_size)
