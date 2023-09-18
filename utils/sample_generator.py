import os

import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory_path, batch_size):
        self.directory_path = directory_path
        self.batch_size = batch_size

    def _get_npz_files(self) -> list[str]:
        file_list = os.listdir(self.directory_path)
        npz_files = [filename for filename in file_list if filename.endswith(".npz")]
        return npz_files


    def __len__(self):
        # Calculate the total number of batches based on the number of samples in the file
        # and the batch size.
        # You may need to load the file to get the number of samples.
        # Example: self.num_samples = len(np.load(self.file_path)['data'])
        #           return int(np.ceil(self.num_samples / self.batch_size))
        pass

    def __getitem__(self, index):
        # Load a batch of data from the file based on the batch size and current index.
        # Example: data = np.load(self.file_path)['data']
        #          batch_data = data[index * self.batch_size:(index + 1) * self.batch_size]
        #          batch_labels = data[index * self.batch_size:(index + 1) * self.batch_size]
        #          return batch_data, batch_labels
        pass
