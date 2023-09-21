import os

import numpy as np
import tensorflow as tf


class AutoencoderDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory_path, batch_size):
        self.directory_path = directory_path
        self.files = self._get_npz_files()
        self.current_file = self.files.pop()
        self.visited_files = []
        self.positions = np.load(self.current_file)['data']
        self.batch_size = batch_size

    def _get_npz_files(self) -> list[str]:
        file_list = os.listdir(self.directory_path)
        npz_files = [f"{self.directory_path}/{filename}" for filename in file_list if filename.endswith(".npz")]
        return npz_files

    def __len__(self):
        # Calculate the total number of batches based on the number of samples in the file and the batch size.
        return int(np.ceil(len(self.positions) / self.batch_size))

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.positions))
        batch = self.positions[low:high]
        return batch, batch

    def on_epoch_end(self):
        self.visited_files.append(self.current_file)
        if len(self.files) == 0:
            self.files = self.visited_files
            self.visited_files = []
        self.current_file = self.files.pop()
        self.positions = np.load(self.current_file)['data']

    def total_dataset_length(self):
        # Calculate the total number of batches based on the number of samples in the file and the batch size.
        number_samples = 0
        for file_path in self._get_npz_files():
            number_samples += int(np.ceil(len(np.load(file_path)['data']) / self.batch_size))
        return number_samples
