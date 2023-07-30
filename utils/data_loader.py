from typing import Tuple

import numpy as np

def save_train_test(path: str, train_split: np.ndarray , test_split: np.ndarray) -> None:

	np.savez_compressed(f"{path}/train", data=train_split)
	np.savez_compressed(f"{path}/test", data=test_split)

def load_train_test(path: str) -> Tuple[np.ndarray, np.ndarray]:
	train, test = None, None
	with np.load(f"{path}/train.npz") as data:
		train = data["data"]
	with np.load(f"{path}/test.npz") as data:
		test = data["data"]
	return train, test