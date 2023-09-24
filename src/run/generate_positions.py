import numpy as np

from src.preprocessing.board_representation import board_to_tensor
from src.preprocessing.generate_board import generate_random_board
from src.preprocessing.extract_board import extract_board
from src.utils.data_loader import save_train_test
	
if __name__ == "__main__":
	POSITIONS = 1_000_000
	TRAIN_RATIO = 0.9
	PATH = "./data"
	MODE = "artificial"

	extractor = extract_board()
	board_tensors = np.empty((POSITIONS,8,8,15), dtype=bool)
	for i in range(POSITIONS):
		print(f"generating position {i+1} of {POSITIONS}", end="\r")
		if MODE == "artificial":
			new_board = generate_random_board(max_pieces=6)
		elif MODE == "real":
			new_board = next(extract_board())
		new_board_tensor = board_to_tensor(new_board)
		board_tensors[i] = new_board_tensor
	
	train_cutoff = int(POSITIONS*TRAIN_RATIO)
	save_train_test(
		path=PATH,
		train_split=board_tensors[:train_cutoff,:],
		test_split=board_tensors[train_cutoff:,:]
	)
