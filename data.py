import random

import numpy as np
import chess

from utils.board_representations import board_to_tensor
from utils.data_loader import save_train_test

def generate_board() -> chess.Board:
	"""
	Generate random legal positions with two kings and a random piece.
	"""
	board = chess.Board(fen=None)
	# generate a random piece that is not a king
	random_piece = chess.Piece(
		piece_type=np.random.randint(1,6),
		color=random.choice([True, False])
	)
	board.set_piece_map({
		random.choice(chess.SQUARES): random_piece,
		random.choice(chess.SQUARES): chess.Piece.from_symbol("K"),
		random.choice(chess.SQUARES): chess.Piece.from_symbol("k"),
	})
	# Random turn and no castling rights
	board.turn = random.choice([True, False])
	if board.is_valid():
		return board
	return generate_board()
	
if __name__ == "__main__":
	POSITIONS = 100_000
	TRAIN_RATIO = 0.9
	PATH = "./data"

	board_tensors = np.empty((POSITIONS,8,8,15), dtype=bool)
	for i in range(POSITIONS):
		new_board = generate_board()
		new_board_tensor = board_to_tensor(new_board)
		board_tensors[i] = new_board_tensor
	
	train_cutoff = int(POSITIONS*TRAIN_RATIO)
	save_train_test(
		path=PATH,
		train_split=board_tensors[:train_cutoff,:],
		test_split=board_tensors[train_cutoff:,:]
	)
