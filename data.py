import random

import numpy as np
import chess

from utils.board_representations import board_to_tensor
from utils.data_loader import save_train_test

def generate_board(max_pieces: int = 1) -> chess.Board:
	"""
	Generate random legal positions with two kings and maximum n random pieces.
	The more random pieces required the more inefficient the algorithm becomes.
	"""
	pieces = random.choice(np.arange(max_pieces)) + 1
	board = chess.Board(fen=None)
	white_king_square = random.choice(chess.SQUARES)
	free_sqares = list(chess.SQUARES)
	free_sqares.remove(white_king_square)
	black_king_square = random.choice(free_sqares)
	free_sqares.remove(black_king_square)
	piece_map = {
		white_king_square: chess.Piece.from_symbol("K"),
		black_king_square: chess.Piece.from_symbol("k")
	}

	for _ in range(pieces):
		random_piece = chess.Piece(
			piece_type=np.random.randint(1,6),
			color=random.choice([True, False])
		)
		random_square = random.choice(free_sqares)
		piece_map[random_square] = random_piece
		free_sqares.remove(random_square)
		
	board.set_piece_map(piece_map)
	# Random turn and no castling rights
	board.turn = random.choice([True, False])
	if board.is_valid():
		return board
	return generate_board(max_pieces=max_pieces)
	
if __name__ == "__main__":
	POSITIONS = 100_000
	TRAIN_RATIO = 0.9
	PATH = "./data"

	board_tensors = np.empty((POSITIONS,8,8,15), dtype=bool)
	for i in range(POSITIONS):
		new_board = generate_board(max_pieces=6)
		new_board_tensor = board_to_tensor(new_board)
		board_tensors[i] = new_board_tensor
	
	train_cutoff = int(POSITIONS*TRAIN_RATIO)
	save_train_test(
		path=PATH,
		train_split=board_tensors[:train_cutoff,:],
		test_split=board_tensors[train_cutoff:,:]
	)
