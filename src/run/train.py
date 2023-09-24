import os
import datetime as dt

import tensorflow as tf
from tensorflow import keras
import mlflow

from src.modeling.model import get_model
from src.training.sample_generator import AutoencoderDataGenerator, ReconstructAutoencoderDataGenerator
from src.modeling import custom_losses as cl

if __name__ == "__main__":

	mlflow.autolog()

	# parameters
	DATA_DIR = "./data"
	MODEL_DIR = "./model"
	BATCH_SIZE = 32
	EPOCHS = 30
	STEPS_PER_EPOCH = None #1000
	VALIDATION_STEPS = None #100

	# create required directories if they do not yet exist
	os.makedirs(DATA_DIR, exist_ok=True)
	os.makedirs(f"{MODEL_DIR}/checkpoints", exist_ok=True)

	# get model definition
	autoencoder: keras.Model = get_model("trivial")["autoencoder"]

	# compile model
	autoencoder.compile(
		optimizer='rmsprop',
		loss=cl.custom_regularized_loss,
		metrics=[
			cl.sum_squared_loss,
			cl.num_pc_reg,
			cl.pc_column_reg,
			cl.pc_plane_reg
		],
		jit_compile=True
	)

	# print model architecture
	autoencoder.summary()

	# load train and test data
	train_data = ReconstructAutoencoderDataGenerator(f"{DATA_DIR}/train", number_squares=7, batch_size=BATCH_SIZE)
	test_data = ReconstructAutoencoderDataGenerator(f"{DATA_DIR}/test", number_squares=7, batch_size=BATCH_SIZE)

	# TODO: also print a couple of layers

	print("Train data:")
	print(f"Number of train samples: {train_data.total_dataset_length()}")
	train_batch = train_data.__getitem__(0)
	print(f"First batch: len={len(train_batch)}")
	train_sample = train_batch[0]
	print(f"First item: shape={train_sample.shape}, dtype={train_sample.dtype}")

	pieces = ["pawn", "knight", "bishop", "rook", "queen", "king"]
	piece_map = ["white "+piece for piece in pieces] + \
    			["black "+piece for piece in pieces] + \
    			["castling rights", "en passant", "turn"]
	print("First train position:")
	for i, piece in enumerate(piece_map):
		print(piece)
		print(train_sample[0,:,:,i].astype(int))

	print("Test data:")
	print(f"Number of test samples: {test_data.total_dataset_length()}")
	test_batch = test_data.__getitem__(0)
	print(f"First batch: len={len(test_batch)}")
	test_sample = test_batch[0]
	print(f"First item: shape={test_sample.shape}, dtype={test_sample.dtype}")

	print("First test position:")
	for i, piece in enumerate(piece_map):
		print(piece)
		print(test_sample[0,:,:,i].astype(int))

	train_sample = tf.convert_to_tensor(train_batch[0][0:1], dtype=tf.float32)
	test_sample = tf.convert_to_tensor(test_batch[0][0:1], dtype=tf.float32)

	print("Sum squared loss on applied to first train-test samples:")
	print(cl.sum_squared_loss(train_sample, test_sample))

	print("Total number of pieces loss:")
	print(cl.num_pc_reg(train_sample, test_sample))

	print("Number of pieces per square loss:")
	print(cl.pc_column_reg(train_sample, test_sample))

	print("Number of pieces per plane loss:")
	print(cl.pc_plane_reg(train_sample, test_sample))

	# Define Callbacks
	callbacks = [
		keras.callbacks.EarlyStopping(
			monitor="val_loss",
			min_delta=0.25,
			patience=15,
			verbose=0,
			mode="auto",
			restore_best_weights=True,
			start_from_epoch=3,
		),
		keras.callbacks.ReduceLROnPlateau(
			monitor="val_loss",
			factor=0.3,
			patience=5,
			verbose=0,
			mode="auto",
			min_delta=0.1,
			cooldown=0,
			min_lr=1e-6
		),
		tf.keras.callbacks.BackupAndRestore(
			f"{MODEL_DIR}/checkpoints",
			save_freq="epoch",
			delete_checkpoint=True,
			save_before_preemption=False
		)
	]


	# train model
	history = autoencoder.fit(
		train_data,
		epochs=EPOCHS,
		steps_per_epoch=STEPS_PER_EPOCH,
		validation_steps=VALIDATION_STEPS,
		batch_size=BATCH_SIZE,
		shuffle=True,
		validation_data=test_data,
		callbacks=callbacks
	)

	# save
	autoencoder.save(f"{MODEL_DIR}/{dt.datetime.now():%Y%m%d%H%M%S}_autoencoder.keras")
