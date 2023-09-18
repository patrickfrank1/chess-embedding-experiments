import datetime as dt

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import mlflow

from utils.data_loader import load_train_test
from model import get_model

DTYPE = 'bfloat16'


@keras.saving.register_keras_serializable()
def sum_squared_loss(y_true, y_pred):
	batch_size = tf.cast(tf.shape(y_true)[0], DTYPE)
	y_true = K.cast(y_true, dtype=DTYPE)
	y_pred = K.cast(y_pred, dtype=DTYPE)
	squared_difference = K.square(y_true - y_pred)
	loss = K.sum(squared_difference) / batch_size
	return loss

@keras.saving.register_keras_serializable()
def num_pc_reg(y_true, y_pred):
	epsilon = 1.e-3
	batch_size = tf.cast(tf.shape(y_true)[0], DTYPE)
	y_true = K.cast(y_true, dtype=DTYPE)
	y_pred = K.cast(y_pred, dtype=DTYPE)
	pieces_true = K.sum(y_true)
	pieces_predicted = K.sum(y_pred)
	loss = K.square(pieces_true - pieces_predicted) / (epsilon + pieces_predicted) / batch_size
	return loss

@keras.saving.register_keras_serializable()
def pc_column_reg(y_true, y_pred):
	batch_size = tf.cast(tf.shape(y_true)[0], DTYPE)
	y_true = K.cast(y_true, dtype=DTYPE)
	y_pred = K.cast(y_pred, dtype=DTYPE)
	piece_representation_true = y_true[:,:,:,:12]
	piece_representation_pred = y_pred[:,:,:,:12]
	sum_over_pieces_true = K.sum(piece_representation_true, axis=3)
	sum_over_pieces_pred = K.sum(piece_representation_pred, axis=3)
	deviation_from_legal = K.square(sum_over_pieces_true - sum_over_pieces_pred)
	loss = K.sum(deviation_from_legal) / batch_size
	return loss

@keras.saving.register_keras_serializable()
def pc_plane_reg(y_true, y_pred):
	batch_size = tf.cast(tf.shape(y_true)[0], DTYPE)
	y_true = K.cast(y_true, dtype=DTYPE)
	y_pred = K.cast(y_pred, dtype=DTYPE)
	piece_representation_true = y_true[:,:,:,:12]
	piece_representation_pred = y_pred[:,:,:,:12]
	sum_over_planes_true = K.sum(K.sum(piece_representation_true, axis=2), axis=1)
	sum_over_planes_pred = K.sum(K.sum(piece_representation_pred, axis=2), axis=1)
	deviation_from_legal = K.square(sum_over_planes_true - sum_over_planes_pred)
	loss = K.sum(deviation_from_legal) / batch_size
	return loss

@keras.saving.register_keras_serializable()
def custom_regularized_loss(y_true, y_pred):
	alpha = 1.0
	beta = 0.1
	gamma = 0.1
	loss = sum_squared_loss(y_true, y_pred)
	regularizer_1 = num_pc_reg(y_true, y_pred)
	regularizer_2 = pc_column_reg(y_true, y_pred)
	regularizer_3 = pc_plane_reg(y_true, y_pred)
	return loss + alpha * regularizer_1 + beta * regularizer_2 + gamma * regularizer_3


if __name__ == "__main__":

	mlflow.autolog()

	# parameters
	DATA_DIR = "./data"
	MODEL_DIR = "./model"

	# get model definition
	autoencoder: keras.Model = get_model("trivial")["autoencoder"]

	# compile model
	autoencoder.compile(
		optimizer='rmsprop',
		loss=custom_regularized_loss,
		metrics=[
			sum_squared_loss,
			num_pc_reg,
			pc_column_reg,
			pc_plane_reg
		],
		jit_compile=True
	)

	# print model architecture
	autoencoder.summary()

	# load train and test data
	train_data, test_data = load_train_test(DATA_DIR, "position_tensor_000")

	print("Train data:")
	print(train_data.shape, train_data.dtype, train_data[0,:,:,0])
	print("Test data:")
	print(test_data.shape, test_data.dtype, test_data[0,:,:,0])
	train_sample = tf.convert_to_tensor(train_data[0:1], dtype=tf.float32)
	test_sample = tf.convert_to_tensor(test_data[0:1], dtype=tf.float32)

	print("Sum squared loss on applied to first train-test samples:")
	print(sum_squared_loss(train_sample, test_sample))

	print("Total number of pieces loss:")
	print(num_pc_reg(train_sample, test_sample))

	print("Number of pieces per square loss:")
	print(pc_column_reg(train_sample, test_sample))

	print("Number of pieces per plane loss:")
	print(pc_plane_reg(train_sample, test_sample))

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
		)
	]


	# train model
	history = autoencoder.fit(
		x=train_data,
		y=train_data,
		epochs=1,
		batch_size=32,
		shuffle=True,
		validation_data=(test_data, test_data),
		callbacks=callbacks
	)

	# save
	autoencoder.save(f"{MODEL_DIR}/{dt.datetime.now():%Y%m%d%H%M%S}_autoencoder.keras")
