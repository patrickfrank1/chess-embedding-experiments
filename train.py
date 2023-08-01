import tensorflow
from tensorflow import keras
import mlflow

from utils.data_loader import load_train_test
from model import get_model

if __name__ == "__main__":

	mlflow.autolog()

	# parameters
	DATA_DIR = "./data"
	MODEL_DIR = "./model"

	# get model definition
	autoencoder: keras.Model = get_model("vanilla_dense")["autoencoder"]

	# compile model
	autoencoder.compile(
		optimizer='rmsprop',
		loss=keras.losses.MeanSquaredError(),
		metrics=[keras.losses.MeanSquaredError(), 'binary_crossentropy', keras.losses.BinaryCrossentropy(from_logits=True, name="ce_from_logits")],
		jit_compile=True
	)

	# print model architecture
	autoencoder.summary()

	# load train and test data
	train_data, test_data = load_train_test(DATA_DIR)

	print("Train data:")
	print(train_data.shape, train_data.dtype, train_data[0,:,:,0])
	print("Test data:")
	print(test_data.shape, test_data.dtype, test_data[0,:,:,0])

	# Defiane Callbacks
	callbacks = [
		keras.callbacks.EarlyStopping(
			monitor="val_loss",
			min_delta=2,
			patience=11,
			verbose=0,
			mode="auto",
			restore_best_weights=True,
			start_from_epoch=5,
		),
		keras.callbacks.ReduceLROnPlateau(
			monitor="val_loss",
			factor=0.1,
			patience=5,
			verbose=0,
			mode="auto",
			min_delta=0.33,
			cooldown=0,
			min_lr=1e-6
		)
	]


	# train model
	history = autoencoder.fit(
		x=train_data,
		y=train_data,
		epochs=50,
		batch_size=8,
		shuffle=False,
		validation_data=(test_data, test_data),
		callbacks=callbacks
	)

	# save
	autoencoder.save(f"{MODEL_DIR}")
