import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

def get_model(model: str) -> dict[str, keras.Model]:
	if model == "cnn_dense":
		return cnn_dense()
	elif model == "vanilla_dense":
		return vanilla_dense()
	elif model == "trivial":
		return trivial2()
	else:
		raise ValueError("The requested neral network architecture does not exist.")


def cnn_dense() -> dict[str, keras.Model]:
	EMBEDDING_SIZE = 768
	CONV_FILTERS = 16

	encoder_input = layers.Input(shape=(8,8,15,1), dtype=tf.float32)

	# Encoder
	x = layers.Conv3D(CONV_FILTERS, (3, 3, 15), activation="relu", padding="same")(encoder_input)
	x = layers.MaxPooling3D((2, 2, 1), padding="same")(x)
	x = layers.Conv3D(CONV_FILTERS, (3, 3, 15), activation="relu", padding="same")(x)
	x = layers.MaxPooling3D((2, 2, 1), padding="same")(x)
	x = layers.Flatten()(x)
	x = layers.Dense(EMBEDDING_SIZE, activation="relu")(x)

	# Decoder
	decoder_input = layers.Input(shape=(EMBEDDING_SIZE,), dtype=tf.float32)
	y = layers.Dense(CONV_FILTERS*2*2*15, activation="relu")(decoder_input)
	y = layers.Reshape((2, 2, 15, CONV_FILTERS))(y)
	y = layers.Conv3DTranspose(CONV_FILTERS, (3, 3, 15), strides=(2,2,1), activation="relu", padding="same")(y)
	y = layers.Conv3DTranspose(CONV_FILTERS, (3, 3, 15), strides=(2,2,1), activation="relu", padding="same")(y)
	y = layers.Conv3D(1, (8, 8, 15), activation="sigmoid", padding="same")(y)
	y = layers.Reshape((8,8,15))(y)

	# Autoencoder
	encoder = keras.Model(inputs=encoder_input, outputs=x, name='encoder')
	decoder = keras.Model(inputs=decoder_input, outputs=y, name='decoder')
	autoencoder = keras.Model(inputs=encoder_input, outputs=decoder(encoder(encoder_input)), name='autoencoder')

	return {'encoder': encoder, 'decoder': decoder, 'autoencoder': autoencoder}

def vanilla_dense() -> dict[str, keras.Model]:
	EMBEDDING_SIZE = 768
	dtype = tf.bfloat16

	# Encoder
	encoder_input = layers.Input(shape=(8,8,15), dtype=dtype)
	encoder = layers.Reshape((8*8*15,))(encoder_input)
	# encoder = layers.Dense(2*EMBEDDING_SIZE, activation='relu')(encoder)
	encoder = layers.Dense(EMBEDDING_SIZE, activation='relu')(encoder)

	# Decoder
	decoder_input = layers.Input(shape=(EMBEDDING_SIZE,), dtype=dtype)
	# decoder = layers.Dense(2*EMBEDDING_SIZE, activation='relu')(decoder_input)
	decoder = layers.Dense(8*8*15, activation='relu')(decoder_input)
	decoder = layers.Reshape((8,8,15))(decoder)

	# Autoencoder
	encoder = keras.Model(inputs=encoder_input, outputs=encoder, name='encoder')
	decoder = keras.Model(inputs=decoder_input, outputs=decoder, name='decoder')
	autoencoder = keras.Model(inputs=encoder_input, outputs=decoder(encoder(encoder_input)), name='autoencoder')

	return {'encoder': encoder, 'decoder': decoder, 'autoencoder': autoencoder}

def cnn_dense() -> dict[str, keras.Model]:
	EMBEDDING_SIZE = 768
	CONV_FILTERS = 16
	dtype = tf.bfloat16

	encoder_input = layers.Input(shape=(8,8,15,1), dtype=dtype)

	# Encoder
	x = layers.Conv3D(CONV_FILTERS, (3, 3, 15), activation="relu", padding="same")(encoder_input)
	x = layers.MaxPooling3D((2, 2, 1), padding="same")(x)
	x = layers.Conv3D(CONV_FILTERS, (3, 3, 15), activation="relu", padding="same")(x)
	x = layers.MaxPooling3D((2, 2, 1), padding="same")(x)
	x = layers.Flatten()(x)
	x = layers.Dense(EMBEDDING_SIZE, activation="relu")(x)

	# Decoder
	decoder_input = layers.Input(shape=(EMBEDDING_SIZE,), dtype=dtype)
	y = layers.Dense(CONV_FILTERS*2*2*15, activation="relu")(decoder_input)
	y = layers.Reshape((2, 2, 15, CONV_FILTERS))(y)
	y = layers.Conv3DTranspose(CONV_FILTERS, (3, 3, 15), strides=(2,2,1), activation="relu", padding="same")(y)
	y = layers.Conv3DTranspose(CONV_FILTERS, (3, 3, 15), strides=(2,2,1), activation="relu", padding="same")(y)
	y = layers.Conv3D(1, (8, 8, 15), activation="sigmoid", padding="same")(y)
	y = layers.Reshape((8,8,15))(y)

	# Autoencoder
	encoder = keras.Model(inputs=encoder_input, outputs=x, name='encoder')
	decoder = keras.Model(inputs=decoder_input, outputs=y, name='decoder')
	autoencoder = keras.Model(inputs=encoder_input, outputs=decoder(encoder(encoder_input)), name='autoencoder')

	return {'encoder': encoder, 'decoder': decoder, 'autoencoder': autoencoder}

def trivial() -> dict[str, keras.Model]:
	EMBEDDING_SIZE = 64

	# Encoder
	encoder_input = layers.Input(shape=(8,8,15), dtype=tf.float32)
	encoder = layers.Reshape((8*8*15,))(encoder_input)
	encoder = layers.Dense(EMBEDDING_SIZE, activation='relu')(encoder)
	decoder = layers.Dense(8*8*15, activation='relu')(encoder)
	decoder = layers.Reshape((8,8,15))(decoder)

	# Autoencoder
	autoencoder = keras.Model(inputs=encoder_input, outputs=decoder, name='autoencoder')

	return {'autoencoder': autoencoder}

def trivial2() -> dict[str, keras.Model]:
	EMBEDDING_SIZE = 920
	dtype = tf.bfloat16

	# Encoder
	encoder_input = layers.Input(shape=(8,8,15), dtype=dtype)
	encoder = layers.Reshape((8*8*15,))(encoder_input)
	encoder = layers.Dense(EMBEDDING_SIZE, activation='relu')(encoder)

	# Decoder
	decoder_input = layers.Input(shape=(EMBEDDING_SIZE,), dtype=dtype)
	decoder = layers.Dense(8*8*15, activation='relu')(decoder_input)
	decoder = layers.Reshape((8,8,15))(decoder)

	# Autoencoder
	encoder = keras.Model(inputs=encoder_input, outputs=encoder, name='encoder')
	decoder = keras.Model(inputs=decoder_input, outputs=decoder, name='decoder')
	autoencoder = keras.Model(inputs=encoder_input, outputs=decoder(encoder(encoder_input)), name='autoencoder')

	return {'encoder': encoder, 'decoder': decoder, 'autoencoder': autoencoder}