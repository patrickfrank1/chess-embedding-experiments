import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

def get_model() -> dict:
    EMBEDDING_SIZE = 128

    encoder_input = layers.Input(shape=(8,8,15), dtype=tf.float16)
    encoder = layers.Reshape((8*8*15,))(encoder_input)
    encoder = layers.Dense(2*EMBEDDING_SIZE, activation='relu')(encoder)
    encoder = layers.Dense(EMBEDDING_SIZE, activation='relu')(encoder)

    decoder_input = layers.Input(shape=(EMBEDDING_SIZE,))
    decoder = layers.Dense(2*EMBEDDING_SIZE, activation='relu')(decoder_input)
    decoder = layers.Dense(8*8*15, activation='relu')(decoder_input)
    decoder = layers.Reshape((8,8,15))(decoder)

    encoder = keras.Model(inputs=encoder_input, outputs=encoder, name='encoder')
    decoder = keras.Model(inputs=decoder_input, outputs=decoder, name='decoder')
    autoencoder = keras.Model(inputs=encoder_input, outputs=decoder(encoder(encoder_input)), name='autoencoder')

    return {'encoder': encoder, 'decoder': decoder, 'autoencoder': autoencoder}