import tensorflow as tf
from tensorflow import keras
import numpy as np


class Autoencoder(keras.models.Model):
    def __init__(self,encoder_sub,decoder_sub,observable_vec_size,latent_vec_size):
        super(Autoencoder, self).__init__()
        self.encoder_sub = encoder_sub
        self.decoder_sub = decoder_sub
        self.latent_vec_size = latent_vec_size
        self.observable_vec_size = observable_vec_size


        self.encoder = keras.Sequential([
            self.encoder_sub,
            ])

        self.encoder.add(keras.layers.Dense(self.latent_vec_size,activation='linear'))

        self.decoder = self.decoder_sub


    def call(self, x,training=False):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded


    def get_config(self):
        return {
                "observable_vec_size": self.observable_vec_size,
                "latent_vec_size": self.latent_vec_size,
                "encoder_sub": self.encoder_sub,
                "decoder_sub": self.decoder_sub
               }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
