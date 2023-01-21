from tensorflow.keras import layers, losses
import tensorflow as tf
from tensorflow import keras
import numpy as np


class Sampler(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



class VariationalAutoencoder(tf.keras.models.Model):
    def __init__(self,encoder_sub,decoder_sub,observable_vec_size,latent_vec_size=5,**kwargs):
        super(VariationalAutoencoder, self).__init__(**kwargs)
        self.encoder_sub = encoder_sub
        self.decoder_sub = decoder_sub
        self.observable_vec_size = observable_vec_size
        self.latent_vec_size = latent_vec_size
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")


        encoder_inputs = tf.keras.Input(shape=(observable_vec_size,))
        before_encoded = self.encoder_sub(encoder_inputs)

        z_mean = layers.Dense(self.latent_vec_size, name="z_mean")(before_encoded)
        z_log_var = layers.Dense(self.latent_vec_size, name="z_log_var")(before_encoded)
        z = Sampler()([z_mean,z_log_var])

        self.encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        self.decoder = self.decoder_sub

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data,_=None):
        print(type(data))
        print(data.shape)
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_mean(
                    tf.keras.losses.mse(data, reconstruction)
                    ,axis=(0)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self,x,_=None):
        z_mean,z_log_var,z = self.encoder(x)
        return self.decoder(z)
        



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

