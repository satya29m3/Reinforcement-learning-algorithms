import tensorflow as tf
from keras.callbacks import Tensorboard
class DQNagent:
    def create_model(self):
        model = tf.keras.model.Sequential([
            tf.keras.layers.Conv2D()
        ]) 