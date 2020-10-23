import tensorflow as tf


class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('binary_accuracy') > 0.99:
            print("\nReached 99% Binary Accuracy so cancelling traning!")
            self.model.stop_training = True
