import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint

# define step decay function
def exp_decay(epoch):
    initial_lrate = 1e-4
    k = 0.1
    lrate = initial_lrate * np.exp(-k*epoch)
    return lrate


class LossHistory_(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
        
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(exp_decay(epoch))
        print('lr:', exp_decay(len(self.losses)))


def get_callbacks(checkpoint_path):
    # learning schedule callback
    loss_history_ = LossHistory_()
    lrate_ = LearningRateScheduler(exp_decay)

    keras_callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=4,
            mode='min',
            min_delta=0.01
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
    ]

    callbacks_list_ = [loss_history_, lrate_] + keras_callbacks
    return callbacks_list_