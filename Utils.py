from tensorflow import keras
from os import path, makedirs


def create_cp_callback(cp_path):
    cp_dir = path.dirname(cp_path)
    # check cp dir exists --> if not create
    if not path.isdir(cp_dir):
        makedirs(cp_dir)

    cp_callback = keras.callbacks.ModelCheckpoint(filepath=cp_path,
                                                  save_weights_only=True,
                                                  verbose=1)

    return cp_callback
