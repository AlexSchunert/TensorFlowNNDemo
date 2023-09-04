from models import create_conv_vanilla_model_functional, create_dense_vanilla_model, \
    create_dense_vanilla_model_functional
from datasets import load_full_mnist, load_full_cifar100
from Utils import create_cp_callback

from datetime import datetime
from os import path, listdir, makedirs
from tensorflow import keras
import numpy as np


def train_dense_vanilla_model_mnist():
    train_images, train_labels, test_images, test_labels = load_full_mnist()
    model = create_dense_vanilla_model_functional()

    # create tensorboard
    # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Create a callback that saves the model's weights
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = path.dirname(checkpoint_path)
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)

    model.fit(
        train_images,
        train_labels,
        epochs=30,
        callbacks=[tensorboard_callback, cp_callback]
    )

    listdir(checkpoint_dir)


def train_conv_vanilla_model_mnist(num_epochs, save_result=False, create_tensorboard=False):
    train_images, train_labels, test_images, test_labels = load_full_mnist()
    model = create_conv_vanilla_model_functional(train_images.shape[1:]+(1,), 10)

    # create tensorboard
    # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    log_dir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    makedirs(log_dir)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Create a callback that saves the model's weights
    checkpoint_path = "training_2/cp.ckpt"
    checkpoint_dir = path.dirname(checkpoint_path)
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)
    train_images = np.reshape(train_images,
                              (np.shape(train_images)[0], np.shape(train_images)[1], np.shape(train_images)[2], 1))

    callbacks = []
    if save_result:
        callbacks.append(cp_callback)

    if create_tensorboard:
        callbacks.append(tensorboard_callback)

    model.fit(
        train_images,
        train_labels,
        epochs=num_epochs,
        callbacks=callbacks
    )

    listdir(checkpoint_dir)


def train_conv_vanilla_model_cifar100(num_epochs, save_result=False, create_tensorboard=False):
    train_images, train_labels, test_images, test_labels = load_full_cifar100()
    model = create_conv_vanilla_model_functional(train_images.shape[1:], 100)

    callbacks = []
    if save_result:
        # Create a callback that saves the model's weights
        checkpoint_path = "training_3/cp.ckpt"
        cp_callback = create_cp_callback(checkpoint_path)
        # Append to callbacks
        callbacks.append(cp_callback)

    if create_tensorboard:
        # create tensorboard
        # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        log_dir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
        makedirs(log_dir)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch')
        # Append to callbacks
        callbacks.append(tensorboard_callback)

    model.fit(
        train_images,
        train_labels,
        epochs=num_epochs,
        callbacks=callbacks
    )

