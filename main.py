import tensorflow as tf
import numpy as np
from tensorflow import keras
import datetime
from tensorflow.python.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
import os
from models import create_conv_vanilla_model_functional, create_dense_vanilla_model, \
    create_dense_vanilla_model_functional
from datasets import load_full_mnist
from models import create_conv_vanilla_model_functional

from training import train_conv_vanilla_model_mnist, train_conv_vanilla_model_cifar100
from inference import inference_conv_vanilla_model_mnist

train_conv_vanilla_model_cifar100(30, create_tensorboard=True, save_result=True)

#inference_conv_vanilla_model_mnist()

#train_conv_vanilla_model_mnist(2)
# evaluate_conv_vanilla_model()
# train_dense_vanilla_model()
# evaluate_dense_vanilla_model()

# print(tf.test.is_built_with_cuda())
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# cifar100 = keras.datasets.cifar100

# (train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
# print("tt")


