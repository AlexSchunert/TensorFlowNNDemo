import pandas as pd
#from sklearn.model_selection import train_test_split
from tensorflow import cast as tf_cast, float32 as tf_float32, data as tf_data
from tensorflow_datasets import load as tf_load
from keras.models import Sequential as keras_Sequential
from keras.layers import Dense as keras_Dense, Flatten as keras_Flatten
from keras.optimizers import Adam as keras_Adam
from keras.losses import SparseCategoricalCrossentropy as keras_SCE
from keras.metrics import SparseCategoricalAccuracy as keras_SCA

from time import time
from tensorflow.python.keras.callbacks import TensorBoard

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf_cast(image, tf_float32) / 255., label


(ds_train, ds_test), ds_info = tf_load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Training Data
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf_data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf_data.AUTOTUNE)

# Test Data
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf_data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf_data.AUTOTUNE)

# Create and train the model
model = keras_Sequential([
  keras_Flatten(input_shape=(28, 28)),
  keras_Dense(128, activation='relu'),
  keras_Dense(10)
])

# create tensorboard
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))


model.compile(
    optimizer=keras_Adam(0.001),
    loss=keras_SCE(from_logits=True),
    metrics=[keras_SCA()],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
    callbacks=[tensorboard],
)

print("test")