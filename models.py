from tensorflow import keras
from tensorflow.nn import softmax, relu


def create_conv_vanilla_model_functional(input_shape, num_outputs):

    inputs = keras.Input(shape=input_shape, dtype='float64')
    #x = keras.layers.Conv2D(9, 3, padding='same', input_shape=(28, 28, 1))(inputs)
    #x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    #x = keras.layers.Conv2D(18, 3, padding='same', input_shape=(14, 14, 3))(x)
    #x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = keras.layers.Conv2D(25, 3, padding='same')(inputs)
    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(50, 3, padding='same')(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(100, 3, padding='same')(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(200, 3, padding='same')(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    # x1 = x[:, :, :, 0:1]
    # x2 = x[:, :, :, 1:2]
    # x3 = x[:, :, :, 2:]
    # x1 = keras.layers.Conv2D(3, 3, padding='same', input_shape=(28, 28, 1))(x1)
    # x2 = keras.layers.Conv2D(3, 3, padding='same', input_shape=(28, 28, 1))(x2)
    # x3 = keras.layers.Conv2D(3, 3, padding='same', input_shape=(28, 28, 1))(x3)
    # x = tf.concat([x1, x2, x3], axis=3)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(300, activation=relu)(x)
    #x = keras.layers.Dense(300, activation=relu)(x)
    outputs = keras.layers.Dense(num_outputs, activation=softmax)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='VanillaConvModel')

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=[keras.metrics.sparse_categorical_accuracy])

    model.summary()

    return model


def create_dense_vanilla_model_functional():
    inputs = keras.Input(shape=(28, 28), dtype='float64')

    x = keras.layers.Flatten(input_shape=(28, 28))(inputs)
    dense_layer = keras.layers.Dense(128, activation="relu")
    x = dense_layer(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(10, activation=softmax)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='VanillaFuncModel')

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=keras.metrics.sparse_categorical_accuracy)
    model.summary()
    return model


def create_dense_vanilla_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=relu),
        keras.layers.Dense(128, activation=relu),
        keras.layers.Dense(10, activation=softmax)
    ])

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=keras.metrics.sparse_categorical_accuracy)
    model.summary()
    return model
