from tensorflow import keras


def load_full_mnist():
    fashion_mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return train_images, train_labels, test_images, test_labels

def load_full_cifar100():
    cifar100 = keras.datasets.cifar100
    (train_images, train_labels), (test_images, test_labels) = cifar100.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return train_images, train_labels, test_images, test_labels
