from models import create_dense_vanilla_model, create_conv_vanilla_model_functional
from datasets import load_full_mnist
from matplotlib import pyplot as plt


def evaluate_dense_vanilla_model_mnist():
    checkpoint_path = "training_1/cp.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    # latest_model = tf.train.latest_checkpoint(checkpoint_path)

    # Create a new model instance
    model = create_dense_vanilla_model()
    # Load
    model.load_weights(checkpoint_path)

    # Load mnist
    train_images, train_labels, test_images, test_labels = load_full_mnist()

    # Evaluate
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


def evaluate_conv_vanilla_model_mnist():
    # Load mnist
    train_images, train_labels, test_images, test_labels = load_full_mnist()

    # Create a new model instance
    #model = create_conv_vanilla_model_functional(test_images.shape[1:]+(1,), 10)
    # Load
    #checkpoint_path = "training_2/cp.ckpt"
    #model.load_weights(checkpoint_path)
    model = load_conv_vanilla_model("training_2/cp.ckpt", test_images.shape[1:] + (1,), 10)


    # Evaluate
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


def load_conv_vanilla_model(cp_path, input_shape, num_outputs):
    model = create_conv_vanilla_model_functional(input_shape, num_outputs)
    checkpoint_path = cp_path#
    model.load_weights(checkpoint_path)
    return model


def inference_conv_vanilla_model_mnist():
    train_images, train_labels, test_images, test_labels = load_full_mnist()
    model = load_conv_vanilla_model("training_2/cp.ckpt", test_images.shape[1:]+(1,), 10)
    for t_img, t_label in zip(test_images[0:10], test_labels[0:10]):
        plt.imshow(t_img)
        prediction = model.predict(t_img.reshape(1, 28, 28, 1))[0].argmax()
        print("Predicted: ", prediction, " Label: ", t_label)
        plt.show()
