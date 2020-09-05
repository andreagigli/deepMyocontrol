import matplotlib.pyplot as plt
import numpy as np
import warnings


def create_sliding_window_generator(data, label, w):
    """ Implements sliding window by defining a generator over data and label.

    Creates a generator that produces sliding windows along the first dimension
    of data and label. Data and label are numpy arrays, w is the length of the window.

    https://sknadig.dev/TensorFlow2.0-dataset/unfortunately for examples on how
    to use tf.data.Dataset.from_generator()
    """

    for i in range(len(data) - w + 1):
        # anytime the generator is used, i is incremented by 1, a window of indices is
        # computed as idx_current_window = [i, i+1, i+2, ..., i+w] and the indexed data
        # is yielded data[idx_current_window], label[idx_current_window]
        yield data[np.arange(i, i + w)], label[i + w - 1]


def plot_history(history, metrics=None, plot_validation=False):
    """ Plots the fitting history of a Tensorflow model.

    Produces one subplot for the loss and one for all the existing metrics.
    If plot_validation=True, then also plot the corresponding val loss and metrics.
    """

    if metrics is None:
        metrics = []
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    for metric in metrics:
        plt.plot(history.history[metric], label=metric)
        if plot_validation:
            plt.plot(history.history["val_" + metric], label="val_" + metric)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Metric")
    ax1.legend(loc='lower right')
    ax2 = fig.add_subplot(212)
    plt.plot(history.history["loss"], label="loss")
    if plot_validation:
        plt.plot(history.history["val_loss"], label="val_loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc='lower right')

    return fig


def get_db_labels(dataset):
    """ Return the labels of a Tensorflow dataset as numpy array.

    It works both for un-batched and batched datasets.
    If the dataset does not contain labels, then return an empty array.
    """

    if len(dataset.as_numpy_iterator().next()) != 2:
        labels = np.array([])
        warnings.warn("The dataset does not contain labels")
    else:
        labels = []
        for _, y in dataset:  # only take first element of dataset
            labels.append(y.numpy())
        labels = np.vstack(labels)

    return labels


def scatter_1d_pred(y_true, y_pred):
    """ Plots 1-dim y_true against 1-dim y_pred as a 2-dim scatter plot.

    y_true and y_pred are numpy arrays (flattened or not).
    """

    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same number of " \
                                       "elements "
    fig = plt.figure()
    fig.axes(aspect='equal')
    plt.scatter(y_true[:, 0], y_pred[:, 0])
    plt.xlabel('True Values')
    plt.ylabel('Predictions')

    return plt
