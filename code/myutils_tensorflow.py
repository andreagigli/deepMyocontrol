import matplotlib.pyplot as plt
import numpy as np
import warnings


# region dataset manipulation

def create_sliding_window_generator(data, label, w_len, w_stride=1):
    """ Implements sliding window by defining a generator over data and label.

    Creates a generator that produces sliding windows along the first dimension
    of data and label. Data and label are numpy arrays, w is the length of the window.

    https://sknadig.dev/TensorFlow2.0-dataset/unfortunately for examples on how
    to use tf.data.Dataset.from_generator().

    Important note: although the generator stops correctly before producing
    out-of-bound indices, when this function is used to generate databases
    you will receive a "warning" from tensorflow "tensorflow/core/kernels/data/
    generator_dataset_op.cc:103] Error occurred when finalizing GeneratorDataset
    iterator: Cancelled: Operation was cancelled". This is harmless, and the
    dataset will contain all the elements that one would expect in it.
    """

    n_windows = int((len(data) - w_len) / w_stride) + 1
    for i in range(n_windows):
        # anytime the generator is used, i is incremented by 1, a window of indices is
        # computed as idx_current_window = [i, i+1, i+2, ..., i+w] and the indexed data
        # is yielded data[idx_current_window], label[idx_current_window]
        idx = np.arange(i*w_stride,i*w_stride+w_len)
        yield data[idx], label[idx[-1]]


# endregion


# region dataset info


def get_db_elems_labels(dataset, discard_last_batch=True):
    """ Return elements and labels of a Tensorflow dataset as numpy arrays.

    It works both for un-batched and batched datasets.
    If the dataset does not contain labels, returns only the elements.
    """

    if len(dataset.as_numpy_iterator().next()) < 2:
        labels = np.array([])
        warnings.warn("The dataset does not contain labels")
        labels = []
        for _, y in dataset:  # only take first element of dataset
            labels.append(y.numpy())
        labels = np.vstack(labels)
        return labels

    if len(dataset.as_numpy_iterator().next()) == 2:
        elems = []
        labels = []
        for x, y in dataset:  # only take first element of dataset
            elems.append(x.numpy())
            labels.append(y.numpy())
        elems = np.vstack(elems)
        labels = np.vstack(labels)
        return elems, labels


def get_db_shape(dataset, batched=False):
    """ Computes the number of elements and their shape.
    Returns dataset shape as (num_el, shape each element).

    Different computation for batched or unbatched dataset.
    """

    if batched:
        num_el = 0
        for el in dataset:
            num_el += el[0].shape[0]
        el = next(dataset.as_numpy_iterator())
        el_size = (el[0][0].shape, el[1][0].shape)
        dataset_shape = ((num_el,) + el_size[0], (num_el,) + el_size[1])
    else:  # unbatched dataset
        num_el = 0
        for _ in dataset.as_numpy_iterator():
            num_el = num_el + 1
        first_el = next(dataset.as_numpy_iterator())
        dataset_shape = [(num_el,) + first_el[0].shape]  # shape x
        if len(first_el) == 2:
            dataset_shape.append((num_el,) + first_el[1].shape)  # shape y

    return dataset_shape


# endregion


# region plotting

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

# endregion