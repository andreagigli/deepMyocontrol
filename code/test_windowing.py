import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import time

def createSlidingWindowGenerator(data,label,w):
    # returns generator that implements sliding window along the first dimension of data and label
    for i in range(len(data)-w+1):
        # generate sliding window of indices. By using "yield" i is incremented
        # (by 1 in this case) anytime the generator is used, the window
        # idx = [i, i+1, i+2, ..., i+w] is created, and the value data[idx] and
        # label[idx] is yielded.
        yield data[np.arange(i,i+w)], label[i+w-1]
def plotHistory(history, metrics = [], plot_validation=False):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    for metric in metrics:
        plt.plot(history.history[metric], label=metric)
        if plot_validation:
            plt.plot(history.history["val_"+metric], label="val_"+metric)
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
def getDbLabels(dataset):
    labels = []
    for _, y in test_expanded_shuffled:  # only take first element of dataset
        labels.append(y.numpy())
    return np.vstack(labels)


# logs
NAME = "toy_windowing_CNN_{}".format(int(time.time()))

# moving window using Dataset.from_generator
# examples of how to use from_generator here https://sknadig.dev/TensorFlow2.0-dataset/
# unfortunately, I found no reference of the moving window approach online, cannot
# guarantee about the correctness of this approach.
DATASET_SIZE = 10000
x, y = make_regression(n_samples=DATASET_SIZE, n_features=3, n_targets=2, random_state=1)
# x = np.hstack((np.arange(1,10001,3)[:,None],
#                np.arange(2,10002,3)[:,None],
#                np.arange(3,10003,3)[:,None]))
# y = x[:,0:2]*10
print(x.shape)
print(y.shape)





dataset = tf.data.Dataset.from_generator(createSlidingWindowGenerator,
                                         output_types=(tf.float32, tf.float32),
                                         output_shapes=(tf.TensorShape([3,3]), tf.TensorShape([2])),
                                         args=[x,y,3])
train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.2 * train_size) # 30% of the training set
test_dataset = dataset.skip(train_size)
train_dataset = dataset.take(train_size)
val_dataset = train_dataset.take(val_size)
train_dataset = train_dataset.skip(val_size)

train_expanded = train_dataset.map(lambda x, y: (tf.expand_dims(x, axis=-1), y))
train_expanded_shuffled = train_expanded.shuffle(1000)
train_expanded_shuffled_batched = train_expanded_shuffled.batch(128)

test_expanded = test_dataset.map(lambda x, y: (tf.expand_dims(x, axis=-1), y))
test_expanded_shuffled = test_expanded.shuffle(1000)
test_expanded_shuffled_batched = test_expanded_shuffled.batch(128)

val_expanded = val_dataset.map(lambda x, y: (tf.expand_dims(x, axis=-1), y))
val_expanded_shuffled = val_expanded.shuffle(1000)
val_expanded_shuffled_batched = val_expanded_shuffled.batch(128)


model_input = keras.Input(shape=(3, 3, 1), name="img")
x = layers.Conv2D(16, 3, activation="relu")(model_input)
x = layers.Flatten()(x)
model_output = layers.Dense(2)(x)
model = keras.Model(inputs=model_input, outputs=model_output, name="my_cnn")
model.summary()

# using tensorboard to visualize the model's evolution in realtime
# instructions here https://pythonprogramming.net/tensorboard-analysis-deep-learning-python-tensorflow-keras/
# (use callback to tensorflow in fit, run an instance of tensorflow indicating the log folder,
# open the give url in the browser.)
tensorboard = TensorBoard(log_dir="logs\\{}".format(NAME))

model.compile(
    loss=tf.losses.MeanSquaredError(),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["mse"],
)


history = model.fit(x=train_expanded_shuffled_batched,
                    validation_data=val_expanded_shuffled_batched,
                    epochs=50,
                    callbacks=[tensorboard])
plotHistory(history, metrics=["mse"], plot_validation=True)


test_predictions = model.predict(test_expanded_shuffled_batched)
test_labels = getDbLabels(test_expanded_shuffled_batched)
test_mse = tf.keras.metrics.MSE(test_labels, test_predictions)
print(test_mse) # the two mse differ a little, possibly because the loss is not computed on the last batch (incomplete batch discarded)
test_loss, test_mse = model.evaluate(test_expanded_shuffled_batched, verbose=2)
print(test_mse)


# plot results as scatter plot
plt.figure()
plt.axes(aspect='equal')
plt.scatter(test_labels[:,0], test_predictions[:,0])
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.figure()
plt.axes(aspect='equal')
plt.scatter(test_labels[:,1], test_predictions[:,1])
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()




placeholder = 1




# # example of tf.data.Dataset.from_generator from https://sknadig.dev/TensorFlow2.0-dataset/
# def our_generator():
#     for i in range(1000):
#       x = np.random.rand(28,28)
#       y = np.random.randint(1,10, size=1)
#       yield x,y
#
# dataset = tf.data.Dataset.from_generator(our_generator, (tf.float32, tf.int16))
#
# # the dataset is generated at runtime. Anytime the generator is called, it produces
# # one element. The database ends when the generator is exhausted. However, you can
# # regenerate the database multiple times.
# print("num produced elements: ", len(list(dataset.as_numpy_iterator())))
# print("Data shape: ", next(dataset.as_numpy_iterator())[0].shape, next(dataset.as_numpy_iterator())[1].shape)
#
# # you can "repeat" the database n times to make sure that the generation is repeated n times
# dataset = tf.data.Dataset.from_generator(our_generator, (tf.float32, tf.int16))
# dataset = dataset.repeat(count=2) # -1 to repeat indefinitely
# print("num produced elements: ", len(list(dataset.as_numpy_iterator())))
# print("Data shape: ", next(dataset.as_numpy_iterator())[0].shape, next(dataset.as_numpy_iterator())[1].shape)
#
# # batching works as usual
# dataset = tf.data.Dataset.from_generator(our_generator, (tf.float32, tf.int16))
# dataset = dataset.batch(batch_size=10)
# print("num produced elements: ", len(list(dataset.as_numpy_iterator())))
# print("Data shape: ", next(dataset.as_numpy_iterator())[0].shape, next(dataset.as_numpy_iterator())[1].shape)
#
# iterator = dataset.as_numpy_iterator()
# x,y = next(iterator)
# print(x.shape, y.shape)
#
# # shuffling the database
# dataset = dataset.shuffle(buffer_size=1000) # nothing new





