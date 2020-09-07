import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import time

import myutils_tensorflow as tf_util


# logs
NAME = "toy_windowing_CNN_{}".format(int(time.time()))
DATASET_SIZE = 10000


# using tensorboard to visualize the model's evolution in realtime instructions here
# https://pythonprogramming.net/tensorboard-analysis-deep-learning-python-tensorflow-keras/
# (use callback to tensorflow in fit, run an instance of tensorflow from anaconda
# tensorboard --logdir = log_dir_path, open the given url in the browser.)
tensorboard = TensorBoard(log_dir="logs\\{}".format(NAME))

# create data
X, Y = make_regression(n_samples=DATASET_SIZE, n_features=3, n_targets=2, random_state=1)
print(X.shape)
print(Y.shape)

# create dataset with moving window using Dataset.from_generator
dataset = tf.data.Dataset.from_generator(generator=tf_util.create_sliding_window_generator(),
                                         output_types=(tf.float32, tf.float32),
                                         output_shapes=(
                                         tf.TensorShape([3, 3]), tf.TensorShape([2])),
                                         args=[X, Y, 3])

# split dataset
train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.2 * train_size)  # 30% of the training set
test_dataset = dataset.skip(train_size)
train_dataset = dataset.take(train_size)
val_dataset = train_dataset.take(val_size)
train_dataset = train_dataset.skip(val_size)

# reshape elements, shuffle them, batch them (batching required by CNN: 1-st dim = batch)
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

model.compile(
    loss=tf.losses.MeanSquaredError(),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["mse"],
)

history = model.fit(x=train_expanded_shuffled_batched,
                    validation_data=val_expanded_shuffled_batched,
                    epochs=5,
                    callbacks=[tensorboard])

tf_util.plot_history(history, metrics=["mse"], plot_validation=True)

# compute the mse (either model.evaluate() or tf.keras.metrics.chosen_metric())
# the two metrics may differ a little, because (?) the loss is not computed on
# the last batch (incomplete batch discarded)
test_predictions = model.predict(test_expanded_shuffled_batched)
test_labels = tf_util.get_db_labels(test_expanded_shuffled_batched)
test_mse = tf.keras.metrics.MSE(test_labels, test_predictions)
print(test_mse)
test_loss, test_mse = model.evaluate(test_expanded_shuffled_batched, verbose=2)
print(test_mse)

# plot results as scatter plot
tf_util.scatter_1d_pred(test_labels[:, 0], test_predictions[:, 0])
plt.show()

placeholder = 1
