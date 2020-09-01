import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

import matplotlib.pyplot as plt
import os
import pandas as pd
import datetime

########################## tf.keras ##########################
# tf.keras (different from the standalone keras that will be discontinued)
# wraps most deep-learning functionalities of tensorflow in a high-level API.

# tf.keras.Sequential allows to implement models made of stacked layers, each
# layer having 1 input and 1 output tensors. A sequential model is a stack of
# layers where the output of one layer is fed to the following one.
# Docs https://www.tensorflow.org/api_docs/python/tf/keras/Sequential,
# tutorials https://www.tensorflow.org/guide/keras/sequential_model.

# NOTE: For more complicated topologies (that is, graphs that are not perfectly
# stratified) use the tf.keras functional API. Complex topologies involve
# multiple input-output layers, layer-sharing, nonlinear topologies (residual
# nets = with skip-layer connections).
# The general deep learning keras model can be implemented with tf.keras.Model.
# Docs https://www.tensorflow.org/api_docs/python/tf/keras/Model,
# tutorials https://www.tensorflow.org/guide/keras/functional.


# region keras_sequential

###############################################################################
########################### tf.keras.Sequential ###############################
###############################################################################


# CREATION OF SEQUENTIAL MODEL
model = keras.Sequential(name="my_sequential")
# add input object. It's not a layer, it's just used to tell the model what's
# the input dimension, and therefore to compute the number of weights in the
# net. Instead of such object you can add a parameter input_shape=(4,) to the
# first layer.
# In case the input dimension is not known in advance, one can omit this info
# and the number of weights will be computed when the net will first see the
# input.
model.add(keras.Input(shape=(4,)))
# stack the layers on top of each other.
model.add(layers.Dense(2, activation="relu", name="layer1"))
model.add(layers.Dense(3, activation="relu", name="layer2"))
model.add(layers.Dense(4, name="layer3"))
#
# alternative ways to stack layers are:
#
# model = keras.Sequential(
#     [
#         layers.Dense(2, activation="relu"),
#         layers.Dense(3, activation="relu"),
#         layers.Dense(4),
#     ]
# )
#
# or having separate layers
#
# Create 3 layers
# layer1 = layers.Dense(2, activation="relu", name="layer1")
# layer2 = layers.Dense(3, activation="relu", name="layer2")
# layer3 = layers.Dense(4, name="layer3")
# x = tf.ones((3, 3))
# y = layer3(layer2(layer1(x)))
#
model.add(layers.Dense(5, name="layer4", activation="sigmoid"))
# note on number of parameters: if Dense(.., use_bias=true) then the output of
# the neuron is activation(dot(input,kernel)+bias) where kernel are the weights
# and bias is a tunable scalar. Therefore, the number of parameters such dense
# layer will be num_input * num_weights + num_neurons (num_neurons corresponds
# to having one bias scalar for each neuron).


# To VISUALIZE AND DEBUG THE TOPOLOGY, print the "summary" or render the graph
# in an image (saved but not shown). Even multiple times throughout the model
# building!
model.summary()
keras.utils.plot_model(model, model.name + ".png", show_shapes=True)

# To use a model for FEATURE EXTRACTION we can create another model
# and specify that its input is the input of the previous model, while it's
# output is the output of the previous model.
feature_extractor = keras.Model(
    inputs=model.inputs,
    outputs=model.get_layer(name="layer3").output,
)
# you can even return multiple levels' output as feature
# feature_extractor = keras.Model(
#     inputs=model.inputs,
#     outputs=[layer.output for layer in initial_model.layers],
# )
# Call feature extractor on test input.
x = tf.ones((1, 4))
features = feature_extractor(x)

# To perform TRASNFER LEARNING (fine-tuning) we need to freeze the bottom
# layers (that is the first layers) and leave the top ones trainable. We can
# do this by either (1) freezing each layer manually except for the top ones, or
# (2) freeze the whole pretrained model m and nest it into another model m' where
# m is stacked to some trainable layers, that is m'=sequential([m,trainable-layers]).
# The second technique is necessary specially when we use complex pre-available
# architectures.
#
#
# (1)
model = keras.Sequential([
    keras.Input(shape=(784)),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10),
])
## Presumably you would want to first load pre-trained weights.
# model.load_weights(...)
# Freeze all layers except the last one.
for layer in model.layers[:-1]:
    layer.trainable = False
# # Recompile and train (this will only update the weights of the last layer).
# model.compile(...)
# model.fit(...)
#
#
# (2)
# Load a convolutional base with pre-trained weights
base_model = keras.applications.Xception(
    weights='imagenet',
    include_top=False,
    pooling='avg')
# Freeze the base model
base_model.trainable = False
# Use a Sequential model to add a trainable classifier on top
model = keras.Sequential([
    base_model,
    layers.Dense(1000),
])
# # Compile & train
# model.compile(...)
# model.fit(...)
model.summary()

# endregion


# region keras_functional

###############################################################################
######################### tf.keras functional API #############################
###############################################################################

# CREATING ARCHITECTURE WITH FUNCTIONAL API
# notice python's CALLABLE SYNTAX: the syntax x=Dense(...)(x) indicates that
# the object obj=Dense(...) is callable,
# meaning that it can be also called as a function by giving it proper
# arguments. Therefore x=Dense(...)(x) is equivalent to obj=Dense(...) followed
# by obj(x). The callable tells the object to use the argument as input.
#
input = keras.Input(shape=(784,))
x = layers.Dense(64, activation="relu")(input)
x = layers.Dense(64, activation="relu")(x)
output = layers.Dense(10)(x)
model = keras.Model(inputs=input, outputs=output, name="my_model")
model.summary()
keras.utils.plot_model(model, model.name + ".png", show_shapes=True)

# TRAINING/TESTING tf.keras.Model

# TRAINING LOSS AND VALIDATION LOSS
# The training loss is used for backpropagation, whereas the validation loss
# is used as a stopping criterion or for monitoring purposes. At the end of
# every training epoch the training loss is computed on the training batch
# while the validation loss is computed on the whole validation set. By
# comparing these 2 losses we can get an insight on the model fitting:
# * train loss == val loss -> good fitting without overfitting
# * train loss << val loss -> underfitting
# * train loss >> val loss -> overfitting
# NOTE: although we expect the train loss to be lower than the val loss, it can
# happen to see train loss > val loss; this can be due to the regularization
# (dropout) that is taking place during training but not during prediction
# (validation is, in fact, prediction). Tensorflow provides another metrics,
# regularization loss, that corresponds to the difference between the training
# loss without regularization (dropout) and the training loss with regularization.
# In general, if good fitting, we should observe that
# shifted by 1/2 epoch(training loss + regulariz loss) = val loss. The shifting
# is to account for the fact that the val loss is computed at the end of the
# epoch, while the other losses are computed for each sample in the bach (therefore,
# on average, 1/2 epoch before the validation loss). See post from A. Geron
# https://twitter.com/aureliengeron/status/1110839223878184960?lang=en
#
#
# TEMPLATE for (using the model defined before) generating DB, training and
# testing the model. Store the history of train and val loss during training.
# Return the evaluation metrics.
#
# DEFINE MODEL (before).
#
# GENERATE DB
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255
# COMPILE MODEL
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)
#
# FIT + STATS
model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)
# alternatively you can fetch the history as the return of fit: history = model.fit()
print("Epochs: " + str(model.history.epoch))
print("Train loss history: " + str(model.history.history['loss']))
print("Train accuracy history: " + str(model.history.history['accuracy']))
print("Val loss history: " + str(model.history.history['val_loss']))
print("Val accuracy history: " + str(model.history.history['val_accuracy']))
# more/less metrics may be available, depending on the arguments of fit()
#
# TEST + STATS
test_scores = model.evaluate(x_test, y_test, verbose=2)
print("The metrics used for this model are: " + str(model.metrics_names))
print("Test " + model.metrics_names[0] + ": ", test_scores[0])
print("Test " + model.metrics_names[1] + ": ", test_scores[1])
#
# DO PREDICTIONS
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)

# SERIALIZATION
# Store the architecture, trained weights, training config, state of the optimizer
model.save("path_to_my_model")
del model
model = keras.models.load_model("path_to_my_model")

# ORDER OF CHANNELS
# When processing images:
# data_format='channels_first' -> batch_shape + (channels, rows, cols)
# data_format='channels_last' -> batch_shape + (rows, cols, channels)
tf.keras.backend.image_data_format()

# COMBINING MODELS using functional API.
# https://www.tensorflow.org/guide/keras/functional#all_models_are_callable_just_like_layers
#
# AUTOENCODER = concatenation of encoder and decoder
encoder_input = keras.Input(shape=(28, 28, 1), name="img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)
encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()
#
# decoder
decoder_input = keras.Input(shape=(16,), name="encoded_img")
x = layers.Reshape((4, 4, 16))(decoder_input)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)
decoder = keras.Model(decoder_input, decoder_output, name="decoder")  # decoder
decoder.summary()
#
# autoencoder = decoder(encoder(input))
autoencoder_input = keras.Input(shape=(28, 28, 1), name="img")
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = keras.Model(autoencoder_input, decoded_img, name="autoencoder")
autoencoder.summary()


#
#
# ENSEMBLE MODEL
def get_model():  # just generate a dummy model
    inputs = keras.Input(shape=(128,))
    outputs = layers.Dense(1)(inputs)
    return keras.Model(inputs, outputs)


model1 = get_model()
model2 = get_model()
model3 = get_model()
#
inputs = keras.Input(shape=(128,))
y1 = model(inputs)
y2 = model(inputs)
y3 = model(inputs)
outputs = layers.average([y1, y2, y3])  # combine predictions
ensemble_model = keras.Model(inputs=inputs, outputs=outputs)


# CREATE COMPLEX TOPOLOGIES
#
# MULTIPLE-INPUT MULTIPLE-OUTPUT MODEL
# https://www.tensorflow.org/guide/keras/functional#models_with_multiple_inputs_and_outputs
#
# RESIDUAL NETWORKS
# https://www.tensorflow.org/guide/keras/functional#a_toy_resnet_model
#
# SHARED LAYERS
# https://www.tensorflow.org/guide/keras/functional#shared_layers


# CUSTOM LAYERS defined through the functional API
# It's enough to extend the class tf.keras.layers.Layer by defining custom
# build() (create weights), call() (forward pass), get_config() and from_config() (serialize).
# DOCS: https://www.tensorflow.org/guide/keras/functional#extend_the_api_using_custom_layers
# https://www.tensorflow.org/guide/keras/custom_layers_and_models

# endregion


# region compile_train_evaluate_dbrepresentation

# TRAINING AND EVALUATION GUIDE
# https://www.tensorflow.org/guide/keras/train_and_evaluate/
#
#
# COMPILING a model requires defining
# * the OPTIMIZER https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
# * the LOSS  https://www.tensorflow.org/api_docs/python/tf/keras/losses
# * (optionally) a list of METRICS to evaluate how well the model is doing https://www.tensorflow.org/api_docs/python/tf/keras/metrics.
# Note: The loss is computed for each batch, whereas the metric is evaluated at the
# end of each epoch (https://www.tensorflow.org/js/guide/train_models).
#
# model.compile(
#     optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
#     loss=keras.losses.SparseCategoricalCrossentropy(),
#     metrics=[keras.metrics.SparseCategoricalAccuracy()],
# )
#
# Specify a CUSTOM LOSS by extending the tf.keras.losses.Loss class. In
# __init__() pass all the parameters that the loss needs (if any), while in
# call() pass only y_true and y_pred. Then just use it in compile()
class CustomMSE(keras.losses.Loss):
    def __init__(self, name="custom_mse", regularization_factor=0.1):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        return mse + reg * self.regularization_factor


model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=CustomMSE(regularization_factor=0.01),
)


#
# Specify a CUSTOM METRIC by extending tf.keras.metrics.Metric. This is quite
# harder, just refer to https://www.tensorflow.org/guide/keras/train_and_evaluate#custom_metrics
class CategoricalTruePositives(keras.metrics.Metric):
    def __init__(self, name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
        values = tf.cast(values, "float32")
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.0)


model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[CategoricalTruePositives()],
)
#
# If the loss or metrics do not follow the standard signature (a factor of the
# loss or one metric are computed on something else than y_true-ypred) use the
# functions model.add_loss(loss_tensor) or model.add_metric(metric_tensor, name, aggregation).
# Note: these functions add, respectively, a term to the total loss (that still needs to
# be declared in compile()!) and a new metric to the list of metrics.
# Example: loss has a term 0.1*sum(activation of layer i) and one metric is the average
# std of the activation of layer i over the whole epoch.
inputs = keras.Input(shape=(784,), name="digits")  # just to define a model
x1 = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x2 = layers.Dense(64, activation="relu", name="dense_2")(x1)
outputs = layers.Dense(10, name="predictions")(x2)
model = keras.Model(inputs=inputs, outputs=outputs)
model.add_loss(tf.reduce_sum(x1) * 0.1)  # add loss term
model.add_metric(keras.backend.std(x1), name="std_of_activation",
                 aggregation="mean")  # declare metric
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[]  # the list of metrics already contains "std_of_activation"
)
#
#
# Model FITTING
# (https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit)
# Example:
# history = model.fit(
#     x_train,
#     y_train,
#     batch_size=64,
#     epochs=2,
#     validation_data=(x_val, y_val),
# )
#
# HOW TO REPRESENT DATABASE IN MEMORY
# x can be a tensor, a tf.data.Dataset, or a generator.
#
# 1) whole dataset in memory as tensor
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]
# model.fit(x_train, y_train, batch_size=64, epochs=2, validation_data=(x_val, y_val))
#
# 2) create tf.data.Database object by slicing existing tensors along the first dimension
x = x_train  # pretend all db is in x_train
y = y_train
dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = dataset.skip(1000)  # from 1000 to end
test_dataset = dataset.take(1000)  # from 0 to 999
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)  # the batching must be done by the dataset object and not by the fit() method.
# model.fit(train_dataset, epochs=3)
# model.fit(train_dataset, epochs=3, steps_per_epoch=100) # in case you want to just use 100 batches in each epoch (the next epoch uses the remaining batches)
#
# print the dataset elements and element components. A dataset contains "elements", each
# of which has the same structure. Each element consists of one or multiple "element components".
dataset = tf.data.Dataset.from_tensor_slices([[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15], [16,17,18]])
print(dataset.element_spec) # prints the type of the element components
list(dataset.as_numpy_iterator()) # (1) convert to list. WARNING! it loads the whole dataset in memory!
list(dataset.as_numpy_iterator())[0] # (2) convert to list and get a specific element
for element in dataset.take(2):  # (3) iterate over it (it does not load the whole dataset in memory)
    print(element)
iterator = dataset.as_numpy_iterator() # (4) convert to iterator
iterator.next()
# for element, label in dataset.take(1): # (if the dataset has got labels)
#     print(label)
# a "batched" database iterates over the batches
dataset = tf.data.Dataset.from_tensor_slices([[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15], [16,17,18]])
dataset = dataset.batch(2)
list(dataset.as_numpy_iterator())[0] # first element = first batch
list(dataset.as_numpy_iterator())[0][1] # second component of the first element
print(dataset.element_spec)
#
# note the difference between slicing lists, tuples and numpy arrays
dataset = tf.data.Dataset.from_tensor_slices([[1,2,3], [4,5,6]])
list(dataset.as_numpy_iterator())
dataset = tf.data.Dataset.from_tensor_slices(([1,2,3], [4,5,6]))
list(dataset.as_numpy_iterator())
dataset = tf.data.Dataset.from_tensor_slices(np.array([[1,2,3], [4,5,6]]))
list(dataset.as_numpy_iterator())
#
# 3) create tf.data.Database object from files (text, csv, sets of files)
# https://www.tensorflow.org/guide/data#reading_input_data.
# For example create image db from nested folders, each folder representing one class.
# The name of the folder is the class label.
flowers_root = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)
import pathlib
import os
flowers_root = pathlib.Path(flowers_root)
dataset = tf.data.Dataset.list_files(str(flowers_root/'*/*'))
def parse_image(filename):
  parts = tf.strings.split(filename, os.sep)
  label = parts[-2]
  image = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(image)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [128, 128])
  return image, label
labeled_dataset = dataset.map(parse_image)
for element, label in labeled_dataset.take(1): # (if the dataset has got labels)
    print(label)
#
# 4) create dataset from a generator. See https://sknadig.dev/TensorFlow2.0-dataset/
# It comes handy when you need to generate the db on the fly, or the dataset elements are
# different views of the same data that you don't want to replicate (e.g. moving window).
#
#
# HOW TO APPLY AN OPERATION TO ALL THE ELEMENTS OF THE DATASET
# use dataset.map(function)
dataset = tf.data.Dataset.from_tensor_slices([[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15], [16,17,18]])
list(dataset.as_numpy_iterator())
def tensor_sum(x):
    return tf.math.reduce_sum(x)
dataset = dataset.map(tensor_sum)
list(dataset.as_numpy_iterator())
#
#
# VALIDATION SET
# When the training set is provided explicitely, the validation data can be
# computed implicitely
model.fit(x_train, y_train, batch_size=64, epochs=2,
          validation_split=0.2)  # 20% of the training data, before shuffling
# or passed explicitely
model.fit(x_train, y_train, batch_size=64, epochs=2, validation_data=(x_val, y_val))
# or passed explicitely as a Dataset object
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(64)
model.fit(train_dataset, epochs=1, validation_data=val_dataset)
# Important note: The validation set CAN NOT BE DEFINED WHEN USING tf.data.Dataset
# or generators! Either you define explicitely the validation samples
# or you provide the whole training set to fit and declare validation_split.
#
#
# BATCHING AND PADDED BATCHING
#
# Normal batching dataset.batch(64). Note that it does not return a shape because the
# last batch can have less elements than expected. One can drop the last batch with the
# argument drop_remainder=True.
#
# Padded batching allows to pad all the samples in a batch so that they have the same
# dimension. (It doesn't add virtual samples, but pads the existing ones!).
dataset = tf.data.Dataset.range(20)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)],
                                        x))  # example dataset: [(1),(2,2),(3,3,3),(4,4,4,4),...]
dataset = dataset.padded_batch(4, padded_shapes=(None,))
for batch in dataset.take(2):
    print(batch.numpy())
    print()

# endregion


# learn about TensorBoard https://www.tensorflow.org/guide/keras/train_and_evaluate#visualizing_loss_and_metrics_during_training
# Database creation pipeline https://www.tensorflow.org/guide/data#time_series_windowing
# Example of time series forecasting https://www.tensorflow.org/tutorials/structured_data/time_series







# region dataset windowing
###############################################################################
############################# DATASET WINDOWING ###############################
###############################################################################
# Requires adaptation because tf.keras.preprocessing.timeseries_dataset_from_array
# is only available in nightly release, which is unstable. Timeseries.py was downloaded
# from https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/preprocessing/timeseries.py#L29-L199
# and added to the path.
#
# The idea is to create an window generator object that takes the data and the
# the characteristics of the desired window and creates a windowed db (internally)
#
# The core is the function make_dataset that obtains windowed dataset from the data
# and applies the function split_window to all the elements. Split data is necessary
# only if the prediction is one (future) element of the time series.
# I guess split_data can be adapted to take as target one specific value of the
# targets array (value corresponding to the head of the window).
#
# The "calls" procede this way: create w2=WindowGenerator(...window props, data) object, then
# call w2.train -> calls make_dataset on the training set applying split_window to each
# element.




# load pandas a database
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)
# slice [start:stop:step], starting from index 5 take every 6th record.
df = df[5::6]
date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
df.head()
df.describe().transpose()
wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0
max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0
# The above inplace edits are reflected in the DataFrame
timestamp_s = date_time.map(datetime.datetime.timestamp)
column_indices = {name: i for i, name in enumerate(df.columns)}
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]
num_features = df.shape[1]

# create window generator class
class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
          self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
          inputs = features[:, self.input_slice, :]
          labels = features[:, self.labels_slice, :]
          if self.label_columns is not None:
              labels = tf.stack(
                  [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                  axis=-1)

          # Slicing doesn't preserve static shape information, so set the shapes
          # manually. This way the `tf.data.Datasets` are easier to inspect.
          inputs.set_shape([None, self.input_width, None])
          labels.set_shape([None, self.label_width, None])

          return inputs, labels

    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
          inputs, labels = self.example
          plt.figure(figsize=(12, 8))
          plot_col_index = self.column_indices[plot_col]
          max_n = min(max_subplots, len(inputs))
          for n in range(max_n):
              plt.subplot(3, 1, n + 1)
              plt.ylabel(f'{plot_col} [normed]')
              plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                       label='Inputs', marker='.', zorder=-10)

              if self.label_columns:
                  label_col_index = self.label_columns_indices.get(plot_col, None)
              else:
                  label_col_index = plot_col_index

              if label_col_index is None:
                  continue

              plt.scatter(self.label_indices, labels[n, :, label_col_index],
                          edgecolors='k', label='Labels', c='#2ca02c', s=64)
              if model is not None:
                  predictions = model(inputs)
                  plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                              marker='X', edgecolors='k', label='Predictions',
                              c='#ff7f0e', s=64)

              if n == 0:
                  plt.legend()

          plt.xlabel('Time [h]')

          return


    def make_dataset(self, data):
          data = np.array(data, dtype=np.float32)
          from timeseries import timeseries_dataset_from_array # timeseries.py added manually in the path
          ds = timeseries_dataset_from_array( #tf.keras.preprocessing.timeseries_dataset_from_array(
              data=data,
              targets=None,
              sequence_length=self.total_window_size,
              sequence_stride=1,
              shuffle=True,
              batch_size=32, )

          ds = ds.map(self.split_window)

          return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
          """Get and cache an example batch of `inputs, labels` for plotting."""
          result = getattr(self, '_example', None)
          if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
          return result

# instantiate a window generator object with the characteristics of the desired window
# w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
#                      label_columns=['T (degC)'], train_df=train_df,
#                      test_df=test_df, val_df=val_df)
# w1
# w3 = WindowGenerator(input_width=6, label_width=1, shift=0,
#                      label_columns=['T (degC)'], train_df=train_df,
#                      test_df=test_df, val_df=val_df)
# w3

w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                     label_columns=['T (degC)'], train_df=train_df,
                     test_df=test_df, val_df=val_df)
w2

# example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
#                            np.array(train_df[100:100+w2.total_window_size]),
#                            np.array(train_df[200:200+w2.total_window_size])])
# example_inputs, example_labels = w2.split_window(example_window)
# w2.example = example_inputs, example_labels

# create the train/val/test sets within the window generator object
w2.train
w2.val
w2.test
w2.example

for example_inputs, example_labels in w2.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')

# use the windowed data (for example) in a cnn. More example are available here
# https://www.tensorflow.org/tutorials/structured_data/time_series
CONV_WIDTH = 3
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    label_columns=['T (degC)'],
    train_df=train_df,
    test_df=test_df,
    val_df=val_df
)
conv_window
conv_window.plot()
plt.title("Given 3h as input, predict 1h into the future.")

conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', multi_step_dense(conv_window.example[0]).shape)

MAX_EPOCHS = 2

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

history = compile_and_fit(multi_step_dense, conv_window)

import IPython
val_performance = {}
performance = {}
class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

baseline = Baseline(label_index=column_indices['T (degC)'])

baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=['T (degC)'],
    train_df=train_df,
    test_df=test_df,
    val_df=val_df)
single_step_window
val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)

IPython.display.clear_output()
val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)

conv_window.plot(multi_step_dense)

wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    label_columns=['T (degC)'],
    train_df=train_df,
    test_df=test_df,
    val_df=val_df)

wide_window
print('Input shape:', wide_window.example[0].shape)
try:
  print('Output shape:', multi_step_dense(wide_window.example[0]).shape)
except Exception as e:
  print(f'\n{type(e).__name__}:{e}')

LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
  input_width=INPUT_WIDTH,
  label_width=LABEL_WIDTH,
  shift=1,
  label_columns=['T (degC)'],
    train_df=train_df,
    test_df=test_df,
    val_df=val_df)

wide_conv_window

print("Wide conv window")
print('Input shape:', wide_conv_window.example[0].shape)
print('Labels shape:', wide_conv_window.example[1].shape)
print('Output shape:', conv_model(wide_conv_window.example[0]).shape)

wide_conv_window.plot(conv_model)

# endregion