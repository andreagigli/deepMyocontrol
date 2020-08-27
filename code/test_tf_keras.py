import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

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
#
#
# DATASET WINDOWING https://www.tensorflow.org/guide/data#time_series_windowing and https://www.tensorflow.org/tutorials/structured_data/time_series
#

# endregion


# learn about TensorBoard https://www.tensorflow.org/guide/keras/train_and_evaluate#visualizing_loss_and_metrics_during_training
# Database creation pipeline https://www.tensorflow.org/guide/data#time_series_windowing
# Example of time series forecasting https://www.tensorflow.org/tutorials/structured_data/time_series


placeholder = 1







import graspRegression as gr

fields = ["regrasp", "regrasprepetition", "reobject", "reposition", "redynamic",
          "emg", "ts"]
x_train, y_train, x_test, y_test, cv_splits = \
    gr.load_megane_database("D:\\Documenti\\PROJECTS\\20_logReg\\data\\S010_ex1.mat",
                             fields, train_reps=np.array([1,2,3]), test_reps=np.array([4]),
                             subsamplerate=1, feature_set=["raw"])
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# how the heck do I create images from windows in the dataset?
# maybe this can be helpful
# https://www.tensorflow.org/tutorials/structured_data/time_series#convolution_neural_network


myo_cnn_in = keras.Input(shape=(40, 40, 1), name="in")
x = layers.Conv2D(16, 3, activation="relu", padding="same", name="cnn1")(myo_cnn_in)
x = layers.BatchNormalization(name="bn1")(x)
x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="pool1")(x)
x = layers.Conv2D(16, 3, activation="relu", padding="same", name="cnn2")(x)
x = layers.BatchNormalization(name="bn2")(x)
x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="pool2")(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same", name="cnn3")(x)
x = layers.BatchNormalization(name="bn3")(x)
x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="pool3")(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same", name="cnn4")(x)
x = layers.BatchNormalization(name="bn4")(x)
x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="pool4")(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same", name="cnn5")(x)
x = layers.BatchNormalization(name="bn5")(x)
x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="pool5")(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same", name="cnn6")(x)
x = layers.BatchNormalization(name="bn6")(x)
x = layers.Conv2D(16, 3, activation="relu", padding="same", name="cnn7")(x)
x = layers.BatchNormalization(name="bn7")(x)
x = layers.Conv2D(16, 3, activation="relu", padding="same", name="cnn8")(x)
x = layers.BatchNormalization(name="bn8")(x)
x = layers.Dense(2, activation="relu", name="FC1")(x)
myo_cnn_out = layers.Dense(1, activation="linear", name="out")(x)
myo_cnn = keras.Model(myo_cnn_in, myo_cnn_out, name="myocnn")
myo_cnn.summary()

myo_cnn.compile(
    loss=keras.losses.mean_squared_error(),
    optimizer=keras.optimizers.RMSprop(),
    metrics=[keras.metrics.mean_squared_error()],
)

history = myo_cnn.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=30,
    validation_data=(x_val, y_val),
)






