import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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
model.add(layers.Dense(5, name="layer4",activation="sigmoid"))
# note on number of parameters: if Dense(.., use_bias=true) then the output of
# the neuron is activation(dot(input,kernel)+bias) where kernel are the weights
# and bias is a tunable scalar. Therefore, the number of parameters such dense
# layer will be num_input * num_weights + num_neurons (num_neurons corresponds
# to having one bias scalar for each neuron).



# To VISUALIZE AND DEBUG THE TOPOLOGY, print the "summary" or render the graph
# in an image (saved but not shown). Even multiple times throughout the model
# building!
model.summary()
keras.utils.plot_model(model, model.name+".png", show_shapes=True)



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
#model.load_weights(...)
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
keras.utils.plot_model(model, model.name+".png", show_shapes=True)



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
print("Train val loss history: " + str(model.history.history['val_loss']))
print("Train val accuracy history: " + str(model.history.history['val_accuracy']))
# more/less metrics may be available, depending on the arguments of fit()
#
# TEST + STATS
test_scores = model.evaluate(x_test, y_test, verbose=2)
print("The metrics used for this model are: " + str(model.metrics_names))
print("Test " + model.metrics_names[0] + ": ", test_scores[0])
print("Test " + model.metrics_names[1] + ": ", test_scores[1])



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
decoder = keras.Model(decoder_input, decoder_output, name="decoder") # decoder
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
def get_model(): # just generate a dummy model
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
outputs = layers.average([y1, y2, y3]) # combine predictions
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




placeholder = 1



# TRAINING AND EVALUATION GUIDE
# https://www.tensorflow.org/guide/keras/train_and_evaluate/