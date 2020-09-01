import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import datetime

x = np.hstack((np.arange(1,101,3)[:,None],
               np.arange(2,102,3)[:,None],
               np.arange(3,103,3)[:,None]))
y = x[:,0:2]*10
x.shape


from timeseries import timeseries_dataset_from_array  # timeseries.py added manually in the path
dataset = timeseries_dataset_from_array(data=x, targets=y, sequence_length=4, sequence_stride=1)
el = dataset.as_numpy_iterator().next()
list(dataset.as_numpy_iterator())
for batch in dataset:
  inputs, targets = batch
# dataset = dataset.map(lambda x: tf.expand_dims(x, axis=0))

model_input = keras.Input(shape=(4, 3, 1), name="img")
x = layers.Conv2D(16, 3, activation="relu")(model_input)
model_output = layers.Dense(2)(x)
model = keras.Model(inputs=model_input, outputs=model_output, name="my_cnn")
model.summary()

model.compile(
    loss=tf.losses.MeanSquaredError(),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

model.fit(dataset, epochs=2)



# MAYBE THIS WORKS!!!!!!!!
x = np.hstack((np.arange(1,101,3)[:,None],
               np.arange(2,102,3)[:,None],
               np.arange(3,103,3)[:,None]))
y = x[:,0:2]*10
x.shape
y.shape

def createSlidingWindowGenerator(x,y,w):
    # returns generator that implements sliding window along the first dimension of v
    for i in range(len(x)-w+1):
        # yield: i is incremented (by 1 in this case) anytime the generator is used to generate a new value
        yield x[np.arange(i,i+w)], y[i+w-1]

xy_gen = createSlidingWindowGenerator(x,y,3)
for window in xy_gen:
    print(window)

xy_gen = createSlidingWindowGenerator(x,y,3)
list_windows = list(xy_gen)
list_windows[0]
list_windows[1]

xy_gen = createSlidingWindowGenerator(x,y,3)
next(xy_gen)

dataset = tf.data.Dataset.from_generator(createSlidingWindowGenerator, (tf.float32, tf.float32),args=[x,y,3])
iterator = dataset.as_numpy_iterator()
el = next(iterator)
print(el[0].shape)
print(el[1].shape)





# example of tf.data.Dataset.from_generator from https://sknadig.dev/TensorFlow2.0-dataset/
def our_generator():
    for i in range(1000):
      x = np.random.rand(28,28)
      y = np.random.randint(1,10, size=1)
      yield x,y

dataset = tf.data.Dataset.from_generator(our_generator, (tf.float32, tf.int16))

# the dataset is generated at runtime. Anytime the generator is called, it produces
# one element. The database ends when the generator is exhausted. However, you can
# regenerate the database multiple times.
print("num produced elements: ", len(list(dataset.as_numpy_iterator())))
print("Data shape: ", next(dataset.as_numpy_iterator())[0].shape, next(dataset.as_numpy_iterator())[1].shape)

# you can "repeat" the database n times to make sure that the generation is repeated n times
dataset = tf.data.Dataset.from_generator(our_generator, (tf.float32, tf.int16))
dataset = dataset.repeat(count=2) # -1 to repeat indefinitely
print("num produced elements: ", len(list(dataset.as_numpy_iterator())))
print("Data shape: ", next(dataset.as_numpy_iterator())[0].shape, next(dataset.as_numpy_iterator())[1].shape)

# batching works as usual
dataset = tf.data.Dataset.from_generator(our_generator, (tf.float32, tf.int16))
dataset = dataset.batch(batch_size=10)
print("num produced elements: ", len(list(dataset.as_numpy_iterator())))
print("Data shape: ", next(dataset.as_numpy_iterator())[0].shape, next(dataset.as_numpy_iterator())[1].shape)

iterator = dataset.as_numpy_iterator()
x,y = next(iterator)
print(x.shape, y.shape)

# shuffling the database
dataset = dataset.shuffle(buffer_size=1000) # nothing new





