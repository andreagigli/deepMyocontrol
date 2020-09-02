import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# moving window using Dataset.from_generator
# examples of how to use from_generator here https://sknadig.dev/TensorFlow2.0-dataset/
# unfortunately, I found no reference of the moving window approach online, cannot
# guarantee about the correctness of this approach.
x = np.hstack((np.arange(1,10001,3)[:,None],
               np.arange(2,10002,3)[:,None],
               np.arange(3,10003,3)[:,None]))
y = x[:,0:2]*10
x.shape
y.shape


def createSlidingWindowGenerator(data,label,w):
    # returns generator that implements sliding window along the first dimension of data and label
    for i in range(len(data)-w+1):
        # generate sliding window of indices. By using "yield" i is incremented
        # (by 1 in this case) anytime the generator is used, the window
        # idx = [i, i+1, i+2, ..., i+w] is created, and the value data[idx] and
        # label[idx] is yielded.
        yield data[np.arange(i,i+w)], label[i+w-1]

# xy_gen = createSlidingWindowGenerator(x,y,3)
# for window in xy_gen:
#     print(window)
#
# xy_gen = createSlidingWindowGenerator(x,y,3)
# list_windows = list(xy_gen)
# list_windows[0]
# list_windows[1]
#
# xy_gen = createSlidingWindowGenerator(x,y,3)
# next(xy_gen)

dataset = tf.data.Dataset.from_generator(createSlidingWindowGenerator,
                                         output_types=(tf.float32, tf.float32),
                                         output_shapes=(tf.TensorShape([3,3]), tf.TensorShape([2])),
                                         args=[x,y,3])
iterator = dataset.as_numpy_iterator()
el = next(iterator)
print(el[0].shape)
print(el[1].shape)

dataset_expanded = dataset.map(lambda x, y: (tf.expand_dims(x, axis=-1), y))
iterator = dataset_expanded.as_numpy_iterator()
el = next(iterator)
print(el[0].shape)
print(el[1].shape)

dataset_expanded_batched = dataset_expanded.batch(128)
iterator = dataset_expanded_batched.as_numpy_iterator()
el = next(iterator)
print(el[0].shape)
print(el[1].shape)

model_input = keras.Input(shape=(3, 3, 1), name="img")
x = layers.Conv2D(16, 3, activation="relu")(model_input)
x = layers.Flatten()(x)
model_output = layers.Dense(2)(x)
model = keras.Model(inputs=model_input, outputs=model_output, name="my_cnn")
model.summary()

model.compile(
    loss=tf.losses.MeanSquaredError(),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

model.fit(dataset_expanded_batched, epochs=30)





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





