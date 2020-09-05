import tensorflow as tf
import numpy as np



########################## tensors ##########################

# create 1, 2, 3 dim tensors
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)

rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)

rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor)

rank_3_tensor = tf.constant([
    [[0, 1, 2, 3, 4],
     [5, 6, 7, 8, 9]],
    [[10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19]],
    [[20, 21, 22, 23, 24],
     [25, 26, 27, 28, 29]], ])
print(rank_3_tensor)


# convert to numpy
v = np.array(rank_2_tensor)
v = rank_2_tensor.numpy()


# do math on tensors
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) # Could have also said `tf.ones([2,2])`

tf.add(a, b), "\n"
a + b, "\n"
tf.multiply(a, b), "\n" # element-wise multiplication
a * b, "\n"
tf.matmul(a, b), "\n" # matrix multiplication
a @ b, "\n"


# tensors properties
a.ndim
a.shape
print("numel a: " + str(tf.size(a).numpy()))
a.dtype
# important about dimensions.
# Typical tensors are 4d: batch size, width, height, feature size
a.device # on which device it resides (cpu or gpu)


# tensors indexing (as in numpy!)
a[:,1]
rank_1_tensor[:,None] # reshape to column vector
tf.reshape(rank_2_tensor, [-1]) # flatten
tf.reshape(rank_3_tensor, [3*2,5]) # just flatten certain dimensions
tf.reshape(rank_3_tensor, [3,-1])
tf.transpose(rank_2_tensor)


# ragged tensors (different number of elements in different dimensions)
ragged_tensor = tf.ragged.constant([
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9]])
ragged_tensor.shape


# tensors of strings (functions to manipulate strings are in tf.strings)
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])
print(tensor_of_strings)


# sparse tensors
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print(sparse_tensor, "\n")


# variables are analogous to (a wrapper around) tensors. They are the preferred
# way to represent a "shared, persistent status" (?).
a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.Variable([[1.0, 2.0, 3.0]])
k = a * b


# placing tensors and variables on CPU or GPU. They are typically placed
# automatically. However one can force the placement.
with tf.device('CPU:0'):
    a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
with tf.device('GPU:0'):
    a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# one can also decide to execute operations on a different device wrt that
# where the tensors are memorized (necessary sometimes, but requires extra
# copying of the variable!)
with tf.device('CPU:0'):
    a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.Variable([[1.0, 2.0, 3.0]])
with tf.device('GPU:0'):
  k = a * b


# pre-made estimators: They are an high-level ML models (both deep and standard
# statistical learning models). They represent an alternative to keras (that
# only provides DL models), to be used when non-DL models are needed or when
# more advanced functionalities are needed (parameter server based tuning and
# TFX integration). https://www.tensorflow.org/guide/estimator.
# NOTE: there are both deep and standard ML models!
# Workflow: (1) write database importing functions, (2) define feature columns,
# (3) istanciate the estimator, (4) train the estimator and do inference.


# dataset abstraction: https://www.tensorflow.org/guide/data#basic_mechanics
# Implemented by tf.data.Dataset provides an abstraction of a dataset: allows
# to treat the object as a dataset even if it automatically takes the data from
# memory (i.e. existing tensors) or from the disk, allows to apply
# transformations to all the elements of the dataset, and to combine multiple
# Dataset objects. Again, keras provides a higher-level abstraction for the
# Dataset class.




placeholder = 1