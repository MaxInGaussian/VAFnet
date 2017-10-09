# Copyright 2017 Max W. Y. Lam
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import edward as ed
import tensorflow as tf
from edward.models import Normal
from observations import boston_housing

data_dir = "./boston/data"
p_train = 0.8
n_batch = 10
n_epoch = 500
n_print = 50

def VAFnet(X):
    h = tf.tanh(tf.matmul(X, W_0) + b_0)
    h = tf.tanh(tf.matmul(h, W_1) + b_1)
    h = tf.matmul(h, W_2) + b_2
    return h

def generator(array, batch_size):
    """Generate batches, one with respect to each array's first axis."""
    starts = [0] * len(arrays)  # pointers to where we are in iteration
    while True:
        batches = []
        for i, array in enumerate(arrays):
            start = starts[i]
            stop = start + batch_size
            diff = stop - array.shape[0]
            if diff <= 0:
                batch = array[start:stop]
                starts[i] += batch_size
            else:
                batch = np.concatenate((array[start:], array[:diff]))
                starts[i] = diff
            batches.append(batch)
    yield batches


data, meta = boston_housing(data_dir)
n_data = data.shape[0]
train_idx = np.random.choice(range(n_data), int(n_data*p_train), replace=False)
test_idx = np.setdiff1d(np.arange(n_data), train_idx)
train, test = data[train_idx], data[test_idx]
D, N = train.shape[1]-1, train.shape[0]//n_batch
batch_generator = generator(train, N)
X_train = tf.placeholder(tf.float32, [N, D])
y_train = tf.placeholder(tf.float32, [N, 1])


# MODEL
with tf.name_scope("model"):
    W_0 = Normal(loc=tf.zeros([D, 10]), scale=tf.ones([D, 10]), name="W_0")
    W_1 = Normal(loc=tf.zeros([10, 10]), scale=tf.ones([10, 10]), name="W_1")
    W_2 = Normal(loc=tf.zeros([10, 1]), scale=tf.ones([10, 1]), name="W_2")
    b_0 = Normal(loc=tf.zeros(10), scale=tf.ones(10), name="b_0")
    b_1 = Normal(loc=tf.zeros(10), scale=tf.ones(10), name="b_1")
    b_2 = Normal(loc=tf.zeros(1), scale=tf.ones(1), name="b_2")
    
    X = tf.placeholder(tf.float32, [N, D], name="X")
    y = Normal(loc=VAFnet(X), scale=0.1 * tf.ones(N), name="y")

# INFERENCE
with tf.name_scope("posterior"):
    with tf.name_scope("qW_0"):
        qW_0 = Normal(loc=tf.Variable(tf.random_normal([D, 10]), name="loc"),
            scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([D, 10]), name="scale")))
    with tf.name_scope("qW_1"):
        qW_1 = Normal(loc=tf.Variable(tf.random_normal([10, 10]), name="loc"),
            scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([10, 10]), name="scale")))
    with tf.name_scope("qW_2"):
        qW_2 = Normal(loc=tf.Variable(tf.random_normal([10, 1]), name="loc"),
            scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([10, 1]), name="scale")))
    with tf.name_scope("qb_0"):
        qb_0 = Normal(loc=tf.Variable(tf.random_normal([10]), name="loc"),
            scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([10]), name="scale")))
    with tf.name_scope("qb_1"):
        qb_1 = Normal(loc=tf.Variable(tf.random_normal([10]), name="loc"),
            scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([10]), name="scale")))
    with tf.name_scope("qb_2"):
        qb_2 = Normal(loc=tf.Variable(tf.random_normal([1]), name="loc"),
            scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([1]), name="scale")))

optimizer = tf.train.AdamOptimizer()
inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                     W_1: qW_1, b_1: qb_1,
                     W_2: qW_2, b_2: qb_2},
                     data={X: X_train, y: y_train})
inference.initialize(
    optimizer=optimizer, n_iter=n_batch*n_epoch, n_print=n_print)
tf.global_variables_initializer().run()

for t in range(inference.n_iter):
    batch = next(batch_generator)
    X_batch, y_batch = batch[:, :-1], batch[:, -1][:, None]
    info_dict = inference.update(feed_dict={X_train: X_batch, y_train: y_batch})
    inference.print_progress(info_dict)

y_post = ed.copy(y, {w: qw, b: qb})

X_test, y_test = test[:, :-1], test[:, -1][:, None]

print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test}))

print("Mean absolute error on test data:")
print(ed.evaluate('mean_absolute_error', data={X: X_test, y_post: y_test}))




