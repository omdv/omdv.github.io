---
title: Udacity Artificial Intelligence nanodegree
layout: post
page.categories: [udacity, notes, artificial intelligence]
---

# Term 1

# Term 2

## 2.1 Convolutional Neural Networks

### 2.1.2 Dog Breed Project

### Links
1. [Stanford course on CNN](http://cs231n.stanford.edu)


## Lesson 8: Tensorflow

### Hello World

[Tensorflow](www.tensorflow.org) Hello World script:
```
import tensorflow as tf

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)
```

### Data as Tensors
In TF data represented as tensors, so tf.constant is 0-dimensional string tensor. Other examples:

```
# A is a 0-dimensional int32 tensor
A = tf.constant(1234) 
# B is a 1-dimensional int32 tensor
B = tf.constant([123,456,789]) 
# C is a 2-dimensional int32 tensor
C = tf.constant([ [123,456,789], [222,333,444] ])
```
TF session is an environment to run a computational graph, in the example above we call it to evaluate the `hello_constant`.

### Variables, Placeholders, Constants
`tf.placeholder()` returns a tensor that gets its value from data passed to the tf.session.run() function, allowing you to set the input right before the session runs. To pass data use `feed_dict` in `session.run()`.

```
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})
```

TF math requires specific operators, like in the example below:
```
x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.divide(x,y),1)

# TODO: Print z from a session
with tf.Session() as sess:
    output = sess.run(z)
    print(output)
```


`tf.constant()` and `tf.placeholder()` cannot be modified, so we need a Tensor which can be modified. `tf.Variable` class allows it. `tf.Variable` needs to be initialized before running it, like so:
```
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```

### Softmax Implementation
Below is the example of softmax application:
```
import tensorflow as tf

def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)
    
    # TODO: Calculate the softmax of the logits
    softmax = tf.nn.softmax(logits)
    
    with tf.Session() as sess:
        # TODO: Feed in the logit data
        output = sess.run(softmax, feed_dict = {logits: logit_data})

    return output
```

### Cross Entropy Implementation
Below is the example of cross entropy calculation. Note the use of `tf.reduce_sum()`
```
# Solution is available in the other "solution.py" tab
import tensorflow as tf

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# TODO: Print cross entropy from session
cross_entropy = -tf.reduce_sum(tf.multiply(one_hot, tf.log(softmax)))

with tf.Session() as sess:
    output = sess.run(cross_entropy, feed_dict={softmax: softmax_data, one_hot: one_hot_data})
    print(output)
```

### Mini-batching
For mini-batching placeholders need to be defined with undefined batch sizes:
```
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])
```
TF session will then accept any tensor of size larger than zero during calculation. Below is the example of using batching API:

```
import tensorflow as tf
import numpy as np
from helper import batches

learning_rate = 0.001
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

# The features are already scaled and the data is shuffled
train_features = mnist.train.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Logits - xW + b
logits = tf.add(tf.matmul(features, weights), bias)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# TODO: Set batch size
batch_size = 128
assert batch_size is not None, 'You must set the batch size'

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    # TODO: Train optimizer on all batches
    for batch_features, batch_labels in batches(batch_size, train_features, train_labels):
        sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

    # Calculate accuracy for test dataset
    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: test_features, labels: test_labels})

print('Test Accuracy: {}'.format(test_accuracy))
```

### ReLu
Example of a 2-layered NN with ReLu activation:
```
import tensorflow as tf

output = None
hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input
features = tf.Variable([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [11.0, 12.0, 13.0, 14.0]])

# Model
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
output = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])

# Output
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
```

### Saving/Loading
The class `tf.train.Saver()` provides the necessary API:
```
# Class used to save and/or restore Tensor Variables
saver = tf.train.Saver()

# Save the model
saver.save(sess, save_file)

# Load the weights and bias
saver.restore(sess, save_file)
```

The last call will load weights and bias values into the model, however you still have to create variables. You don't need to call initialization however.

If you need to load values into a slightly changed or tuned model you need to assign names to make it more explicit:
```
# Two Tensor Variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]), name='weights_0')
bias = tf.Variable(tf.truncated_normal([3]), name='bias_0')
```

### Links
1. [Udacity Tensorflow Lab repo](https://github.com/udacity/deep-learning)
2. [Tensorflow tutorials and examples with latest API](https://github.com/aymericdamien/TensorFlow-Examples)


## Lesson 9: Autoencoders
Autoencoders are used to compress data, as well as image denoising. Compression and decompression methods are learned from the train data and not engineered by human. Autoencoders are usually worse at compression than traditional methods, like jpeg, mp4, mpeg and they also do not generalize well to previously unseen data. However they found use in image denoising and dimensionality reduction.

The way autoencoders work is by constructing a neural network for a pipeline shown below.
![autoencoder](https://github.com/udacity/deep-learning/blob/master/autoencoder/assets/autoencoder_1.png)

Typical autoencoder NN will have some layers representing encoder and decoder. One hidden layer in between should be a "bottleneck layer" and have smaller number of units. We are really interested in the weights at this bottleneck layer, as this is our compressed representation of the input data.

![bottleneck](https://github.com/udacity/deep-learning/blob/master/autoencoder/assets/simple_autoencoder.png)

Below is the example of Tensorflow graph for a simple AE with three layers, where the middle hidden layer provides a compressed representation.

```
# Size of the encoding layer (the hidden layer)
encoding_dim = 32 # feel free to change this value

# Input and target placeholders
inputs_ = tf.placeholder(tf.float32, [None, len(img)], name='inputs')
targets_ = tf.placeholder(tf.float32, [None, len(img)], name='targets')

# Output of hidden layer, single fully connected layer here with ReLU activation
encoded = tf.layers.dense(inputs_, encoding_dim, activation=tf.nn.relu, name='encoder')

# Output layer logits, fully connected layer with no activation
logits = tf.layers.dense(encoded, len(img), activation=None, name='logits')
# Sigmoid output from logits
decoded = tf.sigmoid(logits)

# Sigmoid cross-entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
# Mean of the loss
cost = tf.reduce_mean(loss)

# Adam optimizer
opt = tf.train.AdamOptimizer(0.001).minimize(cost)
```

Here is the example of autoencoder based on CNN.
![CNN AE](https://github.com/udacity/deep-learning/blob/master/autoencoder/assets/convolutional_autoencoder.png)

Note that for the upscaling it is possible to use transposed convolutional layers which are available in tensorflow. However according to [2] this will lead to checkerboard patterns, so we'll resize layers by using nearest neighbor function also available in tensorflow.


### Links
1. [Deep Learning Lab public repo](https://github.com/udacity/deep-learning)
2. [Upsampling deconvolutional layers](https://distill.pub/2016/deconv-checkerboard/)



