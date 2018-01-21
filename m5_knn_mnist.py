import numpy as np
import tensorflow as tf

# Import MNIST
from tensorflow.examples.tutorials.mnist import input_data

# Store the MNIST data in /tmp/data
mnist = input_data.read_data_sets('mnist_data', one_hot=True)

training_digits, training_labels = mnist.train.next_batch(5000)
test_digits, test_labels = mnist.test.next_batch(200)

training_digits_pl = tf.placeholder('float', [None, 784])

test_digit_pl = tf.placeholder('float', [784])

# Nearest neight calculation using L1 Distance
l1_distance = tf.abs(tf.add(training_digits_pl, tf.negative(test_digit_pl)))

distance = tf.reduce_sum(l1_distance, axis=1)

# Prediction: Get the min distance index (Nearest Neighbor)
pred = tf.argmin(distance, 0)

accuracy = 0

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # loop over test data
    for i in range(len(test_digits)):
        nn_index = sess.run(pred, feed_dict={training_digits_pl: training_digits, test_digit_pl: test_digits[i, :]})

        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction: ", np.argmax(training_labels[nn_index]), "True label: ", np.argmax(test_labels[i]))

        if np.argmax(training_labels[nn_index]) == np.argmax(test_labels[i]):
            accuracy += 1./len(test_digits)

    print("Done")
    print("Accuracy", accuracy)