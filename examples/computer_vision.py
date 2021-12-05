import tensorflow as tf
import tensorflow.keras as k
import numpy as np

EPOCHS = 5

# data
fashion_mnist_data = k.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist_data.load_data()

# build model layers
model = k.Sequential()

# top layer (input format) -WITH NN, ALL DATA MUST BE SAME SIZE
model.add(k.layers.Flatten(input_shape=(28, 28)))  # 2D to 1D, 28 because we know the images are 28px square

# what is nn.relu? it's an activation function
# relu sets all negative outputs to 0 (throwaway rather than skew positives)
model.add(k.layers.Dense(128, activation=tf.nn.relu))  # why 128 neurons?

# bottom layer (output format)
# 10 neurons because there are 10 types of clothing in the fashion_mnist dataset
# the job of each neuron is to calculate the probability that incoming data belongs to its class
# softmax sets largest probability to 1
model.add(k.layers.Dense(10, activation=tf.nn.softmax))

# compile model using optimizer and loss fns
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# vector sequences
xs = train_images
ys = train_labels

# fit model
model.fit(xs, ys, epochs=EPOCHS)

# predict/evaluate from model
# model.predict()  #
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print('loss:')
print(test_loss)
print('\naccuracy:')
print(test_accuracy)
