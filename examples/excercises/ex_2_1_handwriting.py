import numpy as np
import tensorflow as tf
import tensorflow.keras as k
import pprint

pp = pprint.PrettyPrinter(indent=4)

# https://bit.ly/tfw-lab2cvq


class StopEarlyCallback(k.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.99:
            print('Reached or exceeded 99% accuracy. Stopping training.')
            self.model.stop_training = True


callbacks = StopEarlyCallback()

handwriting_data = k.datasets.mnist
(training_images, training_labels), (testing_images, testing_labels) = handwriting_data.load_data()

training_images = training_images/255.0  # normalize
testing_images = testing_images/255.0

i0 = training_images[0]
size_x = len(i0[0])
size_y = len(i0)
num_classes = 10  # digits 0-9


# build model layers
model = k.Sequential()
model.add(k.layers.Flatten(input_shape=(size_x, size_y)))
model.add(k.layers.Dense(units=1024, activation=tf.nn.relu))
model.add(k.layers.Dense(units=num_classes, activation=tf.nn.softmax))

# compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit
model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])

# evaluate and predict
model.evaluate(testing_images, testing_labels)
classifications = model.predict(testing_images)
pp.pprint(classifications[0])
pp.pprint(testing_labels[0])
