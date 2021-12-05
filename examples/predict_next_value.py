import tensorflow.keras as k
import numpy as np

# build model layers
model = k.Sequential()
model.add(k.layers.Dense(units=1, input_shape=[1]))

# compile model (optimizer and loss functions)
model.compile(optimizer='sgd', loss='mean_squared_error')

# vector sequences
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# fit model:
model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))