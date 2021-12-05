import tensorflow.keras as k
import numpy as np

# PROMPT:
# In this exercise you'll try to build a neural network that predicts the price of a house according to a simple formula.
#
# So, imagine if house pricing was as easy as a house costs 50k + 50k per bedroom, so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.
#
# How would you create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k etc.
#
# Hint: Your network might work better if you scale the house price down. You don't have to give the answer 400...it might be better to create something that predicts the number 4, and then your answer is in the 'hundreds of thousands' etc.

# build model layers
model = k.Sequential()
model.add(k.layers.Dense(units=1, input_shape=[1]))

# compile model (optimizer & loss functions)
model.compile(optimizer='sgd', loss='mean_squared_error')

# vector sequences
# x = num rooms, y = cost in $100k USD
xs = np.array([0, 1, 2, 3, 4, 5, 6], dtype=int)
ys = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

# fit model
model.fit(xs, ys, epochs=500)

print(model.predict([7.0]))