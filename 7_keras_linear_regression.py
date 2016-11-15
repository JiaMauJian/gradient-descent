# Sample Multilayer Perceptron Neural Network in Keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import numpy as np


# load and prepare the dataset
x_data = np.loadtxt('.\ex2Data\ex2x.dat')
y_data = np.loadtxt('.\ex2Data\ex2y.dat')

# 1. define the network
model = Sequential()
model.add(Dense(input_dim=1, output_dim=1, init='normal'))
model.add(Activation('linear'))

# 2. compile the network
sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='mse', optimizer=sgd, metrics=['mse'])

# 3. fit the network
model.fit(x_data, y_data, nb_epoch=1000, batch_size=10, verbose=1, show_accuracy=True)

## 4. evaluate the network (loss = accuracy)
loss, accuracy = model.evaluate(x_data, y_data)
print '\nLoss: %f, Accuracy: %f' % (loss, accuracy)

print '\nlayer size: %d' % (len(model.layers))

w, b = model.layers[0].get_weights()
print '\nw: %f' % (w)
print '\nb: %f' % (b)