# Sample Multilayer Perceptron Neural Network in Keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import numpy as np


# load and prepare the dataset
x_data = np.loadtxt('.\ex4Data\ex4x.dat')
y_data = np.loadtxt('.\ex4Data\ex4y.dat')

# 1. define the network
model = Sequential()
model.add(Dense(input_dim=2, output_dim=1, init='normal'))
model.add(Activation('sigmoid'))

# 2. compile the network
sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['binary_accuracy'])

# 3. fit the network
model.fit(x_data, y_data, nb_epoch=1000, batch_size=10)

## 4. evaluate the network
loss, accuracy = model.evaluate(x_data, y_data)
print '\nLoss: %f, Accuracy: %.2f%%' % (loss, accuracy*100)
