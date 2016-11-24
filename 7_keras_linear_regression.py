# Sample Multilayer Perceptron Neural Network in Keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import keras as k
import numpy as np
import matplotlib.pyplot as plt

class LossHistory(k.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
# load and prepare the dataset
x_data = np.loadtxt('.\ex2Data\ex2x.dat')
y_data = np.loadtxt('.\ex2Data\ex2y.dat')

# 1. define the network
model = Sequential()
model.add(Dense(input_dim=1, output_dim=1, init='normal'))
#model.add(Dense(input_dim=1, output_dim=1, weights=[np.array([[-1.]], dtype='float32'), np.array([-1.], dtype='float32')] ))
model.add(Activation('linear'))

# 2. compile the network
sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
history = LossHistory()
model.compile(loss='mse', optimizer=sgd, metrics=['mse'])

# 3. fit the network
model.fit(x_data, y_data, nb_epoch=1000, batch_size=10, callbacks=[history], verbose=2)

plt.plot(history.losses)

## 4. evaluate the network (loss = accuracy)
loss, accuracy = model.evaluate(x_data, y_data)
print '\nLoss: %f, Accuracy: %f' % (loss, accuracy)

print '\nlayer size: %d' % (len(model.layers))

w, b = model.layers[0].get_weights()
print '\nw: %f' % (w)
print '\nb: %f' % (b)