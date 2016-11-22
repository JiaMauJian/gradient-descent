from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import numpy as np 

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y = np.array([[0],[1],[1],[0]], dtype=np.float32)

model = Sequential()
model.add(Dense(input_dim=2,  output_dim=2, init='normal'))
model.add(Activation('sigmoid'))
model.add(Dense(output_dim=1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, nb_epoch=100000, batch_size=1, verbose=0)

loss, accuracy = model.evaluate(X, y)
print '\nLoss: %f, Accuracy: %.2f%%' % (loss, accuracy*100)