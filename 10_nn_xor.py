#http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/Lecture/Theano%20DNN.pdf

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import itertools as it

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

# x1, x2    
x = T.vector()

# init weights
w1 = theano.shared(floatX(np.random.randn(2)))
b1 = theano.shared(floatX(np.random.randn(1)))
w2 = theano.shared(floatX(np.random.randn(2)))
b2 = theano.shared(floatX(np.random.randn(1)))
w = theano.shared(floatX(np.random.randn(2)))
b = theano.shared(floatX(np.random.randn(1)))

# hidden layer
z1 = T.dot(w1, x) + b1
a1 = 1 / (1 + T.exp(-1 * z1))
z2 = T.dot(w2, x) + b2
a2 = 1 / (1 + T.exp(-1 * z2))

# output layer
z = T.dot(w, [a1, a2]) + b
y = 1 / (1 + T.exp(-1 * z))

# cost (cross entropy)
y_hat = T.scalar()
cost = - (y_hat * T.log(y) + (1 - y_hat) * T.log(1 - y)).sum()

# grad
def gd(params, grads, lr):
    updates = []
    for p, g in it.izip(params, grads):      
        updates.append([p, p - lr*g])
    return updates
    
dw, db, dw1, db1, dw2, db2 = T.grad(cost, [w, b, w1, b1, w2, b2])

train = theano.function(inputs=[x, y_hat],
                        outputs=[y, cost, w, b, w1, b1, w2, b2],
                        updates=gd([w, b, w1, b1, w2, b2], [dw, db, dw1, db1, dw2, db2], 0.01))

for i in range(100000):
    y1, c1, _w, _b, _w1, _b1, _w2, _b2 = train([0, 0], 0)
    y2, c2, _w, _b, _w1, _b1, _w2, _b2 = train([0, 1], 1)
    y3, c3, _w, _b, _w1, _b1, _w2, _b2 = train([1, 0], 1)
    y4, c4, _w, _b, _w1, _b1, _w2, _b2 = train([1, 1], 0)
    
print 'cost=%f' % (c1+c2+c3+c4)
print y1,y2,y3,y4

a1_00 =  1 / (1 + np.exp(-1 * (np.dot(_w1, [0,0]) + _b1)))
a1_01 =  1 / (1 + np.exp(-1 * (np.dot(_w1, [0,1]) + _b1)))
a1_10 =  1 / (1 + np.exp(-1 * (np.dot(_w1, [1,0]) + _b1)))
a1_11 =  1 / (1 + np.exp(-1 * (np.dot(_w1, [1,1]) + _b1)))
print a1_00, a1_01, a1_10, a1_11

a2_00 =  1 / (1 + np.exp(-1 * (np.dot(_w2, [0,0]) + _b2)))
a2_01 =  1 / (1 + np.exp(-1 * (np.dot(_w2, [0,1]) + _b2)))
a2_10 =  1 / (1 + np.exp(-1 * (np.dot(_w2, [1,0]) + _b2)))
a2_11 =  1 / (1 + np.exp(-1 * (np.dot(_w2, [1,1]) + _b2)))
print a2_00, a2_01, a2_10, a2_11

plt.plot([a1_00, a2_00], [a1_11, a2_11], 'ro', [a1_01, a2_01], [a1_10, a2_10], 'bo')
plt.xlabel('a1')
plt.ylabel('a2')
plt.axis([-0.2, 1.2, -0.2, 1.22])
plt.show()