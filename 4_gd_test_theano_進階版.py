import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import itertools as it

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

### data
x_data = floatX(np.array(np.linspace(-5.0, 5.0, 100)))
y_data = floatX(np.array(x_data))

### params / init weights
#w = theano.shared(floatX(np.random.randn(*shape[0])))
#b = theano.shared(floatX(np.random.randn(*shape[0])))
w = theano.shared(floatX(np.array([-1. for i in range(x_data.shape[0])])))
b = theano.shared(floatX(np.array([-1. for i in range(x_data.shape[0])])))
#w = theano.shared(floatX(np.random.randn(1)[0]))
#b = theano.shared(floatX(np.random.randn(1)[0]))
#w = theano.shared(floatX(np.array(-1.)))
#b = theano.shared(floatX(np.array(-1.)))
    
### model
x = T.vector()
#x = T.scalar()
y = w*x
f = theano.function([x], y)

### gradients
def gd(params, grads, lr):
    updates = []
    for p, g in it.izip(params, grads):      
        updates.append([p, p - lr*T.sum(g)])
    return updates

def sgd(params, grads, lr):    
    updates = []
    for p, g in it.izip(params, grads):      
        updates.append([p, p - lr*g])    
    return updates

### cost/error/loss
y_hat = T.vector()
#y_hat = T.scalar()
cost = T.sum((y-y_hat)**2)
#cost = (y-y_hat)**2

dw = T.grad(cost, w)
train = theano.function(inputs=[x, y_hat],
                        outputs=[cost,w],
                        updates=gd([w], [dw], 0.001))

cost_list = []
for i in range(3):
    cost_result = train(x_data, y_data)
    print cost_result[0], cost_result[1][0]
    cost_list.append(cost_result[0])
    
plt.plot(cost_list)
plt.xlabel("No. of parameters updates")
plt.ylabel("Loss/Cost/Error")
plt.show()
