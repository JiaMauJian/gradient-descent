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
w = theano.shared(-1.)
    
### model
x = T.vector()
y = w*x
f = theano.function([x], y)

### cost/error/loss
y_hat = T.vector()
cost = T.sum((y-y_hat)**2)

### gradients
def gd(params, grads, lr):
    updates = []
    for p, g in it.izip(params, grads):      
        updates.append([p, p - lr*g])
    return updates
    
dw = T.grad(cost, w)
train = theano.function(inputs=[x, y_hat],
                        outputs=[cost,w],
                        updates=gd([w], [dw], 0.001))

cost_list = []
for i in range(100):    
    cost_result = train(x_data, y_data)
    cost = cost_result[0]
    print "cost=%f, w=%f" % (cost, cost_result[1])
    cost_list.append(cost)
    
    if float(cost) > 0 and float(cost) < 0.0000001:
        break
    
plt.plot(cost_list)
plt.xlabel("No. of parameters updates")
plt.ylabel("Loss/Cost/Error")
plt.show()
