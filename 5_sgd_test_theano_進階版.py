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
x = T.scalar()
y = w*x
f = theano.function([x], y)

### cost/error/loss
y_hat = T.scalar()
cost = (y-y_hat)**2

### gradients
def sgd(params, grads, lr): 
    updates = []
    for p, g in it.izip(params, grads):      
        updates.append([p, p - lr*g])    
    return updates
    
dw = T.grad(cost, w)
train = theano.function(inputs=[x, y_hat],
                        outputs=[cost,w],
                        updates=sgd([w], [dw], 0.001))

# training
cost_list = []
epochs = 4
for t in range(epochs):
    # 每epoch一次就要記得重新shuffle一次
    idx = np.arange(x_data.shape[0])
    np.random.shuffle(idx)

    for i in idx:        
        cost_result = train(x_data[i], y_data[i])
        print cost_result
        cost_list.append(cost_result[0])

plt.plot(cost_list)
plt.xlabel("No. of parameters updates by sgd")
plt.ylabel("Loss/Cost/Error")
plt.show()
