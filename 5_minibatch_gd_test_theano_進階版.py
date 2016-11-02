import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import itertools as it

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

### make batches
def mk_batches(x_data, y_data, batch_size, shuffle=False):
    x_batch = list()
    y_batch = list()
    
    x_data_size = x_data.shape[0]
    y_data_size = y_data.shape[0]
    assert x_data_size == y_data_size , 'the x, y dimension is error'
    
    if shuffle:
        indices = np.arange(x_data_size)
        np.random.shuffle(indices)
    
    #range(start, stop, step)
    for start_idx in range(0, x_data_size, batch_size):
        if shuffle:
            idx = indices[start_idx : start_idx + batch_size]            
        else:
            idx = slice(start_idx, start_idx + batch_size)
            
        x_batch.append(x_data[idx])
        y_batch.append(y_data[idx])
    
    return x_batch, y_batch
    
### data
batch_size = 10
x_data = floatX(np.array(np.linspace(-5.0, 5.0, 100)))
y_data = floatX(np.array(x_data))

### params / init weights
w = theano.shared(floatX(np.array([-1. for i in range(batch_size)])))
    
### model
x = T.vector()
y = w*x
f = theano.function([x], y)

### gradients
def gd(params, grads, lr):
    updates = []
    for p, g in it.izip(params, grads):      
        updates.append([p, p - lr*T.sum(g)])
    return updates

### cost/error/loss
y_hat = T.vector()
cost = T.sum((y-y_hat)**2)

dw = T.grad(cost, w)
train = theano.function(inputs=[x, y_hat],
                        outputs=[cost,w],
                        updates=gd([w], [dw], 0.001))

# training
cost_list = []
for t in range(5):
        cost_val = 0
        x_batches, y_batches = mk_batches(x_data, y_data, batch_size, True)
        batch_num = len(x_batches)
        
        for i in range(len(x_batches)):    
            results = train(x_batches[i], y_batches[i])
            print results[0], results[1][0]
            cost_val += results[0]

        cost_val = cost_val / batch_num
        cost_list.append(cost_val)
        
plt.plot(cost_list)
plt.xlabel("No. of parameters updates")
plt.ylabel("Loss/Cost/Error")
plt.show()
