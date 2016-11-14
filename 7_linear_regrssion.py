#http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex2/ex2.html

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
x_data = floatX(np.loadtxt('.\ex2Data\ex2x.dat'))
y_data = floatX(np.loadtxt('.\ex2Data\ex2y.dat'))

### params / init weights
w = theano.shared(floatX(np.random.randn(1))[0])
b = theano.shared(floatX(np.random.randn(1))[0])
    
### model
x = T.vector()
y = w*x + b

### cost/error/loss
y_hat = T.vector()
cost = T.mean((y-y_hat)**2)

### gradients
def gd(params, grads, lr):
    updates = []
    for p, g in it.izip(params, grads):      
        updates.append([p, p - lr*g])
    return updates
    
dw, db = T.grad(cost, [w, b])
train = theano.function(inputs=[x, y_hat],
                        outputs=[cost, w, b],
                        updates=gd([w, b], [dw, db], 0.01))

# training
costs = []
epochs = 1000

for t in range(epochs): 
        all_cost = 0       
        x_batches, y_batches = mk_batches(x_data, y_data, batch_size, True)        
        batch_num = len(x_batches)
        for i in range(batch_num):        
            results = train(x_batches[i], y_batches[i])
            all_cost += results[0]

        costs.append(all_cost/batch_num)        
        print 'batch avg cost=%f' % (all_cost/batch_num)        

print 'w=%f, b=%f' % (results[1], results[2])

### cost chart
plt.plot(costs)
plt.ylim([0, 0.2])
plt.xlabel("No. of parameters updates by batch")
plt.ylabel("Loss by batch of avg cost")
plt.show()

#the exact closed-form solution
#b = 0.7502
#w = 0.0639

### result chart
w = results[1]
b = results[2]
p1, = plt.plot(x_data, y_data, 'o', label='Training data')
p2, = plt.plot(x_data, w*x_data + b, 'r-', label='Linear regression')
plt.legend(handles=[p1, p2], loc=4)
plt.xlabel("Age in years")
plt.ylabel("Height in meters")
plt.show()