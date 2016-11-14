#http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex4/ex4.html

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

### feature scaling
def feature_scaling(x_data):
    return (x_data - np.mean(x_data)) / np.std(x_data)
    
### 參數調整
BATCH_SIZE = 10
LEARNING_RATE = 0.001
EPOCHS = 1000

### data
x_data = floatX(np.loadtxt('.\ex4Data\ex4x.dat'))
y_data = floatX(np.loadtxt('.\ex4Data\ex4y.dat'))

### data chart
x_data_0 = x_data[y_data==0]
x_data_1 = x_data[y_data==1]
p1 = plt.scatter(x_data_0[:,0], x_data_0[:,1], c='r', marker='x', label='no pass')
p2 = plt.scatter(x_data_1[:,0], x_data_1[:,1], c='b', marker='o', label='pass')
plt.legend(handles=[p1, p2], loc=4)
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.show()

### params / init weights
w = theano.shared(floatX(np.random.randn(2))) #有兩個x就要搭配兩個w
b = theano.shared(floatX(np.random.randn(1))[0])

# Newton's Method 牛頓法求出來的解
#w = theano.shared(floatX([0.1589, 0.1483]))
#b = theano.shared(-16.38)
#cost 大概0.41X多
    
### model
x = T.matrix()
z = T.dot(w, x.T) + b
y = 1 / (1 + T.exp(-1 * z))
f = theano.function([x], y)

### cost/error/loss
y_hat = T.vector()
#cost = -T.mean(y_hat * T.log(y) + (1 - y_hat) * T.log(1 - y))
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
                        updates=gd([w, b], [dw, db], LEARNING_RATE))

# training
def training(epochs, x_data, y_data):
    costs = []
    for t in range(epochs): 
        all_cost = 0       
        x_batches, y_batches = mk_batches(x_data, y_data, BATCH_SIZE, True)        
        batch_num = len(x_batches)
        for i in range(batch_num):        
            results = train(x_batches[i], y_batches[i])            
            all_cost += results[0]
        
        costs.append(all_cost/batch_num)
    
    print 'avg cost=%f' % (costs[-1])        
    print 'w1=%f, w2=%f, b=%f' % (results[1][0], results[1][1], results[2])

    return costs, results

# cost chart
def plot_cost(costs):
    plt.plot(costs)
    plt.ylim([0, 3.0])
    plt.xlabel("No. of parameters updates by batch")
    plt.ylabel("Loss by batch of avg cost")
    plt.show()
     
costs, results = training(EPOCHS, x_data, y_data)
plot_cost(costs)

### feature scaling experiment
w.set_value(floatX(np.random.randn(2)))
b.set_value(floatX(np.random.randn(1))[0])

x_data = feature_scaling(x_data)
print 'mean=%f, std=%f' % (np.mean(x_data), np.std(x_data))
costs, results = training(EPOCHS, x_data, y_data)
plot_cost(costs)