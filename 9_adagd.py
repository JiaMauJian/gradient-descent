#http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex2/ex2.html

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import time

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
w = theano.shared(floatX(-1.))
b = theano.shared(floatX(-1.))
#w = theano.shared(floatX(np.random.randn(1))[0])
#b = theano.shared(floatX(np.random.randn(1))[0])

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

def Adagd(params, grads):
    updates = []
    lr = floatX(1.0)
    
    updates.append([params[0], params[0] - (lr/second_dw)*grads[0]])
    
    updates.append([params[1], params[1] - (lr/second_db)*grads[1]])
    
    return updates
    
dw, db = T.grad(cost, [w, b])

### calculate every step gradient's root mean square
his_grad_dw = []
his_grad_db = []
second_dw = theano.shared(floatX(0.))
second_db = theano.shared(floatX(0.))
def calc_second_derivative(x, y):
    g_dw = f_grad_dw(x, y)          
    g_db = f_grad_db(x, y) 
    his_grad_dw.append(g_dw)
    his_grad_db.append(g_db)
    second_dw.set_value(floatX(np.sqrt(np.sum(np.square(his_grad_dw)))))
    second_db.set_value(floatX(np.sqrt(np.sum(np.square(his_grad_db)))))
    
### theano function
f_model = theano.function([x], y)

f_cost = theano.function([x, y_hat], cost)

f_grad_dw = theano.function([x, y_hat], dw)

f_grad_db = theano.function([x, y_hat], db)

f_train = theano.function(inputs=[x, y_hat],
                          outputs=[cost, w, b],
                          updates=gd([w, b], [dw, db], 0.01))

f_train_adagd = theano.function(inputs=[x, y_hat],
                                outputs=[cost, w, b],
                                updates=Adagd([w, b], [dw, db]))

epochs = 1000

# training
his_cost_by_gd = []
tStart = time.time()
for t in range(epochs):
        all_cost = 0       
        x_batches, y_batches = mk_batches(x_data, y_data, batch_size, True)        
        batch_num = len(x_batches)      
            
        for i in range(batch_num):            
            tr_cost, tr_w, tr_b= f_train(x_batches[i], y_batches[i])            
            all_cost += tr_cost            
            
        his_cost_by_gd.append(all_cost/batch_num)        
        #print 'batch avg cost=%f' % (all_cost/batch_num)        
tEnd = time.time()
print '(sgd) w=%f, b=%f' % (tr_w, tr_b)
print 'Minimum Loss = %f' % (np.min(his_cost_by_gd))
print 'It costs %f sec' % (tEnd-tStart)

w.set_value(floatX(-1.))
b.set_value(floatX(-1.))
#w.set_value(floatX(np.random.randn(1))[0])
#b.set_value(floatX(np.random.randn(1))[0])

his_cost_by_adagd = []
tStart = time.time()
for t in range(epochs):
        all_cost = 0       
        x_batches, y_batches = mk_batches(x_data, y_data, batch_size, True)        
        batch_num = len(x_batches)      
            
        for i in range(batch_num):
            calc_second_derivative(x_batches[i], y_batches[i])            
            tr_cost, tr_w, tr_b= f_train_adagd(x_batches[i], y_batches[i])            
            all_cost += tr_cost            
            
        his_cost_by_adagd.append(all_cost/batch_num)        
        #print 'batch avg cost=%f' % (all_cost/batch_num)        
tEnd = time.time()
print '(Adagd) w=%f, b=%f' % (tr_w, tr_b)
print 'Minimum Loss = %f' % (np.min(his_cost_by_adagd))
print 'It costs %f sec' % (tEnd-tStart)

print '(closed-fom) w=0.0639, b= 0.7502'

### cost chart
plt.plot(his_cost_by_gd, label='sgd')
plt.plot(his_cost_by_adagd, label='adagrad')
plt.legend()
plt.ylim([0, 0.4])
plt.xlabel("No. of parameters updates by batch")
plt.ylabel("Loss by batch of avg cost")
plt.show()