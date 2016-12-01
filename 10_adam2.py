#https://gist.github.com/Newmu/acb738767acb4788bac3

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

### model
x = T.vector()
y = w*x + b

### cost/error/loss
y_hat = T.vector()
cost = T.mean((y-y_hat)**2)

###############################################################################
### adam
adam_dw = theano.shared(floatX(0.))
adam_db = theano.shared(floatX(0.))
def Adam():
    updates = []    

    updates.append([w, w - adam_dw])
    
    updates.append([b, b - adam_db])
    
    return updates

his_m_dw = []
his_m_db = []
his_v_dw = []
his_v_db = []
def calc_adam(x, y, i):
        
    beta1 = 0.9 # momentum
    beta2 = 0.001 # rmsprop
    lr = 0.03
    e=1e-8    
    
    # Compute bias-corrected應該寫這樣，動態調整learning rate
    i_t = i + 1
    fix1 = 1. - (1. - beta1)**i_t
    fix2 = 1. - (1. - beta2)**i_t
    lr_t = lr * (np.sqrt(fix2) / fix1)    
    
    pre_m_dw = 0
    pre_m_db = 0
    pre_v_dw = 0
    pre_v_db = 0
    
    g_dw = f_grad_dw(x, y)          
    g_db = f_grad_db(x, y) 
    
    if i == 0:  
        pre_m_dw = 0
        pre_m_db = 0
        pre_v_dw = 0
        pre_v_db = 0
    else:
        pre_m_dw = his_m_dw[i-1]
        pre_m_db = his_m_db[i-1]
        pre_v_dw = his_v_dw[i-1]
        pre_v_db = his_v_db[i-1]
        
    # momentum
    new_m_dw = beta1 * pre_m_dw + (1 - beta1) * g_dw
    #new_m_dw = new_m / (1. - beta1) 寫錯           
    his_m_dw.append(new_m_dw)
        
    new_m_db = beta1 * pre_m_db + (1. - beta1) * g_db
    #new_m_db = new_m / (1. - beta1) 寫錯
    his_m_db.append(new_m_db)
    
    # rmsprop
    new_v_dw = beta2 * pre_v_dw + (1 - beta2) * np.square(g_dw)
    #new_v_dw = new_v / (1. - beta2) 寫錯       
    his_v_dw.append(new_v_dw)
    
    new_v_db = beta2 * pre_v_db + (1 - beta2) * np.square(g_db)
    #new_v_db = new_v / (1. - beta2) 寫錯      
    his_v_db.append(new_v_db)
    
    adam_dw.set_value(floatX( lr_t * (new_m_dw / np.sqrt(new_v_dw) + e )))   
    adam_db.set_value(floatX( lr_t * (new_m_db / np.sqrt(new_v_db) + e )))

###############################################################################
### adam (Alec Radford)
def Adam2(cost, params, lr=0.03, b1=0.9, b2=0.001, e=1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(floatX(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = ((1. - b1) * g) + (b1 * m)
        v_t = ((1. - b2) * T.sqr(g)) + (b2 * v)
        #m_t = (b1 * g) + ((1. - b1) * m)
        #v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates

###############################################################################
### theano function
dw, db = T.grad(cost, [w, b])

f_model = theano.function([x], y)

f_cost = theano.function([x, y_hat], cost)

f_grad_dw = theano.function([x, y_hat], dw)

f_grad_db = theano.function([x, y_hat], db)

f_train_adam = theano.function(inputs=[x, y_hat],
                               outputs=[cost, w, b],
                               updates=Adam())

f_train_adam2 = theano.function(inputs=[x, y_hat],
                               outputs=[cost, w, b],
                               updates=Adam2(cost, [w, b]))
###
epochs = 300

###############################################################################
# training by adam
w.set_value(floatX(-1.))
b.set_value(floatX(-1.))
his_cost_by_adam = pd.DataFrame(columns=['cost', 'w', 'b'])
tt = 0
for t in range(epochs):
        all_cost = 0       
        x_batches, y_batches = mk_batches(x_data, y_data, batch_size, True)        
        batch_num = len(x_batches)      
            
        for i in range(batch_num):
            calc_adam(x_batches[i], y_batches[i], tt)            
            tr_cost, tr_w, tr_b= f_train_adam(x_batches[i], y_batches[i])
            tt += 1
            all_cost += tr_cost                        

        his_cost_by_adam.loc[t] = [all_cost/batch_num, tr_w, tr_b]                   
print '(Adam) minimum result \n %s' % (his_cost_by_adam.loc[his_cost_by_adam['cost'].argmin()])

###############################################################################
# training by adam (Alec Radford)
w.set_value(floatX(-1.))
b.set_value(floatX(-1.))
his_cost_by_adam2 = pd.DataFrame(columns=['cost', 'w', 'b'])
tt = 0
for t in range(epochs):
        all_cost = 0       
        x_batches, y_batches = mk_batches(x_data, y_data, batch_size, True)        
        batch_num = len(x_batches)      
            
        for i in range(batch_num):
            calc_adam(x_batches[i], y_batches[i], tt)            
            tr_cost, tr_w, tr_b= f_train_adam2(x_batches[i], y_batches[i])
            tt += 1
            all_cost += tr_cost                        

        his_cost_by_adam2.loc[t] = [all_cost/batch_num, tr_w, tr_b]                   
print '(Adam (Alec Radford)) minimum result \n %s' % (his_cost_by_adam2.loc[his_cost_by_adam2['cost'].argmin()])

print '(closed-fom) w=0.0639, b= 0.7502'

###############################################################################
### cost chart
plt.plot(his_cost_by_adam.iloc[:, 0], label='adam')
plt.plot(his_cost_by_adam2.iloc[:, 0], label='adam (Alec Radford)')
plt.legend()
plt.title("learning rate=0.03, beta1=0.9, beta2=0.001")
plt.xlabel("No. of parameters updates by batch")
plt.ylabel("Loss by batch of avg cost")
plt.ylim([0, 0.4])
#plt.yscale('log')
plt.savefig('adam.png')
plt.show()

