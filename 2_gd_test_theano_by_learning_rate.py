import theano
import theano.tensor as T
import numpy as np
import pandas as pd

floatX = theano.config.floatX
print floatX

lr = np.array([0.00008, 0.00006, 0.00003, 0.00001, 0.000001]) #0.0008以上直接飛出去
errs_by_learningRate = pd.DataFrame()

x_data = np.array(np.linspace(-5.0, 5.0, 100), dtype=floatX)
y_data = np.array(x_data**2+1, dtype=floatX)
    
for l in range(0, len(lr)):

    #theano variables
    x = T.vector()
    
    #theano shared variables (global variables)
    #w = theano.shared(np.array(np.random.randn(1)[0], dtype=floatX), name='w')
    #b = theano.shared(np.array(np.random.randn(1)[0], dtype=floatX), name='b')
    w = theano.shared(-1., name='w')
    b = theano.shared(-1., name='b')
    
    # define function
    y = w*x**2 + b
    f = theano.function([x], y)
    
    # define gradient desecent
    y_hat = T.vector()
    cost = T.sum((y-y_hat)**2)
    dw, db = T.grad(cost, [w,b])
    
    gradient = theano.function(inputs=[x, y_hat],
                               outputs=[dw, db],
                               updates=[(w, w-lr[l]*dw), (b, b-lr[l]*db)])
        
    _iter = np.array([100*i for i in range(1, 21)])
    errs = np.array([])    
    
    for i in range(0, len(_iter)):            
        for j in range(0, _iter[i]):
            gradient(x_data, y_data)
            
        #print "w=%f, b=%f" % (w.get_value(), b.get_value())    
        
        # define test
        cost_f = theano.function([x, y_hat], cost)
        
        err = cost_f(x_data, y_data)
        #print "t=%d, error=%f" % (_iter[i], err)   
        errs = np.append(errs, err)
        
    errs_by_learningRate[str(lr[l])] = errs

print '0.00008飛出去了'
ax = errs_by_learningRate.plot()
ax.set(xlabel="No. of parameters updates", ylabel="Loss")
ax = errs_by_learningRate.plot(logy='True')
ax.set(xlabel="No. of parameters updates", ylabel="Log(Loss)")

print '拿掉0.00008'
ax = errs_by_learningRate.ix[:, 1:].plot()
ax.set(xlabel="No. of parameters updates", ylabel="Loss")
ax = errs_by_learningRate.ix[:, 1:].plot(logy='True')
ax.set(xlabel="No. of parameters updates", ylabel="Log(Loss)")
#ref
#http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/Gradient%20Descent%20(v2).pdf
