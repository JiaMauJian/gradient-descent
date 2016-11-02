import theano
import theano.tensor as T
import numpy as np

floatX = theano.config.floatX
print floatX

#theano variables
x = T.vector()

#theano shared variables (global variables)
#w = theano.shared(np.array(np.ones(2), dtype=floatX), name='w')
#b = theano.shared(np.array(np.ones(2), dtype=floatX), name='b')
w = theano.shared(np.array([-1. for i in range(100)], dtype=floatX), name='w')
b = theano.shared(np.array([-1. for i in range(100)], dtype=floatX), name='b')

    
# define function
y = w*x**2 + b
f = theano.function([x], y)

# define gradient desecent
y_hat = T.vector()
cost = T.sum((y-y_hat)**2)
dw, db = T.grad(cost, [w,b])
lr = 0.00001
gradient = theano.function(inputs=[x, y_hat],
                           outputs=[dw, db],
                           updates=[(w, w-lr*T.sum(dw)), (b, b-lr*T.sum(db))])

x_data = np.array(np.linspace(-5.0, 5.0, 100), dtype=floatX)
y_data = np.array(x_data**2+1, dtype=floatX)

T = 1
t = 10000

for i in range(0, T):
    for j in range(0, t):
        gradient(x_data, y_data)        
    print "w=%f, b=%f" % (w.get_value()[0], b.get_value()[0])
    
# define test
cost_f = theano.function([x, y_hat], cost)
print "error=%f" % (cost_f(x_data, y_data))   
