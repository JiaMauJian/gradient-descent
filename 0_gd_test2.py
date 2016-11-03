import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n = 100
x = pd.Series(np.linspace(-5.0, 5.0, n))
y = pd.Series()

# y = ax**2 + bx + c => 簡化 y = ax**2 + b

a = 1
b = 1
print "answer w=%d, b=%d" % (a, b)

for i in range(0, len(x)):
    y = y.set_value(i, a*x[i]**2 + b)
plt.plot(x, y, 'o-')
plt.title('data')

# loss/error/cost function
# L = sum(yhat - y)**2
# 簡單化 b=0, c變b, y = ax**2 + b
# dx = 2*(yhat - (ax**2 + b))(-2ax)
# dc = 2*(yhat - (ax**2 + b))(-1)

epochs = 10000
# epochs=1, w=-0.445657, b=-0.961993, dw.sum()=-55434.34206019657, db.sum()=-3800.6734006733986

# init weights
#w = pd.Series([-1.])
#b = pd.Series([-1.])
# 用隨機整數值才有好結果，用固定值如2,1,...反而跑不出好結果，可能我假設的function太複雜，算cost是4次方了
w = pd.Series([np.random.rand()])
b = pd.Series([np.random.rand()])

learning_rate = 0.00001 #learning rate太大會飛出去

# training    
for t in range(0, epochs):
    dw = pd.Series()   
    db = pd.Series()
    #for i in range(0, len(x)): 盡量不要寫迴圈
    #    dw.loc[i] = 2 * (y[i] - (w[t]*x[i]**2 + b[t])) * (-x[i]**2)
    #    db.loc[i] = 2 * (y[i] - (w[t]*x[i]**2 + b[t])) *(-1)        
    dw = 2 * (y - (w[t]*x**2 + b[t])) * (-x**2)
    db = 2 * (y - (w[t]*x**2 + b[t])) *(-1)
    
    w.loc[t+1] = w[t] - learning_rate * dw.sum()
    b.loc[t+1] = b[t] - learning_rate * db.sum()
                   
#testing
w1 = w[epochs]
b1 = b[epochs]
err = np.sum((y - (w1 * x**2 + b1))**2)
print "trained w=%f, b=%f" % (w1, b1)
print "testing error=%f" % (err)
