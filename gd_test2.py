import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = pd.Series(np.linspace(-5.0, 5.0, 100))
y = pd.Series()

# y = ax**2 + bx + c => 簡化 y = ax**2 + b

a = 1
b = 1

for i in range(0, len(x)):
    y = y.set_value(i, a*x[i]**2 + b)

plt.figure(1)
plt.plot(x, y, 'o-')

# loss/error/cost function
# L = sum(yhat - y)**2
# 簡單化 b=0, c變b, y = ax**2 + b
# dx = 2*(yhat - (ax**2 + b))(-2ax)
# dc = 2*(yhat - (ax**2 + b))(-1)
min_err = pd.DataFrame(columns=('min_err', 'w', 'b'))
E_min_err = pd.DataFrame(columns=('min_err', 'w', 'b'))

T = 30
t = 100

for T in range(0, T, 1):        
    w = pd.Series([np.random.rand()]) # w=a 用隨機整數值 反而跑不出好結果 如2,1,...
    b = pd.Series([np.random.rand()])
    learning_rate = 0.00001 #learning rate太大會飛出去
    
    for t in range(0, t, 1):
        dw = pd.Series()   
        db = pd.Series()
        #for i in range(0, len(x)): 盡量不要寫迴圈
        #    dw.loc[i] = 2 * (y[i] - (w[t]*x[i]**2 + b[t])) * (-x[i]**2)
        #    db.loc[i] = 2 * (y[i] - (w[t]*x[i]**2 + b[t])) *(-1)        
        dw = 2 * (y - (w[t]*x**2 + b[t])) * (-x**2)
        db = 2 * (y - (w[t]*x**2 + b[t])) *(-1)
        
        w.loc[t+1] = w[t] - learning_rate * dw.sum()
        b.loc[t+1] = b[t] - learning_rate * db.sum()
     
    err = pd.Series()
    errs = pd.DataFrame(columns=('err', 'w', 'b'))
                       
    for t in range(0, t, 1):
        #for i in range(0, len(x)): 盡量不要寫迴圈
        #    err.loc[i] = (y[i] - (w[t] * x[i]**2 + b[t]))**2
        err = (y - (w[t] * x**2 + b[t]))**2
        errs.loc[t] = np.array([err.sum(), w[t], b[t]])
        
    plt.figure(2)
    plt.plot(errs, 'bo-')
    print(errs.min())
    
    idx = errs['err'].idxmin()
    E_min_err.loc[T] = np.array(errs.loc[idx])
    
print("answer w, b")
print(a)
print(b)