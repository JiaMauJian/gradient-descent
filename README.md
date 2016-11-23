"# 紀錄"
* 實作gradient-desenct
* 盡量不要寫迴圈
* 2016/10/15  gd_test2.py 改寫不要跑迴圈
* 2016/10/25 gd_test_theano.py 用theano寫gd
* 2016/10/26 gd_test_theano_learning_rate.py 比較不同learning rate的影響
* 2016/11/1 mk_batches.py 練習
* 2016/11/2 theano進階寫法練習(all data/mini batch/sgd)
* 2016/11/17 feature scaling (有比較滑喔)
* 2016/11/17 adagrad 好難寫
* 2016/11/22 xor

"# 想實作看看的東西"
* rmsprop
* adam
* MNIST / t-SNE

"# gradient desecnt / stochastic gradient descent"
* 你不是已經知道Cost最小值在哪邊了嗎?  因為你每次只能看到視野範圍內的東西 (os: 那是因為窮舉法 把所有的w值都畫出來，如果有多個w參數 你還能窮舉嗎？) [link](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/Lecture/DNN%20(v4).pdf#page=47)

* 怎麼做gd [link](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/Regression%20(v6).pdf) (對b的偏微分少乘-1，因為ppt動畫關係)
* 怎麼做sgd [link](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/Gradient%20Descent%20(v2).pdf)

"# mini batch"
* 怎麼做mini batch [link](http://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python)
* mini batch比較快? 因為batch比較快，那batch size越大越好??? [link](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/Keras.pdf#page=18)
* mini batch比較快? 不同batch size的比較 [link](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/Lecture/DNN%20(v4).pdf#page=66)

"# 不同learning rate的影響"
* [link](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/Gradient%20Descent%20(v2).pdf#page=5)
* ![alt tag](http://cs231n.github.io/assets/nn3/learningrates.jpeg)


