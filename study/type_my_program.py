import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

xData = [1978,1981,1994,1997,2001,2002,2005,2008,2010,2012]
yData = [50,100,300,400,480,520,600,750,700,760]

num_points = 1000
vectors_set =[]
for i in range(num_points):
    x1 = np.random.normal(0.0,0.55)
    y1 = x1*0.1 + 0.3 + np.random.normal(0.0,0.03)
    vectors_set.append([x1,y1])

xData = [v[0] for v in vectors_set]
yData = [v[1] for v in vectors_set]

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)
H = tf.multiply(W,X)+b

cost = tf.reduce_mean(tf.square(H-Y))
a = tf.Variable(0.5)

optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
plt.plot(xData, yData, 'ro')
plt.legend()
plt.show()
for i in range(30):
    sess.run(train,feed_dict={X:xData,Y:yData})
    if i%5 == 0:
        W_, b_ = sess.run([W, b])
        plt.plot(xData, yData, 'ro')
        plt.plot(xData, W_ * xData + b_, label='Original data')
        plt.xlim(-2, 2)
        plt.ylim(0.1, 0.6)
        plt.legend()
        plt.show()
        print(i,sess.run(cost,feed_dict={X:xData,Y:yData}))
#print(sess.run(H,feed_dict={X:2018}))
W_,b_=sess.run([W,b])
plt.plot(xData,yData,'ro')
plt.plot(xData,W_*xData+b_, label='Original data')
plt.legend()
plt.show()