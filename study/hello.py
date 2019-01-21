import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
hello=tf.constant('Hello, TensorFlow!')
xData = [1963,1978,1981,1994,1997,2001,2002,2005,2008,2010,2012]
yData = [10,50,100,300,400,480,520,600,750,700,760]
W = tf.Variable(tf.random_uniform([1],-100,100))
b = tf.Variable(tf.random_uniform([1],-100,100))
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
H = W * X + b
cost = tf.reduce_mean(tf.square(H-Y))
a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(5000):
    sess.run(train,feed_dict={X: xData, Y:yData})
    if i%500 ==0:
        print(i,sess.run(cost,feed_dict={X: xData,Y:yData}),sess.run(W),sess.run(b))
print(sess.run(H,feed_dict={X:[2018]}))