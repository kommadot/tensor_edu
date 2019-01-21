import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

X_data = [1.,2.,3.]
Y_data = [2.,3.,4.]
X = tf.placeholder(dtype=tf.float32,shape=[None])
Y = tf.placeholder(dtype=tf.float32,shape=[None])

W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

H = W*X + b
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost = tf.reduce_mean(tf.square(H-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

for i in range(2000):
    sess.run(train,feed_dict={X:X_data,Y:Y_data})
    if i%100 == 0:
        print(sess.run(W),sess.run(b))