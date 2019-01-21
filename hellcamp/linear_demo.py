import tensorflow as tf
import csv
import numpy as np


xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_batch = xy[:, 0:-1]
y_batch = xy[:, [-1]]

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([3,1]))
b = tf.Variable(tf.random_normal([1]))
H = tf.matmul(x, W) + b

cost = tf.reduce_mean(tf.square(H-y))

optimizer = tf.train.GradientDescentOptimizer(1e-5)

train = optimizer.minimize(cost)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(40000):
    sess.run(train,feed_dict={x:x_batch,y:y_batch})
    if i%5000==0:
        print(sess.run([W, b]), i)

print(sess.run(H, feed_dict={x:[[82.,96.,90.]]}))