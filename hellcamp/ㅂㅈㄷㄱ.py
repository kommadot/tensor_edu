import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-01-test-score.csv',dtype=np.float32,delimiter=',')

x_data = xy[:,0:-1]
y_data = xy[:,[-1]]


X = tf.placeholder(tf.float32,shape=[None,3])
Y = tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]))
b = tf.Variable(tf.random_normal([1]))

H = tf.matmul(X,W) + b

cost = tf.reduce_mean(tf.square(H-Y))

train = tf.train.GradientDescentOptimizer(1e-5).minimize(cost)

sess= tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(20001):
    sess.run(train,feed_dict={X:x_data,Y:y_data})

print(sess.run(H,feed_dict={X:[[82.,79.,92.]]}))