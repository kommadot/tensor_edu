import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv',delimiter=',',dtype=np.float32)

x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

X = tf.placeholder(dtype=tf.float32,shape=[None,16])
Y = tf.placeholder(dtype=tf.int32,shape=[None,1])
nb_classes = 7
Y_one_hot = tf.one_hot(Y,nb_classes)
Y_one_hot = tf.reshape(Y_one_hot,[-1,nb_classes])
W = tf.Variable(tf.random_normal([16,nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

logits = tf.matmul(X,W)+b
H = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
predicted = tf.arg_max(H,1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,tf.arg_max(Y_one_hot,1)),tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(20001):
        sess.run(optimizer,feed_dict={X:x_data,Y:y_data})
        if i % 5000 == 0 :
            print(sess.run(accuracy,feed_dict={X:x_data,Y:y_data}))
