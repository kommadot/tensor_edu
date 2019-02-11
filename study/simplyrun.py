import tensorflow as tf
import numpy as np

class Model:
    def __init__(self,sess,name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            keep_prob = tf.placeholder(tf.float32)

            self.X = tf.placeholder(tf.float32, [None, 784])
            X_img = tf.reshape(X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            W3 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01))
            L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
            L3 = tf.nn.relu(L3)

            W4 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
            L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
            L4 = tf.nn.relu(L4)
            L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
            L4 = tf.reshape(L4, [-1, 128 * 4 * 4])

            W5 = tf.get_variable("W5_", shape=[4 * 4 * 128, 10], initializer=tf.contrib.layers.xavier_initializer())

            b = tf.Variable(tf.random_normal([10]))
            H = tf.matmul(L4, W5) + b

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=H, labels=Y))
            self.optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

    def predict(self,x_test,keep_prop=1.0):
        return self.sess.run(self.logits,feed_dict={self.X:x_test,self.keep_prob:keep_prop})

    def get_accuracy(self,x_test,y_test,keep_prop=1.0):
        return self.sess.run(self.accuracy,feed_dict={self.X:x_test,self.Y:y_test,self.keep_prob:keep_prop})

    def train(self,x_data,y_data,keep_prop=0.7):
        return self.sess.run([self.cost,self.optimizer],feed_dict={self.X:x_data,self.Y:y_data,self.keep_prob:keep_prop})
