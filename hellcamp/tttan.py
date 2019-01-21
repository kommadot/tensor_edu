import tensorflow as tf
import csv
import numpy as np

sess = tf.Session()
'''
filename_queue = tf.train.string_input_producer(['data-01-test-score.csv'],shuffle=False,name='filenname_queue')
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
record_defaults = [[0.],[0.],[0.],[0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch,train_y_batch=tf.train.batch([xy[0:-1],xy[-1:]],batch_size=10)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
'''
xy = np.loadtxt('data-01-test-score.csv',delimiter=',',dtype=np.float32)
x_batch = xy[:,0:-1]
y_batch = xy[:,[-1]]

x = tf.placeholder(tf.float32,shape=[None,3])
y = tf.placeholder(tf.float32,shape=[None,1])
W = tf.Variable(tf.random_normal([3,1]))
b = tf.Variable(tf.random_normal([1]))
H = tf.matmul(x,W) + b

cost = tf.reduce_mean(tf.square(H-y))

optimizer = tf.train.GradientDescentOptimizer(1e-5)

train = optimizer.minimize(cost)

sess.run(tf.global_variables_initializer())

for i in range(40000):
    #x_batch, y_batch = sess.run([train_x_batch,train_y_batch])
    sess.run(train,feed_dict={x:x_batch,y:y_batch})
    if i%5000==0:
        print(sess.run([W,b]),i)

print(sess.run(H,feed_dict={x:[[82.,96.,90.]]}))
#coord.request_stop()
#coord.join(threads)