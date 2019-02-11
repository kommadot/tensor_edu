import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("samples/MNIST_data/" ,one_hot=True)

nb_classes = 10
keep_prob =tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32 ,[None ,784])
X_img = tf.reshape(X ,[-1 ,28 ,28 ,1])
Y = tf.placeholder(tf.float32 ,[None ,10])

W1 = tf.Variable(tf.random_normal([3 ,3 ,1 ,32] ,stddev=0.01))
L1 = tf.nn.conv2d(X_img ,W1 ,strides=[1 ,1 ,1 ,1] ,padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1 ,ksize=[1 ,2 ,2 ,1] ,strides=[1 ,2 ,2 ,1] ,padding='SAME')

W2 = tf.Variable(tf.random_normal([3 ,3 ,32 ,64] ,stddev=0.01))
L2 = tf.nn.conv2d(L1 ,W2 ,strides=[1 ,1 ,1 ,1] ,padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2 ,ksize=[1 ,2 ,2 ,1] ,strides=[1 ,2 ,2 ,1] ,padding='SAME')
print(L2)
W3 = tf.Variable(tf.random_normal([3,3,64,64],stddev=0.01))
L3 = tf.nn.conv2d(L2,W3,strides=[1,1,1,1],padding='SAME')
L3 = tf.nn.relu(L3)

W4 = tf.Variable(tf.random_normal([3,3,64,128],stddev=0.01))
L4 = tf.nn.conv2d(L3,W4,strides=[1,1,1,1],padding='SAME')
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
L4 = tf.nn.dropout(L4,keep_prob=keep_prob)
L4 = tf.reshape(L4,[-1,128*4*4])


W5 = tf.get_variable("W5_" ,shape=[4*4*128,10],initializer=tf.contrib.layers.xavier_initializer())

b = tf.Variable(tf.random_normal([10]))
H = tf.matmul(L4,W5) +b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=H ,labels=Y))
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

training_epochs =20
batch_size = 100
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples /batch_size)
    for i in range(total_batch):
        batch_xs ,batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X :batch_xs ,Y :batch_ys,keep_prob:0.7}
        c ,_ , =sess.run([cost ,optimizer] ,feed_dict=feed_dict)
        avg_cost += c/ total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

correct_prediction = tf.equal(tf.argmax(H, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Accuracy: ', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels,keep_prob:1.0}))