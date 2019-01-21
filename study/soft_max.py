import tensorflow as tf
x_data = [[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]


X = tf.placeholder(shape=[None,4],dtype=tf.float32)
Y = tf.placeholder(shape=[None,3],dtype=tf.float32)

nb_classes = 3

W = tf.Variable(tf.random_normal([4,nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))


H = tf.nn.softmax(tf.matmul(X,W)+b)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(H),axis=1))

optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

predicted = tf.arg_max(H,1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,tf.arg_max(Y,1)),tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(20001):
        sess.run(optimizer,feed_dict={X:x_data,Y:y_data})

        if i % 5000 == 0 :
            print(sess.run([accuracy],feed_dict={X:x_data,Y:y_data}))
