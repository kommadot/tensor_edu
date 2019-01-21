import tensorflow as tf

x_data = [1,2,3,4]
y_data = [4,5,6,7]

X = tf.placeholder(tf.float32,shape=[None])
Y = tf.placeholder(tf.float32,shape=[None])

W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

H = W*x_data + b

cost = tf.reduce_mean(tf.square(H-y_data))

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        sess.run(train)

    print(sess.run([H,W,b]))