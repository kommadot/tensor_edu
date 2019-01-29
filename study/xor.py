import tensorflow as tf

x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [[0],[1],[1],[0]]

X = tf.placeholder(tf.float32,shape=[None,2])
Y = tf.placeholder(tf.float32,shape=[None,1])

W1 = tf.Variable(tf.random_normal([2,5]))
b1 = tf.Variable(tf.random_normal([5]))

W2 = tf.Variable(tf.random_normal([5,10]))
b2 = tf.Variable(tf.random_normal([10]))

W3 = tf.Variable(tf.random_normal([10,10]))
b3 = tf.Variable(tf.random_normal([10]))

W4 = tf.Variable(tf.random_normal([10,1]))
b4 = tf.Variable(tf.random_normal([1]))


with tf.name_scope("layer1") as scope:
    layer1 = tf.nn.relu(tf.matmul(X,W1)+b1)
with tf.name_scope("layer2") as scope:
    layer2 = tf.nn.relu(tf.matmul(layer1,W2)+b2)
with tf.name_scope("layer3") as scope:
    layer3 = tf.nn.relu(tf.matmul(layer2,W3)+b3)
H = tf.sigmoid(tf.matmul(layer3,W4)+b4)

cost = -tf.reduce_mean(Y*tf.log(H)+(1-Y)*tf.log(1-H))

optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

predict = tf.cast(H>=0.5,tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,Y),tf.float32))

H_hist = tf.summary.histogram("H",H)
cost_summ = tf.summary.scalar("cost",cost)

summary = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./logs')
    writer.add_graph(sess.graph)

    for i in range(20001):
        s,_= sess.run([summary,optimizer],feed_dict={X:x_data,Y:y_data})
        writer.add_summary(s,global_step=0.001)

        if i % 5000 == 0:
            print(sess.run(accuracy,feed_dict={X:x_data,Y:y_data}))
