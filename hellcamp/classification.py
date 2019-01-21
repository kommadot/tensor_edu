import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-03-diabetes.csv',dtype=np.float32,delimiter=',')

x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

X = tf.placeholder(tf.float32,shape=[None,8])
Y = tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([8,1]))
b = tf.Variable(tf.random_normal([1]))

H = tf.sigmoid(tf.matmul(X,W) + b)

cost = -tf.reduce_mean(Y*tf.log(H)+(1-Y)*tf.log(1-H))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

predicted = tf.cast(H>0.5,tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),tf.float32))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(20001):
    sess.run(train,feed_dict={X:x_data,Y:y_data})
    if i%5000==0:
        h,p,a=sess.run([H,predicted,accuracy],feed_dict={X:x_data,Y:y_data})
        print(h)
        print(p)
        print(a)


'''

X = tf.placeholder(tf.float32,shape=[None,2])
Y = tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([2,1]))
b = tf.Variable(tf.random_normal([1]))

H = tf.sigmoid(tf.matmul(X,W) + b)


cost = -tf.reduce_mean(Y*tf.log(H) + (1-Y)*tf.log(1-H))

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

predicted = tf.cast(H>0.5,tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),tf.float32))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(20001):
    cost_val, _ = sess.run([cost,train],feed_dict={X:x_data,Y:y_data})
    if step % 5000 == 0:
        print(step,cost_val)

        h,c,a = sess.run([H,predicted,accuracy],feed_dict={X:x_data,Y:y_data})
        print(" H : ",h,"\nCorrect (Y):",c,"\nAccuracy: ",a)

'''