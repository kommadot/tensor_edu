import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.Variable(1)
b = tf.Variable(2)
sess = tf.Session()
c = a*b
d = tf.add(a,c)
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(c))