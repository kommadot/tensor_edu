import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant(1)
b = tf.constant(2)
c = tf.add(a,b)
sess = tf.Session()

print(sess.run(c))