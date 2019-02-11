from .simplyrun import Model
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
sess = tf.Session()
m1 = Model(sess,"m1")

sess.run(tf.global_variables_initializer())
mnist = input_data.read_data_sets("samples/MNIST_data/" ,one_hot=True)

nb_classes = 10
keep_prob =tf.placeholder(tf.float32)
batch_size = 10
training_epochs = 1
for epoch in range(training_epochs):
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        m1.train(batch_xs,batch_ys)

print(m1.get_accuracy())