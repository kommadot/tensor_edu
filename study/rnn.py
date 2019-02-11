import tensorflow as tf
import numpy as np
import pprint
import pandas as pd
sess = tf.InteractiveSession()
pp = pprint.PrettyPrinter(indent=4)

h = [1,0,0,0,0]
i = [0,1,0,0,0]
e = [0,0,1,0,0]
l = [0,0,0,1,0]
o = [0,0,0,0,1]
sequence_length=6
hidden_size = 5
input_dim=5
batch_size = 1

idx2char = ['h','i','e','l','o']
x_data = [[0,1,0,2,3,3]]
n_values = np.max(x_data)+2
b = np.eye(n_values)[x_data]
b=b[np.newaxis]
pp.pprint(b)
y_data = [[1,0,2,3,3,4]]

X = tf.placeholder(tf.float32,[None,sequence_length,input_dim])
Y = tf.placeholder(tf.int32,[None,sequence_length])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,state_is_tuple=True)
initial_state = cell.zero_state(batch_size,tf.float32)
output, _states = tf.nn.dynamic_rnn(cell,X,initial_state=initial_state,dtype=tf.float32)
weihts = tf.ones([batch_size,sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=output,targets=Y,weights=weihts)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(0.1).minimize(loss)

prediction = tf.argmax(output,axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        l,_ = sess.run([loss,train],feed_dict={X : b, Y : y_data})
        result = sess.run(prediction,feed_dict={X:b})
        print(i,"loss: ",l,"prediction: ",result,"true Y: ",y_data)

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("Prediction str: ",''.join(result_str))