import tensorflow as tf
import numpy as np
import pprint

sample = "if you want you"
idx2char = list(set(sample))
char2idx = {c:i for i, c in enumerate(idx2char)}
sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]
ses = tf.InteractiveSession()
ses.run(tf.global_variables_initializer())
sequence_length = len(sample)-1
num_classes = len(char2idx)
rnn_hidden_size = len(char2idx)
batch_size = 1

X = tf.placeholder(tf.int32,[None,sequence_length])
Y = tf.placeholder(tf.int32,[None,sequence_length])

X_one_hot = tf.one_hot(X,num_classes)

cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size,state_is_tuple=True)
initial_state = cell.zero_state(batch_size,tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell,X_one_hot,initial_state=initial_state,dtype=tf.float32)
weights = tf.ones([batch_size,sequence_length])
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(weights.eval())

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs,targets=Y,weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(0.01).minimize(loss)

prediction = tf.argmax(outputs,axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l,_ = sess.run([loss,train],feed_dict={X:x_data,Y:y_data})
        result = sess.run(prediction,feed_dict={X:x_data})

        if i%100==0:
            result_str = [idx2char[c] for c in np.squeeze(result)]
            print(''.join(result_str))