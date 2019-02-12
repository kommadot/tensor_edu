import tensorflow as tf
import numpy as np

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dic = {w: i for i ,w in enumerate(char_set)}

data_dim = len(char_set)
hidden_size = len(char_set)
num_classed = len(char_set)
sequence_len = 10
learning_rate = 0.1

x_data = []
y_data = []

for i in range(0,len(sentence)-sequence_len):
    x_str = sentence[i:i + sequence_len]
    y_str = sentence[i+1:i+1+sequence_len]

    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    x_data.append(x)
    y_data.append(y)

batch_size = len(x_data)

X = tf.placeholder(tf.int32,[None,sequence_len])
Y = tf.placeholder(tf.int32,[None,sequence_len])

X_one_hot = tf.one_hot(X,num_classed)
print(X_one_hot)

cell = tf.contrib.rnn.BasicLSTMCell(hidden_size,state_is_tuple=True)

cell = tf.contrib.rnn.MultiRNNCell([cell]*2,state_is_tuple=True)
initial_state = cell.zero_state(batch_size,tf.float32)

outputs, _states = tf.nn.dynamic_rnn(cell,X_one_hot,dtype=tf.float32,initial_state=initial_state)

X_for_softmax = tf.reshape(outputs,[-1,hidden_size])

softmax_w = tf.get_variable("softmax_w",[hidden_size,num_classed])
softmax_b = tf.get_variable("softmax_b",[num_classed])
outputs = tf.matmul(X_for_softmax,softmax_w)+softmax_b
outputs = tf.reshape(outputs,[batch_size,sequence_len,num_classed])

weight = tf.ones([batch_size,sequence_len])

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs,targets=Y,weights=weight)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs,axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(500):
        _,l,results = sess.run([train,loss,outputs],feed_dict={X:x_data,Y:y_data})
        #results = sess.run(prediction,feed_dict={X:x_data})

        if i%500==0:
            for j,result in enumerate(results):
                index = np.argmax(result,axis=1)
                result_str = [char_set[c] for c in index]
                print(''.join(result_str))
    result = sess.run(outputs,feed_dict={X:x_data})
    for j,result in enumerate(results):
        index = np.argmax(result,axis=1)
        if j is 0:
            print(''.join([char_set[t] for t in index]),end='')
        else:
            print(char_set[index[-1]],end='')