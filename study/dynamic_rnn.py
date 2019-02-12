import tensorflow as tf
import numpy as np


def MinMaxScaler(data):
    numerator = data - np.min(data,0)
    denominator = np.max(data,0) - np.min(data,0)
    return numerator/(denominator+1e-7)

data = np.loadtxt('./data-02-stock_daily.csv',dtype=np.float32,delimiter=',')
data = data[::-1]
data = MinMaxScaler(data)
x = data
y = data[:,[-1]]

input_dim = 5
timesteps = sequence_len = 7
output_dim = 1


dataX = []
dataY = []

for i in range(0,len(y)-sequence_len):
    _x = x[i:i+sequence_len]
    _y = y[i+sequence_len]
    dataX.append(_x)
    dataY.append(_y)

train_size = int(len(dataY)*0.7)
test_size = len(dataY) - train_size

trainX,testX = np.array(dataX[0:train_size]),np.array(dataX[train_size:len(dataX)])
trainY,testY = np.array(dataY[0:train_size]),np.array(dataY[train_size:len(dataY)])

X = tf.placeholder(tf.float32,[None,sequence_len,input_dim])
Y = tf.placeholder(tf.float32,[None,1])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=3,state_is_tuple=True)
cell = tf.contrib.rnn.MultiRNNCell([cell]*2,state_is_tuple=True)
outputs,_states = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1],output_dim,activation_fn=None)

loss = tf.reduce_sum(tf.square(Y_pred-Y))
train = tf.train.AdamOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run([train,loss],feed_dict={X:trainX,Y:trainY})

    testPredict = sess.run(Y_pred,feed_dict={X:testX})

    import matplotlib.pyplot as plt
    plt.plot(testY)
    plt.plot(testPredict)
    plt.show()