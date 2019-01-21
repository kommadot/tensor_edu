import tensorflow as tf
import numpy as np

x_train = np.random.random((1000,1))
y_train = x_train*2 + np.random.random((1000,1))/3.0

x_test = np.random.random((100,1))
y_test = x_test *2 +np.random.random((100,1))/3.0

