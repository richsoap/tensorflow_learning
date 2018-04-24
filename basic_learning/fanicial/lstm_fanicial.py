import tensorflow as tf
import numpy as np
import os
import pandas as pd
from tensorflow.contrib import rnn
import random

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

lr = 1e-3
batch_size = tf.placeholder(dtype=tf.int32)
_batch_size = 128
test_batch_size = 128
inputs_size = 1
timestep_size = 40
hidden_size = 1 
layer_num = 10
class_num = 2

train_value = "value_train.csv"
train_label = "label_train.csv"
test_value = "value_test.csv"
test_label = "label_test.csv"

_x = tf.placeholder(tf.float32, [None, 40])
y = tf.placeholder(tf.float32, [None, class_num])
keep_prob = tf.placeholder(tf.float32)

# need a get batch function
# maybe argv can be useful? like repeat step?

def getset():
    train_x = pd.read_csv(train_value, header=None)
    train_y = pd.read_csv(train_label, header=None)
    test_x = pd.read_csv(test_value, header=None)
    test_y = pd.read_csv(test_label, header=None)
    print train_x.shape
    print train_y.shape
    return (train_x, train_y), (test_x, test_y)
'''
def get_batch(value, label, size):
    index = random.randint(0,len(value)-1)
    length = len(value)
    temp_value = np.transpose(value)
    temp_label = np.transpose(label)
    batch_value = temp_value[index]
    batch_label = temp_label[index]
    for i in range(size - 1):
        index = random.randint(0,length-1)
        np.column_stack((batch_value,temp_value[i]))
        np.column_stack((batch_label,temp_label[i]))
        print batch_value.shape
    print batch_value.shape
   # return (np.transpose(batch_value),np.transpose(batch_label))
    return (batch_value, batch_label)    
'''
def get_batch(value, label, size):
    temp_value = np.transpose(value)
    temp_label = np.transpose(label)
    length = len(temp_label)
    batch_value = np.zeros([size, timestep_size], dtype=np.float32)
    batch_label = np.zeros([size, class_num], dtype=np.float32)
    for i in range(size):
        index = random.randint(0, length-1)
        batch_value[i] = temp_value[index]
        batch_label[i] = temp_label[index]
    return [batch_value, batch_label]


def main(argv):
    (train_x, train_y), (test_x, test_y) = getset()
    x = tf.reshape(_x,[-1,40,1])
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    mlstm_cell = rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)
    init_state = mlstm_cell.zero_state(_batch_size,dtype=tf.float32)

    outputs = list()
    state = init_state
    with tf.variable_scope('RNN'):
        for timestep in range(timestep_size):
            if(timestep > 0):
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = mlstm_cell(x[:, timestep, :],state)
            outputs.append(cell_output)
    h_state = outputs[-1]

    #outputs, state = tf.nn.dynamic_rnn(cell=mlstm_cell,inputs=x,dtype=tf.float32)
    #print outputs.shape
    #h_state = outputs[-1]
    
    W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1),dtype=tf.float32)
    bias = tf.Variable(tf.constant(0,1,shape=[class_num]), dtype=tf.float32)
    hwmat = tf.matmul(h_state, W)
    y_pre = tf.nn.softmax(hwmat+bias)

    cross_entropy = -tf.reduce_mean(y*tf.log(y_pre))

    train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))
    print correct_prediction
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"))
    print accuracy
    sess.run(tf.global_variables_initializer())
    for i in range(200000):
        if i==0:
            print "start"
        batch = get_batch(train_x,train_y,_batch_size)
        sess.run(train_op, feed_dict={_x:batch[0], y:batch[1], keep_prob: 0.5, batch_size: _batch_size})
        if(i+1)%100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={_x:batch[0], y:batch[1], keep_prob:1.0, batch_size:_batch_size})
            print "step %d, accuracy %g" % ((i+1), train_accuracy)
            batch = get_batch(test_x,test_y,test_batch_size)
            print "test accuracy %g" % sess.run(accuracy, feed_dict={_x:batch[0],y:batch[1], keep_prob:1.0, batch_size: test_batch_size})

if __name__ == '__main__':
    tf.app.run(main)
    print "done"

