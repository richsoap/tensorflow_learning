import pandas as pd
from absl import flags
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn

FLAGS=flags.FLAGS
time_step=30
rnn_unit=10
batch_size=256
input_size=1
output_size=1
lr=0.0006
X=tf.placeholder(tf.float32, [None,time_step,input_size])
Y=tf.placeholder(tf.float32, [None,output_size])
output_path = 0
keep_prob = tf.placeholder(tf.float32)
weights={
        'in':tf.Variable(tf.random_normal([input_size, rnn_unit])),
        'out':tf.Variable(tf.random_normal([rnn_unit, output_size]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1, shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1, shape=[output_size,]))}

def get_data():
    f = open('value_all.csv')
    data = np.array(pd.read_csv(f))
    data = data[::-1]
    plt.figure()
    plt.plot(data)
    #plt.show()
    f.close()
    normalize_data = (data-np.mean(data))/np.std(data)
    normalize_data = normalize_data[:,np.newaxis]
    return np.reshape(normalize_data,[-1,1])

def prepare_train(data):
    train_x,train_y=[],[]
    for i in range(len(data)-time_step-1):
        x = data[i:i+time_step]
        y = data[i+time_step+1]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    train_x = np.array(train_x)
    train_x = np.reshape(train_x,[-1,time_step,input_size])
    train_y = np.array(train_y)
    train_y = np.reshape(train_y,[-1,output_size])
    print np.size(train_x)
    return train_x, train_y

def def_lstm(_batch_size):
    w_in = weights['in']
    b_in = biases['in']
    input_basic = tf.reshape(X, [-1,input_size ,])
    input_rnn = tf.matmul(input_basic, w_in)+b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])
    cell = rnn.BasicLSTMCell(rnn_unit, forget_bias=1.0)
    ####
    cell = rnn.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob = 1.0)
    cell = rnn.MultiRNNCell([cell]*3,state_is_tuple=True)
    ####
    init_state = cell.zero_state(_batch_size, dtype=tf.float32)
    output_rnn,final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn,[-1,rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output,w_out)+b_out
    print ("predshape",np.shape(pred))
    pred = tf.reshape(pred, [-1,time_step])
    return pred[:,-1],final_states

def train_lstm(train_x, train_y):
    with tf.variable_scope('train',reuse = tf.AUTO_REUSE):
        pred,_ = def_lstm(batch_size) 
        print ("predshape",np.shape(pred))
        loss = tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y,[-1])))
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #for i in range(50000):
            for i in range(50000):
                start = random.randint(0,batch_size)
                end = start + batch_size
                while(end<len(train_x)-5):
                    _,loss_ = sess.run([train_op,loss], feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                    start += batch_size
                    end += batch_size
                if i%2500 == 0:
                    print(i,loss_)
                    print("Checkpoint Save: ",saver.save(sess, './fanical_lstm.ckpt', global_step = i))
            print("Model Save: ",saver.save(sess, './fanical_lstm.model'))
            ############################################################################################################
          #  s.system("sudo python lstm_conv.py ./fanical_lstm.model ./fanical_lstm_change.model")
#        vars_to_rename = {
#            "lstm/basic_lstm_cell/weights": "lstm/basic_lstm_cell/kernel",
#            "lstm/basic_lstm_cell/biases": "lstm/basic_lstm_cell/bias",
#        }
#        new_checkpoint_vars = {}
        #reader = tf.train.NewCheckpointReader(FLAGS.checkpoint_path)
        reader = tf.train.NewCheckpointReader('./fanical_lstm.model')
        for old_name in reader.get_variable_to_shape_map():
            print(old_name)
#            if old_name in vars_to_rename:
#                new_name = vars_to_rename[old_name]
#            else:
#                new_name = old_name
#        new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))
#        saver = tf.train.Saver(new_checkpoint_vars)
#        init = tf.global_variables_initializer()
#        with tf.Session() as sess:
#            sess.run(init)
#            print("Model Save: ",saver.save(sess, './fanical_lstm_change.model'))

def pre_lstm(train_x):
    train_x = np.reshape(train_x, [-1,1])
    with tf.variable_scope('train',reuse=tf.AUTO_REUSE):
        pred,_ = def_lstm(1)
        saver = tf.train.Saver()
        with tf.Session() as sess: 
            saver.restore(sess, './fanical_lstm.model')
            predict = []
            for i in range(len(train_x) - time_step - 1):
                next_seq = sess.run(pred, feed_dict={X:[train_x[i:i+time_step]]})
                predict.append(next_seq[-1])
        return predict

def main(argv):
    data = get_data()
    tx, ty = prepare_train(data)
    print np.size(tx)
    train_lstm(tx, ty)
    pre = pre_lstm(data)
    plt.figure()
    plt.plot(list(range(len(pre))),data[len(data)-len(pre)::],color='b')
    plt.plot(list(range(len(pre))),pre,color='r')
    plt.show()


if __name__ == '__main__':
    tf.app.run(main)
    print "done"
