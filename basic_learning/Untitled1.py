
# coding: utf-8

# # LSTM for Data Trance

# In[10]:


import tensorflow as tf
import numpy
import pandas as pd
import os
from tensorflow.contrib import rnn

config =tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

lr = 1e-4
batch_size = tf.placeholder(tf.int32)
input_size = 1
time_step_size = 120
hidden_size = 1
layer_num = 10
class_num = 2

_x = tf.placeholder(tf.float32, [None, 120])
y = tf.placeholder(tf.float32, [None, class_num])
keep_prob = tf.placeholder(tf.float32)

def load_data():
    train_path = "train.csv"
    test_path = "test.csv"
    train = pd.read_csv(train_path, names=["data","label"], header=0)
    train_y = train.pop("label")
    train_x = train
    
    test = pd.read_csv(test_path, names=["data", "label"], header=0)
    test_x, test_y = test.pop("data"), test.pop("label")
    
    return (train_x, train_y), (test_x, test_y)

def main(argv):
    (train_x, train_y),(test_x, test_y) = load_data()
    print train_x[0:50]
    
    
if __name__ == '__main__':
    tf.app.run(main)
    print "done"

