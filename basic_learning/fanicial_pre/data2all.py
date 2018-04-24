import random
import os

input_file = "data.txt"
out_file = "value_all.csv"
data_length = 40
train_number = 150
test_number = 20
divid_number = 60#divid of train and test

def getlist():
    with open(input_file,'r') as putin:
        value_list = []
        for lines in putin.readlines():
            value_list.append(lines[12:18])
    return value_list

def writelist(valuelist):
    output_file = open(out_file, 'w')
    for i in range(len(valuelist)):
        output_file.write("%s\n" % valuelist[i])
    output_file.close()
    print "write done"
    

if  __name__ == '__main__':
    value_list = getlist()
    writelist(value_list)
