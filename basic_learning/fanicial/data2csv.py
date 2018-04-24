import random
import os

input_file = "data.txt"
value_train_file = "value_train.csv"
label_train_file = "label_train.csv"
value_test_file = "value_test.csv"
label_test_file = "label_test.csv"
data_length = 40
train_number = 150
test_number = 20
divid_number = 60#divid of train and test

def getlist():
    with open(input_file,'r') as putin:
        value_list = []
        label_list = []
        for lines in putin.readlines():
            value_list.append(lines[12:18])
            if lines[28] == '-':
                label_list.append(0)
            else:
                label_list.append(1)
    return value_list,label_list

def writelist(valuelist,labellist,mode):
    if mode == 1:#means train set
        putout_value = open(value_train_file,'w')
        putout_label = open(label_train_file,'w')
        count = train_number
    else:
        putout_value = open(value_test_file,'w')
        putout_label = open(label_test_file,'w')
        count = test_number
    for i in range(count):
        startindex = random.randint(data_length,len(valuelist)-1)
        for subnumber in range(data_length):
            if subnumber == 0:
                putout_value.write("%s" % valuelist[startindex])
            else:
                putout_value.write(",%s" % valuelist[startindex - subnumber])
        if labellist[startindex-data_length] == 1:
            putout_label.write("0,1\n")
        else:
            putout_label.write("1,0\n")
        putout_value.write("\n")
    putout_value.close()
    putout_label.close()
    print "write done"

if  __name__ == '__main__':
    value_list, label_list = getlist()
    writelist(value_list[0:divid_number],label_list[0:divid_number],0)
    writelist(value_list[divid_number:len(value_list)-1],label_list[divid_number:len(value_list)-1],1)
