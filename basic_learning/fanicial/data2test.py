import random
import os

input_file = "data.txt"
value_test_file = "value_test.csv"
label_test_file = "label_test.csv"
data_length = 40

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

def writelist(valuelist,labellist):
    putout_value = open(value_test_file,'w')
    putout_label = open(label_test_file,'w')
    today = 1
    for i in range(len(value_list) - data_length - 1):
        if((today == 1 and label_list[i] == 0) or (today == 0 and label_list[i] == 1)):
            for index in range(data_length):
                if index == 0:
                    putout_value.write("%s" % value_list[i + data_length - 1])
                else:
                    putout_value.write(",%s" % value_list[i + data_length - index - 1])
            putout_value.write("\n")
            if today == 1:
                putout_label.write("0,1\n")
            else:
                putout_label.write("1,0\n")
            today = label_list[i]
    putout_value.close()
    putout_label.close()
    print "write done"

if  __name__ == '__main__':
    value_list, label_list = getlist()
    writelist(value_list,label_list)
