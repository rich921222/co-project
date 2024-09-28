import numpy as np

def Bin2Dec(bin_list):
    sum = 0
    num = 1
    list_len = len(bin_list)
    for i in range(list_len-1,-1,-1):
        sum += bin_list[i] * num
        num <<= 1
    return sum

def Dec2Bin(Dec_num, bits = 8):
    sum = []
    while Dec_num > 0:
        sum.append(Dec_num & 1)
        Dec_num >>= 1
    while len(sum) < bits:
        sum.append(0)
    sum.reverse()    
    return sum    

if __name__ == '__main__':
    dec = Bin2Dec(np.array([1, 0, 0, 1, 1, 0, 1, 1]))
    print(dec)

    bin = Dec2Bin(155)
    print(bin)