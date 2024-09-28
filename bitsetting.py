import numpy as np
import B2D

def bitget(num,index):
    arr = B2D.Dec2Bin(num)
    return arr[index]

def bitset(num,index,bit):
    arr = B2D.Dec2Bin(num)
    arr[index] = bit
    num = B2D.Bin2Dec(arr)
    return num

def lsbset(num,ch_arr):
    arr = B2D.Dec2Bin(num)
    a_len = len(arr)-1
    c_len = len(ch_arr)-1
    for i in range(len(ch_arr)):
        arr[a_len - i]  = ch_arr[c_len-i]
    num = B2D.Bin2Dec(arr)
    return num

if __name__ == "__main__":
    n = 255
    print(lsbset(n,np.array([0,0])))