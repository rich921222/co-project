import cv2
import numpy as np
from invert_LSB_module import *
#####[論文主程式]---------
def hong_method(img,len_r,len_b,alpha):
    gray_img = rgb2gray(img)
    img_divid4 = img//4*4#去除1,2的LSB(除以4)
    bin_matrix = dec2bin(img)
    authentication_code = hash_all_pixel(img,len_r,len_b) 
    embedded_matrix = np.zeros((img.shape)) 
    hong_embedded_num = 0
    for i in range(bin_matrix.shape[0]):
        for j in range(bin_matrix.shape[1]):
            embedded_matrix[i][j][2] = int(Embedding(bin_matrix[i][j][2],authentication_code[i][j][:len_r],length=len_r),2)#red
            embedded_matrix[i][j][0] = int(Embedding(bin_matrix[i][j][0],authentication_code[i][j][len_r:],length=len_b),2)#blue
    U = cal_green(embedded_matrix,gray_img)
    #EMBEDDING 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ii = i*gray_img.shape[1]+j+1
            if gray_img[i][j] in U: #使用第二種方法
                embedded_matrix[i][j]=generate_perturbed_pairs(img[i][j],ii,alpha)
                hong_embedded_num += 2
            else : #使用第一種方法
                embedded_matrix[i][j][1]=round((gray_img[i][j]-0.299*embedded_matrix[i][j][2]-0.114*embedded_matrix[i][j][0])/0.587)
                hong_embedded_num += len_r+len_b
                
    # print(f'{U=}')
    cc=0
    for target in U:
        results = find_all_values(gray_img, target)
        cc+=len(results)
    # print('論文方法unsolvable pixel:',cc)
    return embedded_matrix.astype(np.uint8),hong_embedded_num
#####[2023主程式]---------
def hong2023_main(img,len_r,len_b):
    new_img = np.zeros((img.shape))
    hey,heyhey=[] , [] 
    hong2023_embedded_num = 0
    for i in range(img.shape[0]):
        # if i % 50 == 0:
            # print(i)#看進度
        for j in range(img.shape[1]):
            ii =  i*img.shape[1]+j+1
            result1 = MSBA(img[i][j],len_r,len_b,ii)
            if result1 != None:
                new_img[i][j] = result1
                hong2023_embedded_num += (len_r+len_b)
            else :
                result2 = PSS(img[i][j],len_r,len_b,ii)
                if result2 !=None:
                    new_img[i][j] = result2
                    hey.append([i,j])
                    hong2023_embedded_num += (len_r+len_b)
                else :
                    new_img[i][j] = CRS(img[i][j],ii)
                    heyhey.append([i,j])
                    hong2023_embedded_num += 2
    # print('test')
    # print('2023Hong_PSS數量:',len(hey),'CRS數量:',len(heyhey))
    return  new_img.astype(np.uint8),hong2023_embedded_num,heyhey

#####[Propose主程式]---------
def propose_main(img,len_r,len_b,alpha):
    gray_img = rgb2gray(img)
    img_divid4 = img//4*4#去除1,2的LSB(除以4)
    bin_matrix = dec2bin(img)
    authentication_code = hash_all_pixel(img,len_r,len_b) 
    second_matrix = np.zeros((img.shape)) 
    proposed_embedded_num = 0 
    len_bb= (len_b+len_r)//2
    for i in range(bin_matrix.shape[0]):
        for j in range(bin_matrix.shape[1]):
            a = int(Embedding(bin_matrix[i][j][2],authentication_code[i][j][:len_r],length=len_r),2)#red
            b = int(Embedding(bin_matrix[i][j][2],authentication_code[i][j][:len_r],1,length=len_r),2)#red
            second_matrix[i][j][2] = eee(a,b,img[i][j][2])
            c = int(Embedding(bin_matrix[i][j][0],authentication_code[i][j][len_r:],length=len_b),2)#blue
            d = int(Embedding(bin_matrix[i][j][0],authentication_code[i][j][len_r:],1,length=len_b),2)#blue
            second_matrix[i][j][0] = eee(c,d,img[i][j][0])        
    U_second = cal_green(second_matrix,gray_img)
    U_second_un=[]
    for target in U_second:
        results = find_all_values(gray_img, target)
        for i,j in results:
            if proposed_unsolvable_case(img[i][j],i*gray_img.shape[1]+j+1,len_bb=len_bb)==None:
                U_second_un.append(gray_img[i][j])
    U_second_un = list(set(U_second_un))
    # print(f'{U_second=}')
    # print(f'{U_second_un=}')
    #embedding第一種方法使用lsb顛倒 第二種方法使用len_bb=(len_r+len_b)//2 and 2
    tmp=0           
    tmp1=0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ii = i*gray_img.shape[1]+j+1
            if gray_img[i][j] in U_second_un:#嵌入長度只有2
                second_matrix[i][j]=gggg(img[i][j],ii,alpha)#propose方法
                proposed_embedded_num += 2
                tmp1+=1
            elif gray_img[i][j] in U_second: #使用第二種方法
                second_matrix[i][j]=proposed_unsolvable_case(img[i][j],ii,len_bb)#propose方法
                proposed_embedded_num += (len_r+len_b)/2
                tmp+=1
            else : #使用第一種方法
                second_matrix[i][j][1]=round((gray_img[i][j]-0.299*second_matrix[i][j][2]-0.114*second_matrix[i][j][0])/0.587)
                proposed_embedded_num += len_r+len_b
    

    # print('Proposed unsolvable pixel(len_r+len_b)/2:',tmp,'len2=',tmp1)
    
    return  second_matrix.astype(np.uint8),proposed_embedded_num