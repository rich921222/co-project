import numpy as np
import cv2
import math
import hashlib

#顯示圖片
def show(pic):#看照片
    cv2.imshow('pic',pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows
#轉灰階
def rgb2gray(img):
    gray_img = np.zeros((img.shape[0],img.shape[1]))
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            b,g,r = img[i][j]
            gray_img[i][j]=round(0.299*r+0.587*g+0.114*b)
    return gray_img.astype(np.uint8)

#進制
def bin2dec(num,bit=8):
    num = np.array(num)
    h = np.array(num).shape[0]
    w = np.array(num).shape[1]
    # print(h,w)
    tmp=0
    t=[]
    for i in range(h):
        for j in range(w):
            tmp=0
            for k in range(bit-1,-1,-1):        
                tmp +=int(num[i][j][k])*(2**((bit-1)-k))
            num[i][j]=''.join(str(tmp)) 
    return np.array(num,dtype='uint8')

def dec2bin(a):
    a = np.vectorize(lambda x: format(x, '08b'))(a)
    return a

#----------根據修改過後的像素推算出維持灰階值的綠色像素
#會回傳存在unsolvable case的 grayscale集合
def cal_green(img,gray):
    k=0
    U = []

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            tmp = round((gray[i][j]-0.299*img[i][j][2]-0.114*img[i][j][0])/0.587)
            if tmp < 0 or tmp > 255:
                U.append(gray[i][j])
                k+=1
    U = list(set(U))
    U.sort()
    print('U長=',len(U))
    return U
#----------input:(灰階影像,灰階值) output:相同灰階值的座標
def find_all_values(matrix, target):
    positions = []
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            if value == target:
                positions.append((row_index, col_index))
    return positions
#----------

#生成驗證碼
def XOR(str1,str2):
    if len(str1)!=len(str2):
        raise ValueError("Must have the same length")
    result = ''.join([str(int(a) ^ int(b)) for a, b in zip(str1, str2)])   
    return result

def generate_hash(r,b,gv,ii,len_r,len_b): #length是RB要遷入驗證碼長度的總和
    text = str(r)+str(b)+str(gv)+str(ii)
    #判斷hash碼要補多少
    len_of_acc = len_r+len_b
    len_of_zero = len_of_acc
    while(len_of_zero<128):
        len_of_zero *= 2
    len_of_zero-=128

    data = text.encode('utf-8')
    #====
    # 创建一个MD5哈希对象
    md5_hash = hashlib.md5()

    # 更新哈希对象以处理数据
    md5_hash.update(data)

    # 获取MD5哈希值的十六进制字符串表示形式
    hash_value_hex = md5_hash.hexdigest()

    # 将十六进制字符串转换为固定位数的格式（例如，添加前导零以保证32位）
    formatted_hash_value = hash_value_hex.zfill(32)

    result = bin(int(formatted_hash_value,16))[2:]
    while(len(result)!=128): #16進制轉2進制 前方為0的會被去掉
        result='0'+result
    while(len_of_zero):
        len_of_zero-=1
        result=result+'0'
    # print(len(result))#檢查hash完長度是否符合
    while(len(result)!=len_of_acc):
        result = XOR(result[:len(result)//2],result[len(result)//2:])
    # print(type(result))
    return result

def hash_unsolvable(ii,gv,r,g,b,len_g): #length是RB要遷入驗證碼長度的總和
    text = str(ii)+str(gv)+str(r)+str(g)+str(b)
    #判斷hash碼要補多少
    len_of_acc = len_g
    len_of_zero = len_of_acc
    while(len_of_zero<128):
        len_of_zero *= 2
    len_of_zero-=128

    data = text.encode('utf-8')
    #====
    # 创建一个MD5哈希对象
    md5_hash = hashlib.md5()

    # 更新哈希对象以处理数据
    md5_hash.update(data)

    # 获取MD5哈希值的十六进制字符串表示形式
    hash_value_hex = md5_hash.hexdigest()

    # 将十六进制字符串转换为固定位数的格式（例如，添加前导零以保证32位）
    formatted_hash_value = hash_value_hex.zfill(32)

    result = bin(int(formatted_hash_value,16))[2:]
    while(len(result)!=128): #16進制轉2進制 前方為0的會被去掉
        result='0'+result
    while(len_of_zero):
        len_of_zero-=1
        result=result+'0'
    # print(len(result))#檢查hash完長度是否符合
    while(len(result)!=len_of_acc):
        result = XOR(result[:len(result)//2],result[len(result)//2:])
    # print(type(result))
    return result

def hash_all_pixel(img,len_r,len_b):#將整張照片生成驗證碼
    gray= rgb2gray(img)
    divid4 = (img//(2**len_r))*(2**len_r) #(b,g,r)
    code = []
    for i in range(gray.shape[0]):
        code.append([])
        for j in range(gray.shape[1]):
            ii = i*gray.shape[1]+j+1
            cc = generate_hash(divid4[i][j][2],divid4[i][j][0],gray[i][j],ii,len_r,len_b)
            code[i].append(cc)
    code = np.array(code)
    print("產生驗證碼完成")
    return code

#####[論文unsolvable使用方法]---------
def min_distance_for_unsolvable_pixel(new,pixel):
    min_distance = float('inf')  # 用于存储最小距离，初始化为正无穷大
    min_group = None  # 用于存储最小距离对应的组
    for group in new:
        distance = math.sqrt((group[0] - pixel[0])**2 + (group[1] - pixel[1])**2 + (group[2] - pixel[2])**2)
        if distance < min_distance:
            min_distance = distance
            min_group = group
    return min_group
    
def generate_perturbed_pairs(pixel,ii,alpha): 
    len_b=2
    b, g, r = pixel  # 从像素中获取g和r值
    bb=b//4*4 #2bit LSB = 00
    perturbed_pairs = []
    grayscale=round(0.299*r+0.587*g+0.114*b)
    for dg in range(-alpha, alpha + 1):
        for dr in range(-alpha, alpha + 1):
            for db in range(-alpha, alpha + 1):
                perturbed_g = g + dg
                perturbed_r = r + dr
                perturbed_b = bb+ db*4
                if round(0.299*perturbed_r+0.587*perturbed_g+0.114*perturbed_b)==grayscale:
                    if round(0.299*perturbed_r+0.587*perturbed_g+0.114*(perturbed_b+3))==grayscale:
                        perturbed_pairs.append((perturbed_b,perturbed_g, perturbed_r))
    perturbed_pairs = [pair for pair in perturbed_pairs if all(val >= 0 for val in pair)]
    acc_unsolvable=[]
    new=np.array(perturbed_pairs)
    for i in range(len(perturbed_pairs)):
        b,g,r=perturbed_pairs[i]
        acc_unsolvable.append(hash_unsolvable(ii,grayscale,r,g,b,len_b))
        if grayscale == round(0.299*r+0.587*g+0.114*int(Embedding(str(dec2bin(b)),acc_unsolvable[i],length=len_b),2)):
            new[i][0]=int(Embedding(str(dec2bin(b)),acc_unsolvable[i],length=len_b),2)
    valid_data = new[np.all((new >= 0) & (new <= 255), axis=1)]
    return min_distance_for_unsolvable_pixel(valid_data,pixel)
#------------------------------------

#HONG2023方法
#小function
def bgr_for_same_grayscale(gv):
    results = []
    for r in range(256):
        for g in range(256):
            for b in range(256):       
                tmp = 0.299 * r + 0.587 * g + 0.114*b
                if round(tmp) ==gv :
                    results.append([b,g,r])
    return results

def sorted_with_min_distance (color_list,pixel):
    sorted_color_list = sorted(color_list, key=lambda color: math.sqrt((color[0] - pixel[0])**2 + (color[1] - pixel[1])**2 + (color[2] - pixel[2])**2))
    return sorted_color_list
#主要
def MSBA(pixel,len_r,len_b,ii):
    alpha = 1
    pair = []
    bgr_pair =[]
    b,g,r = pixel
    gv= round(0.299*r+0.587*g+0.114*b)
    rr = r // (2**len_r)
    bb = b // (2**len_b)

    for i in range(-alpha,alpha+1):
        for j in range(-alpha,alpha+1):
            if bb+i>=0 and bb+i<(2**(8-len_b))  and rr+j>0 and rr+j<(2**(8-len_r)):
                pair.append([bb+i,rr+j])
                tmp_bb,tmp_rr = str(dec2bin((bb+i)*(2**len_b))), str(dec2bin((rr+j)*(2**len_r)))
                hash=generate_hash(rr+j,bb+i,gv,ii,len_r,len_b)
                gg = round((gv-0.299*int(Embedding(tmp_rr,hash[:len_r],length=len_r),2)-0.114*int(Embedding(tmp_bb,hash[(len(hash)-len_b):],length=len_b),2))/0.587)
                if  gg>=0 and gg<=255:
                    bgr_pair.append([int(Embedding(tmp_bb,hash[(len(hash)-len_b):],length=len_b),2),gg,int(Embedding(tmp_rr,hash[:len_r],length=len_r),2)])

    return min_distance_for_unsolvable_pixel(bgr_pair,pixel)

def PSS(pixel,len_r,len_b,ii):
    b,g,r = pixel
    gv = round(0.299 * r + 0.587 * g + 0.114*b)
    color_list = bgr_for_same_grayscale(gv)
    sorted_color_list = sorted_with_min_distance(color_list,pixel)
    for i in sorted_color_list:
        new = MSBA(i,len_r,len_b,ii)
        if new != None:
            return new 
    return new
        
   
def CRS(pixel,ii,delta=1,alpha=2):# delta決定RG pair的範圍 alpha決定Bm pair的範圍
    b,g,r = pixel
    gv = round(0.299 * r + 0.587 * g + 0.114*b)
    bm = b//4
    gr_pair = []
    bm_pair =[]
    results = []
    for i in range(-delta,delta+1):
        for j in range(-delta,delta+1):
            if g+i>=0 and g+i<=255 and r+j>=0 and r+j<=255:
                gr_pair.append([g+i,r+j])
    gr_pair = [pair for pair in gr_pair if all(val >= 0 for val in pair)]

    for i in range(-alpha,alpha+1):
        if bm+i >= 0 and bm+i<64:
            bm_pair.append(bm+i)
    #GENERATE ALL RGB THAT HAVE THE SAME GRAYSCALE
    for gg,rr in gr_pair:
        for bb in bm_pair:
            hash_code=hash_unsolvable(rr,gg,bb,ii,gv,2)
            tmp_b = int(Embedding(str(dec2bin(bb*4)),hash_code,length=2),2)
            if round(0.114*tmp_b+0.587*gg+0.299*rr) == gv:
                results.append([tmp_b,gg,rr])
    return min_distance_for_unsolvable_pixel(results,pixel)

#####[我的unsolvable使用方法]---------
def eee(a,b,x): #inverse
    error1 = abs(a-x)
    error2 = abs(b-x)
    if error1 < error2:
        return a
    else:
        return b
def find_bgr_for_x(gv,bb): #return {(b,g,r)}that have the same grayscale target_x
    results = []

    for r in range(256):
        for g in range(256):           
            x = 0.299 * r + 0.587 * g + 0.114*bb
            if round(x)==gv:
                results.append((g, r))
    return results
def convert_to_n_by_3(arr, value):
    n = len(arr)
    result = []

    for i in range(n):
        row = [value]  # 在新数组的第一列添加 value
        row.extend(arr[i])  # 将输入数组的值扩展到新数组
        result.append(row)

    return result
def proposed_hash_unsolvable(ii,gv,b,len_bb): #length是RB要遷入驗證碼長度的總和
    text = str(ii)+str(gv)+str(b)
    #判斷hash碼要補多少
    len_of_acc = len_bb
    len_of_zero = len_of_acc
    while(len_of_zero<128):
        len_of_zero *= 2
    len_of_zero-=128

    data = text.encode('utf-8')
    #====
    # 创建一个MD5哈希对象
    md5_hash = hashlib.md5()

    # 更新哈希对象以处理数据
    md5_hash.update(data)

    # 获取MD5哈希值的十六进制字符串表示形式
    hash_value_hex = md5_hash.hexdigest()

    # 将十六进制字符串转换为固定位数的格式（例如，添加前导零以保证32位）
    formatted_hash_value = hash_value_hex.zfill(32)

    result = bin(int(formatted_hash_value,16))[2:]
    while(len(result)!=128): #16進制轉2進制 前方為0的會被去掉
        result='0'+result
    while(len_of_zero):
        len_of_zero-=1
        result=result+'0'
    # print(len(result))#檢查hash完長度是否符合
    while(len(result)!=len_of_acc):
        result = XOR(result[:len(result)//2],result[len(result)//2:])
    # print(type(result))
    return result
def proposed_unsolvable_case(pixel,ii,len_bb):
    b,g,r=pixel
    gv = round(r*0.299+g*0.587+b*0.114)
    hash_code=proposed_hash_unsolvable(ii,gv,(b//(2**len_bb))*(2**len_bb),len_bb)
    bb = int(Embedding(str(dec2bin(b)),hash_code,length=len_bb),2)
    sumof_g_r = r*0.299+g*0.587+b*0.114 -0.114*bb
    pair_g_r = find_bgr_for_x(gv,bb)
    pair = convert_to_n_by_3(pair_g_r,bb)
    return min_distance_for_unsolvable_pixel(pair,(b,g,r))
def gggg(pixel,ii,alpha):  #和generate_perturbed_pairs差別在於b and b+3皆有相同驗證碼的限制
    len_b=2
    b, g, r = pixel  
    bb=b//4*4 #2bit LSB = 00
    perturbed_pairs = []
    grayscale=round(0.299*r+0.587*g+0.114*b)
    for dg in range(-alpha, alpha + 1):
        for dr in range(-alpha, alpha + 1):
            for db in range(-alpha, alpha + 1):
                perturbed_g = g + dg
                perturbed_r = r + dr
                perturbed_b = bb+ db*4
                if round(0.299*perturbed_r+0.587*perturbed_g+0.114*perturbed_b)==grayscale:    
                    perturbed_pairs.append((perturbed_b,perturbed_g, perturbed_r))
    perturbed_pairs = [pair for pair in perturbed_pairs if all(val >= 0  for val in pair)]
    acc_unsolvable=[]
    new=np.array(perturbed_pairs)

    for i in range(len(perturbed_pairs)):
        b,g,r=perturbed_pairs[i]
        acc_unsolvable.append(hash_unsolvable(ii,grayscale,r,g,b,len_b))
        if grayscale == round(0.299*r+0.587*g+0.114*int(Embedding(str(dec2bin(b)),acc_unsolvable[i],length=len_b),2)):
            new[i][0]=int(Embedding(str(dec2bin(b)),acc_unsolvable[i],length=len_b),2)
    valid_data = new[np.all((new >= 0) & (new <= 255), axis=1)]

    return min_distance_for_unsolvable_pixel(valid_data,pixel)
#驗證階段==================
def check_tamper_num(img,tamper_img):
    print(img.shape,tamper_img.shape)
    difference = np.abs(img - tamper_img)
    bw_difference = np.any(difference > 0, axis=2).astype(np.uint8)
    count_of_ones = np.count_nonzero(bw_difference)
    print('竄改pixel數量:',count_of_ones)
    print('percentage of tampering:',format(count_of_ones/(img.shape[0]*img.shape[1])*100,'.2f'),'%')
    return bw_difference*255

def compare_with_groundtruth(ground_truth,find):
    ground_truth = np.repeat(ground_truth[:, :, np.newaxis], 3, axis=2)
    # 计算 ground_truth 中值为 255 的像素数量
    total_255_in_ground_truth = np.sum(ground_truth == 255)

    # 计算 ground_truth 中值为 255 但 find 为 0 的像素数量
    condition_255_in_ground_truth_and_0_in_find = np.sum((ground_truth == 255) & (find == 0))

    # 计算 find 中值为 255 但 ground_truth 为 0 的像素数量
    condition_255_in_find_and_0_in_ground_truth = np.sum((find == 255) & (ground_truth == 0))

    # 输出结果
    print("Total pixels with value 255 in ground_truth:", total_255_in_ground_truth)
    print("ground_truth is 255 and find is 0:", condition_255_in_ground_truth_and_0_in_find)
    print("find is 255 and ground_truth is 0:", condition_255_in_find_and_0_in_ground_truth)


#=========================


#決定是否需要反轉
def decide_invert(pattern_count,error_count):
    aa=np.where(error_count > pattern_count/2,1,0)
    # print(f'{pattern_count=}')
    # print(f'{error_count=}')
    return aa
    
#====嵌入
#pattern_matrix 是利用rc4加密完的陣列 取他的bit當作pattern
def invert_LSB_via_invert_info(pattern_matrix,binary_matrix,bit_len,message,pattern_type,invert_info):
    h = binary_matrix.shape[0]
    w = binary_matrix.shape[1]
    img =np.array(binary_matrix)
    point=0 #確保不超過message長度
    for i in range(h):
        for j in range(w):
            img[i][j]=Embedding(binary_matrix[i][j],message[point],invert_info[np.where(pattern_type == pattern_matrix[i][j][7-bit_len:7])[0][0]])
            point+=1
            if point == len(message):
                break
        if point == len(message):
                break
    return img
#逐一嵌入 invert=1時翻轉嵌入message,invert=0直接嵌入並回傳更改完的Pixel
#傳送的message為message[point]單一個bit
def QOO(str): #反轉遷入的驗證碼
    s =''
    for i in str :
        if i =='1':
            s+='0'
        else:
            s+='1'
    return s
    
def Embedding(pixel,message,invert=0,length=1): 
    # print(pixel)
    if invert == 0:
        return pixel[:8-length]+str(message)
    else :
        return pixel[:8-length]+str(QOO(message))
    
#==============================實驗
#PSNR
def PSNR(f, g):
    nr, nc = f.shape[:2]
    MSE = 0.0
    for x in range(nr):
        for y in range(nc):
            MSE += (float(f[x, y]) - float(g[x, y]))** 2
    MSE /= (nr * nc)
    PSNR = 10 * np.log10((255 * 255) / MSE)
    return PSNR
def three_dim_psnr(img1,img2):
    return (PSNR(img1[:,:,0],img2[:,:,0])+PSNR(img1[:,:,1],img2[:,:,1])+PSNR(img1[:,:,2],img2[:,:,2]))/3
def compare(f,g):
    from skimage.metrics import structural_similarity
    psnr = PSNR(f,g)
    ssim = structural_similarity(f,g)
    print(f'{psnr=},{ssim=}')
    
#==============================