import numpy as np
import matplotlib.pyplot as plt
from xxhash import xxh32
import B2D
from skimage import io

def APPM_RT256():
    RT = []
    for i in range(256):
        row = []
        count = (i*60)%256
        for j in range(256):
            row.append(count)
            count += 1
            if count >= 256:
                count -= 256
        RT.append(row)
    RT = np.array(RT)
    return RT

def NearestPoint():
    def distance_from_origin(point):
        x, y = point
        return x**2 + y**2
    points = []
    for x in range(-16, 17):
        for y in range(-16, 17):
                points.append((x, y)) 
    points.sort(key=distance_from_origin) 

    return np.array(points)

def Find(RT,NP,r,b,g):
    for i in range(len(NP)):
        if(r+NP[i,0] > 255 or r+NP[i,0] < 0):
            continue
        if(b+NP[i,1] > 255 or b+NP[i,1] < 0):
            continue        
        if(RT[r+NP[i,0],b+NP[i,1]] == g):
            return np.array([r+NP[i,0],b+NP[i,1]])
        
def Sudoku(B):
    x = int(np.sqrt(B))
    small_RT = np.array(np.random.choice(range(B), B, replace=False)).reshape(x, x)
    RT = np.zeros((256,256), dtype=int)  
    offset = True
    for i in range(0, 256, x):
        offset = not offset
        for j in range(0, 256, x):
            if(not offset):
                RT[i:i+x, j:j+x] = small_RT
            else:
                if(j == 0):
                    RT[i:i+x, 0:x>>1] = small_RT[0:x,x>>1:x]
                    RT[i:i+x, 256-(x>>1):256] = small_RT[0:x,0:x>>1]
                else:
                    RT[i:i+x, j-(x>>1):j-(x>>1)+x] = small_RT
                
    return RT 

def hashB(npArray,bits):
    return np.mod(xxh32(npArray).intdigest(), 2**bits)

def fold(Dec_num,bit):
    k = B2D.Dec2Bin(Dec_num)
    l = len(k)
    while(l > bit):
        l >>= 1
        a = k[:l]
        b = k[l:]
        k = []
        for i in range(l):
            k.append(a[i]^b[i])
    return B2D.Bin2Dec(k)

def perturbed(r,g,a):
    possible = []
    for i in range(-1*a,a):
        for j in range(-1*a,a):
            if(not (r+i < 0 or r+i > 255 or g+j < 0 or g+j > 255)):
                possible.append([r+i,g+j])
    return np.array(possible)

def remedy(r,g,b,ac):
    Gray = round(r*0.299+g*0.587+b*0.114)
    rg_poss = perturbed(r,g,2)
    b_poss = []
    origin_b = b
    b >>= 2
    b <<= 2
    for i in range(-1,2):
        if(b + i*4 >= 0 and b + i*4 <= 255):
            b_poss.append(b + i*4)

    b_poss = np.array(b_poss)

    tar = None
    distance = np.inf
    for i in range(len(rg_poss)):
        for j in range(len(b_poss)):
            if(Gray == round(rg_poss[i,0]*0.299+rg_poss[i,1]*0.587+b_poss[j]*0.114) and
               Gray == round(rg_poss[i,0]*0.299+rg_poss[i,1]*0.587+(b_poss[j]+3)*0.114)):
                b_poss[j] >>= 2
                b_poss[j] <<= 2
                b_poss[j] += ac
                
                new_d = ((rg_poss[i,0]-r)**2 + (rg_poss[i,1]-g)**2 + (b_poss[j]-origin_b)**2)**0.5

                if(new_d < distance):
                    distance = new_d
                    tar = [rg_poss[i,0],rg_poss[i,1],b_poss[j]]

    return tar

import pandas as pd
def AVGI(Graph):
    path = 'image/'+Graph
    I=io.imread(path +r'.tiff')
    Stego = I.copy()
    df = pd.read_csv('RT.csv')
    RT_table = df.to_numpy()
    NearestP = NearestPoint()

    p = 0
    MSE = 0
    F = 0
    N = 0
    X = 0

    for i in range(Stego.shape[0]):
        for j in range(Stego.shape[1]):
            Gray = I[i,j,0]*0.299+I[i,j,1]*0.587+I[i,j,2]*0.114
            G_round = round(Gray)
            ac = G_round    
            
            k = Find(RT_table,NearestP,Stego[i,j,0],Stego[i,j,2],ac)          

            # print(f"i: {i} j: {j} k:{k}")
            # print(Stego[i,j,2],Stego[i,j+1,2],k)
            # print(f"i: {i}, j: {j}, I:({I[i,j,0]},{I[i,j,2]}), k:{k}")

            Stego[i,j,0] = k[0]
            Stego[i,j,2] = k[1]  
            g_bar = int((Gray - 0.299*k[0] - 0.114*k[1])/0.587)
            if(round(0.299*k[0]+0.587*g_bar+0.114*k[1]) < round(Gray)):
                g_bar += 1
            elif(round(0.299*k[0]+0.587*g_bar+0.114*k[1]) > round(Gray)):
                g_bar -= 1

            if(g_bar > 255):
                p += 1
                g_bar = 510 - abs(int((G_round - 0.299*k[0] - 0.114*k[1])/0.587))
            elif(g_bar < 0):
                p += 1
                g_bar = abs(int((G_round - 0.299*k[0] - 0.114*k[1])/0.587))
                
            Stego[i,j,1] = g_bar

            # print(f"I:{I[i,j]}, S:{Stego[i,j]}, g_bar:{(Gray - 0.299*k[0] - 0.114*k[1])/0.587}, k:{k}")

            X = 0
            delta = int(Stego[i,j,2]) - int(I[i,j,2])
            X += delta**2
            MSE += delta ** 2
            delta = int(Stego[i,j,1]) - int(I[i,j,1])
            X += delta**2
            MSE += delta ** 2 
            delta = int(Stego[i,j,0]) - int(I[i,j,0])
            X += delta**2           
            MSE += delta ** 2    
            if(X > 128):
                N += 1                                    

    MSE /= (Stego.shape[0]*Stego.shape[1]*3)
    PSNR = 10 * np.log10(65025/MSE)
    print(f"PSNR:{PSNR} , F:{p} , X:{N}")

    with open("processing_data/"+Graph+".txt","w") as file:
        file.write(f"PSNR: {PSNR}\n")
        file.write(f"outliers: {p}")


    io.imshow(Stego)
    io.show()
    io.imsave('processing_image/'+Graph+'.png',Stego)

def Authorize(Graph, Name=None):
    try:
        df = pd.read_csv('RT.csv')
        RT_table = df.to_numpy()
    except:
        RT = Sudoku(256)
        df = pd.DataFrame(RT)
        df.to_csv('RT.csv', index=False, header=True)
        df = pd.read_csv('RT.csv')
        RT_table = df.to_numpy()
    path = "embeding_noise/"+Graph+".png"

    I=io.imread(path)
    Stego = I.copy()
    Flag = False
    detected_error = 0
    diff_pixels = 0
    for i in range(Stego.shape[0]):
        # if(Flag):
        #     break
        for j in range(Stego.shape[1]):
            Gray = I[i,j,0]*0.299+I[i,j,1]*0.587+I[i,j,2]*0.114
            G_round = round(Gray)
            ac = G_round
            flag = False
            if(ac > RT_table[Stego[i,j,0],Stego[i,j,2]]):
                ac = abs(int((RT_table[Stego[i,j,0],Stego[i,j,2]] - 0.299*Stego[i,j,0] - 0.114*Stego[i,j,2])/0.587))
                if(Stego[i,j,1] != ac):
                    Stego[i,j,0] = 255
                    Stego[i,j,1] = 255
                    Stego[i,j,2] = 255
                    flag = True
                    detected_error += 1
                    # print(f"This picture is tampered. i: {i} ,j: {j} ,Stego:{Stego[i,j]} ,Gray:{Gray}, RT_table:{RT_table[Stego[i,j,0],Stego[i,j,2]]}")
                    # Flag = True
                    # break 
            elif(ac < RT_table[Stego[i,j,0],Stego[i,j,2]]):
                ac = 510 - abs(int((RT_table[Stego[i,j,0],Stego[i,j,2]] - 0.299*Stego[i,j,0] - 0.114*Stego[i,j,2])/0.587))
                if(Stego[i,j,1] != ac):                
                    Stego[i,j,0] = 255
                    Stego[i,j,1] = 255
                    Stego[i,j,2] = 255
                    flag = True
                    detected_error += 1
                    # print(f"This picture is tampered. i: {i} ,j: {j} ,Stego:{Stego[i,j]} ,Gray:{Gray}, RT_table:{RT_table[Stego[i,j,0],Stego[i,j,2]]}")
                    # Flag = True
                    # break  
            if(not flag):
                Stego[i,j,0] = 0
                Stego[i,j,1] = 0
                Stego[i,j,2] = 0
                
    io.imshow(Stego, vmin=0, vmax=255)
    io.show()  
    image1 = io.imread('embeding_noise/'+Graph+'.png')
    image2 = io.imread('processing_image/'+Graph+'.png')
    
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            if(image1[i,j] != image2[i,j]).any():
               diff_pixels+=1 
    print(Graph)
    accuracy = detected_error/diff_pixels
    print(f"Detected error: {detected_error}, Actual error: {diff_pixels}, Accuracy: {accuracy}")
    if(Name != None): 
        with open("processing_data/"+Name+".txt","a") as file:
            file.write(f"PSNR: {accuracy}\n")    