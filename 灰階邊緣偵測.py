import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io,transform
import Toollib
import numpy as np
import matplotlib.pyplot as plt
from xxhash import xxh32
import B2D
from skimage import io,transform
import random
import pandas as pd


path = 'image/Lena'
#讀取原圖(未維持灰階不變)
I=io.imread(path +r'.tiff')
Stego = I.copy()
for i in range(Stego.shape[0]):
    for j in range(Stego.shape[1]):
        for k in range(Stego.shape[2]):
            ac = Toollib.hashB(np.array([k,i,j]),4)
            Stego[i,j,k] = (Stego[i,j,k]//16)*16 + ac
not_gray_inversion_img_gray = cv2.cvtColor(Stego, cv2.COLOR_BGR2GRAY)
# edges_not_gray_inversion = cv2.Canny(image_not_gray_inversion, threshold1=50, threshold2=150)

# 讀取彩色圖(維持灰階不變)(選項一)
images = ["Lena"]
noise = ["Lena"]

for i in images:
    delta_RB,extra_bit = Toollib.AVGI(i) 
    accaurcy = 0
    for j in noise:    
        Toollib.embeding(i,j)
        accaurcy += Toollib.Authorize(i,extra_bit)

img_color = cv2.imread('processing_image/Lena.png')  # 注意這裡是讀取彩色

df = pd.read_csv('RT.csv')
RT_table = df.to_numpy()

# 假設 img 是 BGR 格式的彩色圖像 (H, W, 3)
# 且資料可能超出255，例如 float32 或 uint16
B = img_color[:, :, 0].astype(np.int8)
G = img_color[:, :, 1].astype(np.int8)
R = img_color[:, :, 2].astype(np.int8)
img_gray = 0.114 * B + 0.587 * G + 0.299 * R
img_gray = np.round(img_gray)
for i in range(G.shape[0]):
    for j in range(G.shape[1]):
        if(extra_bit[i,j] == 0):
            ac =  Toollib.hashB(np.array([img_gray[i,j],i,j]),8)
        else:
            ac =  Toollib.hashB(np.array([img_gray[i,j],i,j,32]),8)
        if(ac != RT_table[R[i,j],B[i,j]]):
            if(G[i,j] > 240):
                img_gray[i,j] = R[i,j]*0.299+(510-G[i,j])*0.587+B[i,j]*0.114
            elif(I[i,j,1] < 15):
                img_gray[i,j] = R[i,j]*0.299+(-1*G[i,j])*0.587+B[i,j]*0.114
            img_gray[i,j]  = round(img_gray[i,j])
img_gray = img_gray.astype(np.uint8)

# 讀取灰階圖
origin_img_gray = cv2.imread('grayscale_image/Lena.png', cv2.IMREAD_GRAYSCALE)

# 模糊處理
blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.4)
ori_blurred = cv2.GaussianBlur(origin_img_gray, (5, 5), 1.4)
not_inversion_blurred = cv2.GaussianBlur(not_gray_inversion_img_gray, (5, 5), 1.4)
#增強對比度
img_eq = cv2.equalizeHist(blurred)
ori_img_eq = cv2.equalizeHist(ori_blurred)
not_inversion_img_eq = cv2.equalizeHist(not_inversion_blurred)

# Canny 邊緣偵測
edges = cv2.Canny(img_eq, 70, 100)
ori_edges = cv2.Canny(ori_img_eq, 70, 100)
not_inversion_edges = cv2.Canny(not_inversion_img_eq, 70, 100)

# 顯示結果
plt.figure(figsize=(20,10))
plt.subplot(1,3,1)

plt.imshow(ori_edges, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)

plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)

plt.imshow(not_inversion_edges, cmap='gray')
plt.axis('off')

plt.savefig('edge_detected/canny_edge_comparison.png', dpi=300)
plt.show()
