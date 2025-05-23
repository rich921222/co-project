import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io,transform
import Toollib


path = 'image/Lena'
#讀取原圖(未維持灰階不變)
I=io.imread(path +r'.tiff')
Stego = I.copy()
for i in range(Stego.shape[0]):
    for j in range(Stego.shape[1]):
        for k in range(Stego.shape[2]):
            ac = Toollib.hashB(np.array([k,i,j]),4)
            Stego[i,j,k] = (Stego[i,j,k]//16)*16 + ac
img_gray = cv2.cvtColor(Stego, cv2.COLOR_BGR2GRAY)
# edges_not_gray_inversion = cv2.Canny(image_not_gray_inversion, threshold1=50, threshold2=150)

# # 讀取彩色圖(維持灰階不變)(選項一)
# img_color = cv2.imread('processing_image/Baboon.png')  # 注意這裡是讀取彩色
# img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# 讀取灰階圖
# img_gray = cv2.imread('grayscale_image/Lena.png', cv2.IMREAD_GRAYSCALE)

# 模糊處理
blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.4)

#增強對比度
img_eq = cv2.equalizeHist(blurred)

# Canny 邊緣偵測
edges = cv2.Canny(img_eq, 70, 100)

# 顯示結果
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("原始灰階圖")
plt.imshow(img_gray, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Canny 邊緣偵測")
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()