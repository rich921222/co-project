import B2D
import bitsetting
import numpy as np
import random
from skimage import io,transform
from matplotlib import pyplot as plt
from xxhash import xxh32
import Toollib

def RB_histogram_Variation_Frequency(delta_RB):
    delta_RB_ravel = delta_RB.ravel()
    delta_RB_ravel = np.abs(delta_RB_ravel)
    bins = range(0, 16)
    counts, edges = np.histogram(delta_RB_ravel, bins=bins)
    print(counts)

    # 繪製長條圖
    plt.bar(edges[:-1], counts, width=1, edgecolor="black", align="edge")

    # 設置橫軸範圍和標籤
    plt.xticks(range(0, 16))
    plt.xlabel("Variation")
    plt.ylabel("Frequency")
    plt.title("APPM")

    # 顯示圖表
    plt.show()

if __name__ == "__main__":
    delta_RB = Toollib.AVGI('Tiffany')
    delta_RB = np.array(delta_RB)

    plt.figure(figsize=(6, 6))
    # 繪製所有點
    for x, y in delta_RB:
        plt.scatter(x, y, color='blue')  # 使用 scatter 繪製點

    # 設置 X 和 Y 軸範圍
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    # 繪製 (0, 0) 的十字中心
    plt.axhline(0, color='black', linewidth=0.8)  # 水平線
    plt.axvline(0, color='black', linewidth=0.8)  # 垂直線

    # 添加標籤和標題
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("XY Coordinate Plot with Points")

    # 顯示圖表
    plt.grid(color='gray', linestyle='--', linewidth=0.5)  # 顯示網格線
    plt.show()

    Toollib.embeding('Tiffany', 'firefly')
    Toollib.Authorize('Tiffany')
    print(len(delta_RB))