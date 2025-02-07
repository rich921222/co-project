import B2D
import bitsetting
import numpy as np
import random
from skimage import io,transform
from matplotlib import pyplot as plt
from xxhash import xxh32
import Toollib
import pandas as pd

def RB_histogram_Variation_Frequency(delta_RB,image):
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
    plt.savefig(f'Variation-Frequency/{image}_appm.png')

if __name__ == "__main__":
    delta_RB = Toollib.AVGI('Jet')
    # delta_RB = np.array(delta_RB)
    # RB_histogram_Variation_Frequency(delta_RB,'Lena')

    Toollib.embeding('Jet', 'Jet')
    Toollib.Authorize('Jet')
    #print(len(delta_RB))


    # try:
    #     df = pd.read_csv('RT.csv')
    #     RT_table = df.to_numpy()
    # except:
    #     RT = Toollib.APPM_RT256()
    #     df = pd.DataFrame(RT)
    #     df.to_csv('RT.csv', index=False, header=True)
    #     df = pd.read_csv('RT.csv')
    #     RT_table = df.to_numpy()   