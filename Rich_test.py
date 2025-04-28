import Toollib
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# images = [os.path.splitext(path)[0] for path in os.listdir('image')]
images = ["Tiffany"]
noise = ["Tiffany"]

# for i in images:
#     delta_RB,extra_bit = Toollib.AVGI(i) 
#     accaurcy = 0
#     for j in noise:    
#         Toollib.embeding(i,j)
#         accaurcy += Toollib.Authorize(i,extra_bit)
#     accaurcy /= len(noise)
#     print(f"Picture {i}: accaurcy={accaurcy}")