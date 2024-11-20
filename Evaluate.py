import numpy as np
def evaluate(I, S):
    Shape = I.shape
    MSE = 0
    for i in range(Shape[0]):
        for j in range(Shape[1]):
            for k in range(Shape[2]):
                delta = int(I[i,j,k]) - int(S[i,j,k])
                MSE += delta ** 2
    
    MSE /= (Shape[0] * Shape[1] * 3)
    
    PSNR = 10 * np.log10(65025/MSE)
    return PSNR

I = np.array([[[1,1,1]],[[1,1,1]]])
S = np.array([[[1,1,0]],[[1,1000,1]]])