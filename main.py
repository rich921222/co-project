import B2D
import bitsetting
import numpy as np
import random
from skimage import io,transform
from matplotlib import pyplot as plt
from xxhash import xxh32
import Toollib



if __name__ == "__main__":
    Toollib.AVGI('Lena')
    Toollib.embeding('Lena', 'firefly')
    Toollib.Authorize('Lena')