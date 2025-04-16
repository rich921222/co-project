import Toollib
import numpy as np

delta_RB,extra_bit = Toollib.AVGI('Peppers')
delta_RB = np.array(delta_RB)
Toollib.RB_histogram_Variation_Frequency(delta_RB,'Peppers')    
Toollib.embeding('Peppers','tomato')
Toollib.Authorize('Peppers',extra_bit)