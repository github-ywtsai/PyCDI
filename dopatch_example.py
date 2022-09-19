import numpy as np
import matplotlib.pyplot as plt
import doPatch as dp
rawdata = np.ones([9,9])
rawdata[0,3] = 0
ROI = np.full(np.shape(rawdata), True)
ROI[0,3] = False
CenterofMass = [4,4]
A,B = dp.doPatch(rawdata, ROI, CenterofMass)