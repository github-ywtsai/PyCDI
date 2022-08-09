import PyEigerData as Eiger
import numpy as np
import matplotlib.pyplot as plt
import time
import find_Center as FC

""" 

This example demonstrates how to use the 'find_Center.py' in which there are two functions (i.e., Method_CV2 and plot_Center)

Notice:
i. Some modules/packages should be installed first: numpy, numba, matplotlib, opencv-python, and scikit-image.
ii. The main function is Method_CV2 where the user should input some required parameters, and then the center (axis/pixels) will be output.
iii. After running Method_CV2, plot_Center can be sequentially run to check whether the center is found truly or not. 

"""

A = Eiger.GeneralData()
A.open('16M_879_master.h5')
BSMask = A.convCSV2Logical('16M_879_mask.csv') # define mask for beamstop
PMask = A.PixelMask #  define the mask for the defect of detectors
Mask = np.logical_or(BSMask,PMask) # combine masks
A.ROI = A.convMask2ROI(Mask) # convert effective mask to ROI
A.loadData(1) # load frame or frames
A.processData() # default: normalize


# The minimum of two required parameters must to be given: i. np_array of diffraction pattern and ii. np_array of mask.
# In the former, the pixel (axis) of data canceled by the user should be np.nan. In the latter, the type of data is Boolean.
# For a frame, both numbers of height (Y) and width (X) is depicted by the diameter. (Default: diameter = 100)
# The outputs are new_center and plot_data. 
# The type of former is 'list' [new Y, new X], and that of latter is 'dictionary'.
new_center, plot_data = FC.Method_CV2(A.ProcessedData, A.ROI,'CCORR_NORMED', diameter = 1000)

# The data inputed in plot_Center was acquired from the output of Method_CV2.
# Note that plot_Center is just a "optional" function that the user want to check whether the center is found truly or not.
#FC.plot_Center(plot_data) # optional