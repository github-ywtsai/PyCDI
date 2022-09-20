import numpy as np
import matplotlib.pyplot as plt
import FSC
import DataImport as DI
from scipy import io
import qmap_compute as qmap_c
import FSC_improve
A = DI.EigerBasic()
data_name = '16M_220_master.h5'
A.open(data_name)
wave_len = A.Header['Wavelength']
SD_distance = A.Header['DetectorDistance']
filepath = 'C:\\Users\\Administrator\\Desktop\\IPset1\\IPdataSet1_hio_result 7.mat'
reconst1 = io.loadmat(filepath)
R1 = np.real(reconst1['hio_result'])
filepath = 'C:\\Users\\Administrator\\Desktop\\IPset2\\IPdataSet2_hio_result 1.mat'
reconst2 = io.loadmat(filepath)
R2 = np.real(reconst2['hio_result'])
radius = 150
FSC_sumRange_list = np.zeros(radius+2)
qmap = qmap_c.qmap_compute(R1,wave_len,SD_distance)
two_delta_theta = np.arctan(75e-6/SD_distance)
delta_q = 4*np.pi*np.sin(two_delta_theta/2)/wave_len
m = 0
for n in range(0,radius+2,2):
    FSC_sumRange_list[m] = (n-0.5)*delta_q
    FSC_sumRange_list[m+1] = (n+0.5)*delta_q
    m = m+2
X_lebal = np.zeros(radius)
m = 0
for k in range(radius):
    X_lebal[k] = (FSC_sumRange_list[m]+FSC_sumRange_list[m+1])/2
    m = m+1

FSC.FSC(R1,R2,radius,wave_len,SD_distance,FSC_sumRange_list,X_lebal)