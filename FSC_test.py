import numpy as np
import matplotlib.pyplot as plt
import FSC
import pmap
wave_len = 1.4028047323226928e-10
SD_distance = 6.09245
R1 = np.load('dataset1_hioresult_72.npy')
R2 = np.load('dataset2_hioresult_31.npy')

radius = 150 #要切來做FSC的半徑大小
R1 = R1[150-radius:150+radius+1,150-radius:150+radius+1] #已知R1R2大小為301*301，中心點在[150,150]
R2 = R2[150-radius:150+radius+1,150-radius:150+radius+1]

FSC_sumRange_list = np.zeros([radius+1,2]) #做每層shell的q值邊界
R_pmap,delta_p = pmap.pmap(wave_len,SD_distance,1201,radius*2+1) #兩張圖的pmap
for n in range(radius+1):
    FSC_sumRange_list[n,:] = np.array([(n-1)*delta_p,n*delta_p])

X_lebal = np.zeros(radius+1)
for k in range(radius+1):
    X_lebal[k] = FSC_sumRange_list[k,1]

fsc,T_1bit,T_12bit = FSC.FSC(R1,R2,R_pmap,FSC_sumRange_list)

plt.figure()
plt.plot(X_lebal,fsc,label="FSC")
plt.plot(X_lebal,T_1bit,label="T_1bit",linestyle = "--")
plt.plot(X_lebal,T_12bit,label="T_12bit",linestyle = "--")
plt.xlabel("1/d")
plt.ylabel("FSC")
plt.legend()
plt.show()


#################################################################################################################
# import numpy as np
# import matplotlib.pyplot as plt
# import FSC
# import DataImport as DI
# from scipy import io
# import qmap_compute as qmap_c
# A = DI.EigerBasic()
# data_name = '16M_220_master.h5'
# A.open(data_name)
# wave_len = A.Header['Wavelength'] #1.4028047323226928e-10 m
# SD_distance = A.Header['DetectorDistance'] #6.09245 m