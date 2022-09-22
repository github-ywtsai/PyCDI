import numpy as np  
from scipy import io 
import FSC_patch
import matplotlib.pyplot as plt
#拿一張當標準，其他人都對齊它
filepath = 'D:\IPset1\IPdataSet1_hio_result 6.mat'
reconstruction6 = io.loadmat(filepath)
temp = reconstruction6['hio_result']

filepath = 'D:\IPset1\IPdataSet1_hio_result 15.mat'
reconstruction9 = io.loadmat(filepath)
R = reconstruction9['hio_result']

A,B = FSC_patch.FSC_patch(temp,R)
print(A)
print(B)
# plt.figure()
# plt.imshow(np.abs(A))
# plt.title('A')
# plt.figure()
# plt.imshow(np.abs(B))
# plt.title('B')
plt.figure()
plt.imshow(np.abs(temp)+np.abs(R))
plt.title('temp+R')
plt.figure()
plt.imshow(np.abs(A)+np.abs(B))
plt.title('A+B')
plt.show()