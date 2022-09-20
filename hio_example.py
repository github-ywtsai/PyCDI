import numpy as np
import matplotlib.pyplot as plt
import Error_Reduction
from PIL import Image
import hio
import time

#做一個IR sample
img = Image.open('IR.jpg')
Object = img.convert('L')

Object_size = np.shape(Object)
support = np.full((5*Object_size[0], 5*Object_size[1]), False)
support_size = np.shape(support)
s0 = round(support_size[0]/2)-round(Object_size[0]/2)
s1 = round(support_size[1]/2)-round(Object_size[1]/2)
support[s0:s0+Object_size[0], s1:s1+Object_size[1]] = True

R = np.zeros(np.shape(support))
R[s0:s0+Object_size[0], s1:s1+Object_size[1]] = Object
plt.figure()
plt.imshow(R)
plt.title("Object")

amplitude = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(R))))
ROI = np.full((5*Object_size[0], 5*Object_size[1]), True)

#做初始參數
random_Phase = np.random.uniform((-1)*np.pi,np.pi,np.shape(amplitude)) #做一組隨機相位
G_init = amplitude* np.exp(1j*random_Phase)
rho_p_init = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G_init)))
plt.figure()
plt.imshow(np.real(rho_p_init))
plt.title("rho_p_init")

#=========================== 做alternative_hio ================================
last_rho_init = np.zeros(np.shape(support)) #先隨便做一個初始值給第一次的support constrain用

st = time.time()
for iter in range(5):
    new_rho_p,new_rho,diff_err = hio.HIO(rho_p_init, last_rho_init, support, 0.9, amplitude, ROI, 'both')
    rho_p_init = np.copy(new_rho_p) #這一次hio跑完未做support constrain的
    last_rho_init = np.copy(new_rho) #上一次的R_hio做完support constrain的
    print(iter)
ed = time.time()

print(diff_err)
print(ed-st)
plt.figure()
plt.imshow(np.real(new_rho_p))
plt.title("new_rho_p  errF:"+str(diff_err))
plt.show()