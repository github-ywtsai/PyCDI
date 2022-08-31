import numpy as np
import matplotlib.pyplot as plt
import hio
import Error_Reduction
from PIL import Image
import alternative_hio
import dynamic_support
import scipy.signal

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
intensity = amplitude**2
ROI = np.full((5*Object_size[0], 5*Object_size[1]), True)

#做初始參數
random_Phase = np.random.uniform((-1)*np.pi,np.pi,np.shape(amplitude)) #做一組隨機相位
G_init = amplitude* np.exp(1j*random_Phase)
rho_p_init = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G_init))))
plt.figure()
plt.imshow(rho_p_init)
plt.title("rho_p_init")

#做auto correlation support
# adj_inten = np.copy(intensity)
# adj_inten[intensity<np.max(intensity)*4/100] = 0
autoCorr_rho_p_init = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(intensity))))
autoCorr_rho_p_init = autoCorr_rho_p_init.astype(int)
autocorr_support = np.full(np.shape(autoCorr_rho_p_init), False)
autocorr_support[autoCorr_rho_p_init > 0] = True

plt.figure()
plt.imshow(autocorr_support)
plt.title("autocorr_support")

last_rho_init = np.zeros(np.shape(autocorr_support)) #先隨便做一個初始值給第一次的support constrain用
for iter in range(20):
    new_rho_p,new_rho,diff_err = alternative_hio.alter_HIO(rho_p_init, last_rho_init, autocorr_support, 0.9, amplitude, ROI)
    rho_p_init = np.copy(new_rho_p) #這一次hio跑完未做support constrain的
    last_rho_init = np.copy(new_rho) #上一次的R_hio做完support constrain的
    print(iter)
plt.figure()
plt.imshow(new_rho_p)
plt.title("new_rho_p")

std_deviation = 0.03
for iter2 in range(20):
    print(iter2)
    new_support = dynamic_support.dynamic_supp(new_rho_p, 20/100, std_deviation)
    plt.figure()
    plt.imshow(new_support)
    for iter in range(20):
        new_rho_p,new_rho,diff_err = alternative_hio.alter_HIO(rho_p_init, last_rho_init, new_support, 0.9, amplitude, ROI)
        rho_p_init = np.copy(new_rho_p) #這一次hio跑完未做support constrain的
        last_rho_init = np.copy(new_rho) #上一次的R_hio做完support constrain的

    std_deviation = std_deviation-0.0005

print(diff_err)
plt.figure()
plt.imshow(new_rho_p)
plt.title("new_rho_p")
plt.figure()
plt.imshow(new_support)
plt.title("new_support")
plt.ioff()
plt.show()