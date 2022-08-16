import numpy as np
import matplotlib.pyplot as plt
import hio
import Error_Reduction
from PIL import Image
import alternative_hio
import dynamic_support
import scipy.signal

img = Image.open('IR.jpg')
Object = img.convert('L')
# plt.figure()
# plt.imshow(Object)
# plt.title("Object")

Object_size = np.shape(Object)
support = np.full((5*Object_size[0], 5*Object_size[1]), False)
support_size = np.shape(support)
s0 = round(support_size[0]/2)-round(Object_size[0]/2)
s1 = round(support_size[1]/2)-round(Object_size[1]/2)
# support[s0:s0+Object_size[0], s1:s1+Object_size[1]] = True

R = np.zeros(np.shape(support))
R[s0:s0+Object_size[0], s1:s1+Object_size[1]] = Object
plt.figure()
plt.imshow(R)
plt.title("Object")
amplitude = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(R)))) #此代表量測到的amplitude
intensity = amplitude**2
ROI = np.full((5*Object_size[0], 5*Object_size[1]), True)
random_Phase = np.random.uniform((-1)*np.pi,np.pi,np.shape(amplitude)) #做一組隨機相位
F_int = np.multiply(amplitude, np.exp(1j*random_Phase))
R_int = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F_int))))
plt.figure()
plt.imshow(R_int)
plt.title("R_int")

adj_inten = np.copy(intensity)
adj_inten[intensity<intensity*4/100] = 0
autoCorr_R_int = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(adj_inten))))
autoCorr_R_int = autoCorr_R_int.astype(int)
autocorr_support = np.full(np.shape(autoCorr_R_int), False)
autocorr_support[autoCorr_R_int > 0] = True
# autocorr_support[autoCorr_R_int >= np.max(autoCorr_R_int)*4/100] = True #得到一個初始support

plt.figure()
plt.imshow(autocorr_support)
plt.title("autocorr_support")

R_const = np.zeros(np.shape(autocorr_support)) #先隨便做一個初始值給第一次的support constrain用
for iter in range(20):
    R_hio,R_,Err_hio = alternative_hio.alter_HIO(R_int, R_const, autocorr_support, 0.9, 0.9, amplitude, ROI)
    R_int = np.copy(R_hio) #這一次hio跑完未做support constrain的
    R_const = np.copy(R_) #上一次的R_hio做完support constrain的
    print(iter)
plt.figure()
plt.imshow(R_hio)
plt.title("R_hio")

std_deviation = 0.02
for iter2 in range(20):
    print(iter2)
    new_support = dynamic_support.dynamic_supp(R_hio, autocorr_support, 20/100, std_deviation)
    plt.figure()
    plt.imshow(new_support)
    for iter in range(20):
        R_hio,R_,Err_hio = alternative_hio.alter_HIO(R_int, R_const, new_support, 0.9, 0, amplitude, ROI)
        R_int = np.copy(R_hio) #這一次hio跑完未做support constrain的
        R_const = np.copy(R_) #上一次的R_hio做完support constrain的

    std_deviation = std_deviation-0.0005

print(Err_hio)
plt.figure()
plt.imshow(R_hio)
plt.title("R_hio")
plt.figure()
plt.imshow(new_support)
plt.title("new_support")
plt.ioff()
plt.show()