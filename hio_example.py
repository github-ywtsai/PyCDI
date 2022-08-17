# import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import hio
import Error_Reduction
from PIL import Image
import alternative_hio
img = Image.open('IR.jpg')
Object = img.convert('L')

# Object = scipy.misc.ascent()
# plt.figure()
# plt.imshow(Object)
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
# ROI[2,:] = False

random_Phase = np.random.uniform((-1)*np.pi,np.pi,np.shape(amplitude)) #做一組隨機相位
F_int = np.multiply(amplitude, np.exp(1j*random_Phase))
R_int = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F_int))))
plt.figure()
plt.imshow(R_int)
plt.title("R_int")

#=========================== 做alternative_hio ================================
R_const = np.zeros(np.shape(support)) #先隨便做一個初始值給第一次的support constrain用
for iter in range(200):
    R_hio,R_,Err_hio = alternative_hio.alter_HIO(R_int, R_const, support, 0.9, 0.9, amplitude, ROI)
    R_int = np.copy(R_hio) #這一次hio跑完未做support constrain的
    R_const = np.copy(R_) #上一次的R_hio做完support constrain的
    print(iter)

print(Err_hio)
plt.figure()
plt.imshow(R_hio)
plt.title("R_hio")
# plt.ioff()
# plt.show()
#===============================================================================

#===========================加上Error reduction =================================
# for iter2 in range(10):
#     print(iter2)
#     R_int,_ = Error_Reduction.errReduction(R_hio, support, 0, amplitude, ROI)
#     for iter in range(20):
#         R_hio,R_,Err_hio = alternative_hio.alter_HIO(R_int, R_const, support, 0.9, 0, amplitude, ROI)
#         R_int = np.copy(R_hio) #這一次hio跑完未做support constrain的
#         R_const = np.copy(R_) #上一次的R_hio做完support constrain的

# print(Err_hio)
# plt.figure()
# plt.imshow(R_hio)
plt.ioff()
plt.show()