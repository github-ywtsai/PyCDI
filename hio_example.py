# import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import hio
import Error_Reduction
from PIL import Image
img = Image.open('IR.jpg')
Object = img.convert('L')

# Object = scipy.misc.ascent()
plt.figure()
plt.imshow(Object)
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

amplitude = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(R))))
ROI = np.full((5*Object_size[0], 5*Object_size[1]), True)
# ROI[2,:] = False

random_Phase = np.random.uniform((-1)*np.pi,np.pi,np.shape(amplitude)) #做一組隨機相位
F_int = np.multiply(amplitude, np.exp(1j*random_Phase))
R_int = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F_int))))
plt.figure()
plt.imshow(R_int[0:Object_size[0],0:Object_size[1]])

for iter in range(200):
    R_hio,Err_hio = hio.HIO(R_int, support, 0.9, 0.9, amplitude, ROI)
    R_int = np.copy(R_hio)
    print(iter)
print(Err_hio)
plt.figure()
# plt.imshow(R_hio[s0:s0+Object_size[0], s1:s1+Object_size[1]])
plt.imshow(R_hio)
plt.ioff()
plt.show()

# # # R_int = np.copy(R_hio)
# n = 100
# # alpha = np.arange(1,-1/n,-1/n)
# alpha = 0
# for iter in range(n):
#     R_ERhio,Err_ER = ER_hio.ER_HIO(R_int, support, alpha, amplitude, ROI)
#     R_int = np.copy(R_ERhio)
#     print(iter)
# print(Err_ER)
# plt.figure()
# plt.imshow(R_ERhio[0:Object_size[0],0:Object_size[1]])
# plt.ioff()
# plt.show()

# plt.imshow(Object)
# plt.show()