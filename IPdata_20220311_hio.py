import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import alternative_hio
import dynamic_support

filepath = 'D:\IP_20220311_25a\IP_20220311\patched_rawdata_20220311.mat'
patch_data = io.loadmat(filepath)
measured_intensity = (patch_data['sym_rawdata'])
measured_amp = np.sqrt(measured_intensity)
patched_ROI = patch_data['patched_rawdataROI']==1 #TrueFalse存成mat之後會變成10，這裡再次做成true false形式

center = patch_data['sym_center'][0]
cut_inensity = measured_intensity[round(center[0])-600:round(center[0])+600,round(center[1])-600:round(center[1])+600]
cut_inensity[np.isnan(cut_inensity)] = 0
cut_amp = measured_amp[round(center[0])-600:round(center[0])+600,round(center[1])-600:round(center[1])+600]
cut_amp[np.isnan(cut_amp)] = 0
cut_ROI = patched_ROI[round(center[0])-600:round(center[0])+600,round(center[1])-600:round(center[1])+600]

# 做hio初始參數
random_Phase = np.random.uniform((-1)*np.pi,np.pi,np.shape(cut_amp)) #做一組隨機相位
G_init = cut_amp* np.exp(1j*random_Phase)* cut_ROI
rho_p_init = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G_init))))
plt.figure()
plt.imshow(rho_p_init)
plt.title("rho_p_init")

#做auto correlation support
autoCorr_rho_p_init = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(cut_inensity*cut_ROI))))
autoCorr_rho_p_init = autoCorr_rho_p_init.astype(int)
autocorr_support = np.full(np.shape(autoCorr_rho_p_init), False)
autocorr_support[autoCorr_rho_p_init > 0] = True
plt.figure()
plt.imshow(autocorr_support)
plt.title("autocorr_support")

# first hio
last_rho_init = np.zeros(np.shape(rho_p_init)) #先隨便做一個初始值給第一次的support constrain用
for iter in range(20):
    new_rho_p,new_rho,diff_err = alternative_hio.alter_HIO(rho_p_init, last_rho_init, autocorr_support, 0.9, cut_amp, cut_ROI)
    rho_p_init = np.copy(new_rho_p) #這一次hio跑完未做support constrain的
    last_rho_init = np.copy(new_rho) #上一次的R_hio做完support constrain的
    print(iter)
plt.figure()
plt.imshow(new_rho_p)
plt.title("first_new_rho_p")

std_deviation = 0.03
for iter2 in range(20):
    print(iter2)
    new_support = dynamic_support.dynamic_supp(new_rho_p, 20/100, std_deviation)
    plt.imshow(new_support)
    for iter in range(20):
        new_rho_p,new_rho,diff_err = alternative_hio.alter_HIO(rho_p_init, last_rho_init, new_support, 0.9, cut_amp, cut_ROI)
        rho_p_init = np.copy(new_rho_p) #這一次hio跑完未做support constrain的
        last_rho_init = np.copy(new_rho) #上一次的R_hio做完support constrain的

    std_deviation = std_deviation-0.0005

# io.savemat('hio_result_dev0p01_const_threshold0p18_20hio20dynSupp_600x600', {'hio_result':new_rho_p,'dynamic_Supp':new_support,'Diff_err':diff_err})

print(diff_err)
plt.figure()
plt.imshow(new_rho_p)
plt.title("new_rho_p")
plt.figure()
plt.imshow(new_support)
plt.title("new_support")
plt.ioff()
plt.show()