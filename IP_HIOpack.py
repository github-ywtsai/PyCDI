import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import hio
import dynamic_support
import Error_Reduction
# 讀檔
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

# hio初始參數
random_Phase = np.random.uniform((-1)*np.pi,np.pi,np.shape(cut_amp)) #做一組隨機相位
G_init = cut_amp* np.exp(1j*random_Phase)
rho_p_init = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G_init)))

# autoCorrelation support
autoCorr_rho_p_init = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(cut_inensity*cut_ROI))))
autoCorr_rho_p_init = autoCorr_rho_p_init.astype(int)
autocorr_support = np.full(np.shape(autoCorr_rho_p_init), False)
autocorr_support[autoCorr_rho_p_init > 0] = True

support = np.copy(autocorr_support)
# first hio
last_rho_init = np.zeros(np.shape(rho_p_init)) #先隨便做一個初始值給第一次的support constrain用
for iter in range(20):
    new_rho_p,new_rho,diff_err = hio.HIO(rho_p_init, last_rho_init, support, 0.9, cut_amp, cut_ROI, "both")
    rho_p_init1 = np.copy(new_rho_p) #這一次hio跑完未做support constrain的
    last_rho_init1 = np.copy(new_rho) #上一次的R_hio做完support constrain的

# 流程：HIO > dynamic support > Error reduction > HIO
hio_iter = 300
dynamic_support_freq = 6
std_deviation = 0.0255
delta_std_deviation = 0.0005
std_deviation_threshold = 0.01
guassion_threshold = 19/100
dynSupp_hio_iter = 10
error_reduction_freq = 100
ErrReduc_iter = 10
ErrReduc_hio_iter = 20

for n in range(1,hio_iter+1):
    print(n)
    new_rho_p,new_rho,diff_err = hio.HIO(rho_p_init, last_rho_init, support, 0.9, cut_amp, cut_ROI, "both")
    rho_p_init = np.copy(new_rho_p) #這一次hio跑完未做support constrain的
    last_rho_init = np.copy(new_rho) #上一次的R_hio做完support constrain的

    if n % dynamic_support_freq == 0:
        std_deviation = std_deviation-delta_std_deviation
        if std_deviation < std_deviation_threshold:
            std_deviation = np.copy(std_deviation_threshold)        
        new_support = dynamic_support.dynamic_supp(new_rho_p, guassion_threshold, std_deviation)
        support  = np.copy(new_support)
        for m in range(dynSupp_hio_iter):
            new_rho_p,new_rho,diff_err = hio.HIO(rho_p_init, last_rho_init, support, 0.9, cut_amp, cut_ROI, "both")
            rho_p_init = np.copy(new_rho_p) #這一次hio跑完未做support constrain的
            last_rho_init = np.copy(new_rho) #上一次的R_hio做完support constrain的            
    
    if n % error_reduction_freq == 0:
        for k in range(ErrReduc_iter):
            reduct_rho_p,new_rho,reduct_diff_err = Error_Reduction.errReduction(rho_p_init, support, 0, cut_amp, cut_ROI, "both")
            rho_p_init = np.copy(reduct_rho_p) #這一次hio跑完未做support constrain的
            last_rho_init = np.copy(new_rho) #上一次的R_hio做完support constrain的
        for p in range(ErrReduc_hio_iter):
            new_rho_p,new_rho,diff_err = hio.HIO(rho_p_init, last_rho_init, support, 0.9, cut_amp, cut_ROI, "both")
            rho_p_init = np.copy(new_rho_p) #這一次hio跑完未做support constrain的
            last_rho_init = np.copy(new_rho) #上一次的R_hio做完support constrain的       
plt.figure()
plt.imshow(support)
plt.title("final_support"+str(std_deviation))
plt.figure()
plt.imshow(np.real(rho_p_init))
plt.title("hio result  errF:"+str(diff_err))    

# 做方形support
release_support = np.full(np.shape(cut_inensity), False)
support_center = [round((np.shape(support)[0]+1)/2),round((np.shape(support)[1]+1)/2)]
rang = 150
release_support[support_center[0]-rang:support_center[0]+rang , support_center[1]-rang:support_center[1]+rang] = True
for iter2 in range(5):
    print(iter2)
    new_rho_p,new_rho,diff_err = hio.HIO(rho_p_init, last_rho_init, release_support, 0.9, cut_amp, cut_ROI, "both")
    rho_p_init = np.copy(new_rho_p) #這一次hio跑完未做support constrain的
    last_rho_init = np.copy(new_rho) #上一次的R_hio做完support constrain的

plt.figure()
plt.imshow(np.real(rho_p_init))
plt.title("release hio result  errF:"+str(diff_err))
plt.show()