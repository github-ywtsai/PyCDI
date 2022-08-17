import numpy as np
import DiffErr

def alter_HIO(last_Rspace, R, support, beta1, beta2, measured_amplitude, patched_ROI):
    # last_Rspace為上一次跌代完後還未加入support限制的Rspace
    # R為上一次做完support constraints的結果，也就是上上次的跌代結果做完support限制
    # support為布林矩陣
    # beta1為定值，用在loose support外
    # beta2為定值，用在loose support內
    # measured_amplitude為實驗數據，已做完dopatch
    # patched_ROI為做完dopatch後的ROI
    # last_Rspace = 上一次未做調整的Rspace；R = 上一次做完調整的Rspace，也就是上一次的R_
    measured_A = np.copy(measured_amplitude)
    measured_A[patched_ROI==False] = 0

    R_ = np.zeros(np.shape(last_Rspace))
    R_[support==False] = R[support==False]-beta1*last_Rspace[support==False] #將loose support外限制加入
    support_region = last_Rspace[support]
    last_support_region = last_Rspace[support]
    support_region[support_region<0] = last_support_region[support_region<0] - beta2*support_region[support_region<0]#loose support內不能有電子密度<0
    R_[support] = support_region 

    F_ = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(R_))) #將上次的結果fft得到新的相位猜測
    phase = np.angle(F_) #取出F的相位資訊
    F_[patched_ROI] = np.multiply(np.exp(1j*phase[patched_ROI]),measured_A[patched_ROI]) #將ROI區的相位套上measured amplitude，其餘地方放著不動
    R_new = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F_)))) #將F1做ifft得到新的real space

    F_new = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(R_new)))
    A_new = np.absolute(F_new)
    diff_err = DiffErr.DiffErr(A_new,measured_A) #計算diffraction error

    return R_new,R_,diff_err
    