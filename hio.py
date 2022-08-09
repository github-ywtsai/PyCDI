import numpy as np
import DiffErr

def HIO(last_Rspace, support, beta1, beta2, measured_amplitude, patched_ROI):
    # last_Rspace為上一次跌代完後還未加入support限制的Rspace
    # support為布林矩陣
    # beta1為定值，用在loose support外
    # beta2為定值，用在loose support內
    # measured_amplitude為實驗數據，已做完dopatch
    # patched_ROI為做完dopatch後的ROI

    measured_A = np.copy(measured_amplitude)
    measured_A[patched_ROI==False] = 0

    last_Rspace[support==False] = last_Rspace[support==False]-beta1*last_Rspace[support==False] #將loose support外限制加入
    support_region = last_Rspace[support]
    support_region[support_region<0] = support_region[support_region<0] - beta2*support_region[support_region<0]
    last_Rspace[support] = support_region #loose support內不能有電子密度<0

    F = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(last_Rspace))) #將上次的結果fft得到新的相位猜測
    A = np.absolute(F)
    phase = np.angle(F) #取出F的相位資訊
    F[patched_ROI] = np.multiply(np.exp(1j*phase[patched_ROI]),measured_A[patched_ROI]) #將ROI區的相位套上measured amplitude，其餘地方放著不動
    R = np.real(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(F)))) #將F1做ifft得到新的real space

    diff_err = DiffErr.DiffErr(A,measured_A) #計算diffraction error

    return R,diff_err
    