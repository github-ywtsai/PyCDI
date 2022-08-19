# Application of optimization technique to noncrystalline x-ray diffraction microscopy:
# Guided hybrid input-output method
import numpy as np
import DiffErr

def errReduction(rho_p, support, alpha, measured_amplitude, patched_ROI):
    # rho_p為上一次跌代完後未加入support限制的Rspace
    # support為布林矩陣
    # alpha為定值，用在loose support外做限制
    # measured_amplitude為實驗數據，已做完dopatch
    # patched_ROI為做完dopatch後的ROI

    measured_amp = np.copy(measured_amplitude)
    measured_amp[patched_ROI==False] = 0

    #eq(5) support constraints
    inv_supp = np.invert(support)
    new_rho_outSupport = alpha*rho_p*inv_supp
    new_rho_inSupport = rho_p*support
    new_rho = new_rho_outSupport + new_rho_inSupport

    #eq(1) gain the new phase information and apply to measured amplitude
    inv_ROI = np.invert(patched_ROI)
    G = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(new_rho))) #將上次的結果fft得到新的相位猜測
    phase = np.angle(G) #取出F的相位資訊
    G_p_outROI = G*inv_ROI 
    G_p_inROI = measured_amp* np.exp(1j*phase)*patched_ROI #將ROI區的相位套上measured amplitude，其餘地方放著不動
    G_p = G_p_outROI + G_p_inROI

    #ifft(G_p) to get a new real space for next iteration
    new_rho_p = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G_p)))) #將G_p做ifft得到新的real space
    
    #eq(4) calculate diffraction error
    G_p_amp = np.absolute(G_p)
    diff_err = DiffErr.DiffErr(G_p_amp/np.max(G_p_amp),measured_amp/np.max(measured_amp)) #計算diffraction error

    return new_rho_p,diff_err