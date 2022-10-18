# PHYSICAL REVIEW B 76, 064113 (2007)
import numpy as np
import DiffErr

def errReduction(rho_p_nm1, support, alpha, measured_amplitude, ROI, option):
    # rho_p為上一次跌代完後未加入support限制的Rspace
    # support為TrueFalse矩陣
    # alpha為定值，用在loose support外做限制
    # measured_amplitude為實驗數據，已做完dopatch
    # ROI為做完dopatch後的ROI，為TrueFalse矩陣

    measured_amp = np.copy(measured_amplitude)
    measured_amp[ROI==False] = 0

    #eq(5) support constraints
    inv_supp = np.invert(support)
    rho_n_outSupport = alpha*rho_p_nm1*inv_supp
    rho_n_inSupport = rho_p_nm1*support
    rho_n = rho_n_outSupport + rho_n_inSupport

    #eq(5) real space constraints (real>=0)
    if option == "real":
        negtive_real_space = np.real(rho_n_inSupport)<0
        non_negtive_real_space = np.invert(negtive_real_space)
        rho_n_negtive_real_space = alpha*rho_p_nm1*negtive_real_space
        rho_n_non_negtive_real_space = rho_n_inSupport*non_negtive_real_space
        rho_n_inSupport_p = rho_n_negtive_real_space + rho_n_non_negtive_real_space
        rho_n = rho_n_outSupport + rho_n_inSupport_p

    #eq(5) real space constraints (image>=0)
    if option == "image":
        negtive_image_space = np.imag(rho_n_inSupport)<0
        non_negtive_image_space = np.invert(negtive_image_space)
        rho_n_negtive_image_space = alpha*rho_p_nm1*negtive_image_space
        rho_n_non_negtive_image_space = rho_n_inSupport*non_negtive_image_space
        rho_n_inSupport_p = rho_n_negtive_image_space + rho_n_non_negtive_image_space
        rho_n = rho_n_outSupport + rho_n_inSupport_p

    #eq(5) real space constraints (image>=0 && real>=0)
    if option == "both":
        negtive_real_space = np.real(rho_n_inSupport)<0
        negtive_image_space = np.imag(rho_n_inSupport)<0
        negtive_space = negtive_real_space + negtive_image_space
        negtive_space = negtive_space>0
        non_negtive_space = np.invert(negtive_space)
        rho_n_negtive_space = alpha*rho_p_nm1*negtive_space
        rho_n_non_negtive_space = rho_n_inSupport*non_negtive_space
        rho_n_inSupport_p = rho_n_negtive_space + rho_n_non_negtive_space
        rho_n = rho_n_outSupport + rho_n_inSupport_p

    
    #eq(1) gain the new phase information and apply to measured amplitude
    inv_ROI = np.invert(ROI)
    G_n = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(rho_n))) #將上次的結果fft得到新的相位猜測
    phase = np.angle(G_n) #取出F的相位資訊
    G_p_n_outROI = G_n*inv_ROI 
    G_p_n_inROI = measured_amp* np.exp(1j*phase)*ROI #將ROI區的相位套上measured amplitude，其餘地方放著不動
    G_p_n = G_p_n_outROI + G_p_n_inROI

    #ifft(G_p) to get a new real space for next iteration
    rho_p_n = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G_p_n))) #將G_p做ifft得到新的real space
    
    #eq(4) calculate diffraction error
    G_p_n_amp = np.absolute(G_p_n)

    diff_err = DiffErr.DiffErr(G_p_n_amp*ROI,measured_amp*ROI) #計算diffraction error

    return rho_p_n,rho_n,diff_err