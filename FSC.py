#Journal of Structural Biology 151 (2005) 250–262
import numpy as np
import matplotlib.pyplot as plt
import FSC_patch
import qmap_compute as qmap_c
def FSC(reconst1,reconst2,radius,waveleng,SD_distance,FSC_sumRange_list,X_lebal):

    [R1,R2] = FSC_patch.FSC_patch(reconst1,reconst2) #先將兩個reconstruction結果對齊
    F1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(R1))) #兩者作傅立葉轉換
    F2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(R2)))
    center = [round((np.shape(F1)[0]+1)/2)-1,round((np.shape(F1)[1]+1)/2)-1]
    qmap = qmap_c.qmap_compute(F1,waveleng,SD_distance) #計算兩張F的qmap
    
    FSC12_numerator = F1*np.conj(F2) #eq(1)分子
    FSC12_denominator1 = np.abs(F1)**2 #eq(1)分母第一項
    FSC12_denominator2 = np.abs(F2)**2 #eq(1)分母第二項
    FSC12_r = np.zeros(radius)
    T_1bit = np.zeros(radius)
    T_12bit = np.zeros(radius)
    m = 0
    for r in range(radius):
        ROI = np.full(np.shape(F1), False) #第r個shell的區域
        for row in range(np.shape(F1)[0]):
            for col in range(np.shape(F1)[1]):
                if qmap[row,col]>FSC_sumRange_list[m] and qmap[row,col]<FSC_sumRange_list[m+1]:
                    ROI[row,col] = True
        nr = np.sum(ROI)
        # plt.figure()
        # plt.imshow(ROI)
        FSC12_r[r] = np.abs(np.sum(FSC12_numerator*ROI))/np.sqrt((np.sum(FSC12_denominator1*ROI))*(np.sum(FSC12_denominator2*ROI)))
        T_1bit[r] = (0.5+2.4142*1/np.sqrt(nr))/(1.5+1.4142*1/np.sqrt(nr))# eq(14)
        T_12bit[r] = (0.2071+1.9102*1/np.sqrt(nr))/(1.2071+0.9102*1/np.sqrt(nr))# eq(17)
        m = m+1
    
    plt.figure()
    plt.plot(X_lebal,FSC12_r,label="FSC")
    plt.plot(X_lebal,T_1bit,label="T_1bit",linestyle = "--")
    plt.plot(X_lebal,T_12bit,label="T_12bit",linestyle = "--")
    plt.xlabel("q")
    plt.ylabel("FSC")
    plt.legend()
    plt.show()
    return FSC12_r