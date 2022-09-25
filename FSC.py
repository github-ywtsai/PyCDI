#Journal of Structural Biology 151 (2005) 250–262
import numpy as np
import matplotlib.pyplot as plt
import FSC_patch
import pmap
def FSC(reconst1,reconst2,R_pmap,FSC_sumRange_list):
#reconst1,reconst2分別為兩組dataset解出來的結果
#R_pmap為切割要做FSC大小的pmap
#FSC_sumRange_list判斷shell厚度的邊界list
#X_lebal自行定義每個shell的FSC所對應的座標
    [R1,R2] = FSC_patch.FSC_patch(reconst1,reconst2) #先將兩個reconstruction結果對齊
    F1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(R1))) #兩者作傅立葉轉換
    F2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(R2)))
    center = [round((np.shape(F1)[0]+1)/2)-1,round((np.shape(F1)[1]+1)/2)-1]
    radius = round(((np.shape(F1))[0]-1)/2) #shell的總半徑
    
    FSC12_numerator = F1*np.conj(F2) #eq(1)分子
    FSC12_denominator1 = np.abs(F1)**2 #eq(1)分母第一項
    FSC12_denominator2 = np.abs(F2)**2 #eq(1)分母第二項
    FSC12_r = np.zeros(radius+1)#每個半徑shell對應的FSC，+1是因為包含中心點
    T_1bit = np.zeros(radius+1)
    T_12bit = np.zeros(radius+1)
    for r in range(radius+1):
        ROI = np.full(np.shape(F1), False) 
        for row in range(np.shape(F1)[0]):
            for col in range(np.shape(F1)[1]):
                if R_pmap[row,col]>FSC_sumRange_list[r,0] and R_pmap[row,col]<=FSC_sumRange_list[r,1]:
                    ROI[row,col] = True#第r個shell的區域                
        nr = np.sum(ROI)
        # plt.figure()
        # plt.imshow(ROI)
        FSC12_r[r] = np.abs(np.sum(FSC12_numerator*ROI))/np.sqrt((np.sum(FSC12_denominator1*ROI))*(np.sum(FSC12_denominator2*ROI)))
        T_1bit[r] = (0.5+2.4142*1/np.sqrt(nr))/(1.5+1.4142*1/np.sqrt(nr))# eq(14)
        T_12bit[r] = (0.2071+1.9102*1/np.sqrt(nr))/(1.2071+0.9102*1/np.sqrt(nr))# eq(17)
    
    return FSC12_r,T_1bit,T_12bit