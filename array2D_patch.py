import numpy as np
import scipy.signal
import centerOfMass as CM

def FSC_patch(R1,R2):   
    Rcenter = [round((np.shape(R1)[0]+1)/2)-1,round((np.shape(R1)[1]+1)/2)-1]
    R1_CM = CM.CM(R1)
    R1 = np.roll(R1,Rcenter[0]-R1_CM[0],axis=0)
    R1 = np.roll(R1,Rcenter[1]-R1_CM[1],axis=1) #將R1質心移至array中心
    corr1 = scipy.signal.correlate(R1,R2)
    corr2 = scipy.signal.correlate(R1,np.flip(R2)) #計算兩者的互相關
    if np.max(corr2)>np.max(corr1): #檢查是否有flip
        R2 = np.flip(R2)
        corr = corr2
    else:
        corr = corr1
    corrCenter = [round((np.shape(corr)[0]+1)/2)-1,round((np.shape(corr)[1]+1)/2)-1]
    corrMax = np.where(corr==np.max(corr))
    R2 = np.roll(R2,corrMax[0][0]-corrCenter[0],axis=0)
    R2 = np.roll(R2,corrMax[1][0]-corrCenter[1],axis=1) #將R2移至與R1最重合的位置

    return R1,R2