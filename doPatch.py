import numpy as np

def doPatch(rawdata, ROI, CenterofMass):
    # rawdata為要處理的那張pattern
    # ROI為布林矩陣，region of interest，true的地方是要留下的
    # CenterofMass為對稱中心座標，輸入為[y(row),x(col)]
    
    mask_pattern = np.copy(rawdata)
    mask_pattern[ROI==False] = np.nan #不要的地方設為NAN

    y_sym = round(CenterofMass[0])
    x_sym = round(CenterofMass[1]) #弄成整數對稱座標
    size = np.shape(rawdata)
    y1 = y_sym-1
    y2 = size[0]-y_sym
    x1 = x_sym-1
    x2 = size[1]-x_sym
    range = [min(y1,y2),min(x1,x2)] #由於pattern不是方的而且對稱點可能不在中心，所以要找一個可以做對稱的範圍半徑

    cut_data = mask_pattern[int(y_sym-range[0]-1):int(y_sym+range[0]), int(x_sym-range[1]-1):int(x_sym+range[1])] #截出可以做對稱的部分

    #開始做對稱
    P1 = np.rot90(cut_data, 2) #翻轉180度
    P2 = np.copy(cut_data)
    P2[np.isnan(P2)] = P1[np.isnan(P2)] #0度做補點一次
    P1[np.isnan(P1)] = cut_data[np.isnan(P1)] #180度做補點一次

    sym_cut_data = (P1+P2)/2 #取平均強迫對稱點都要是他們的平均值

    sym_pattern = np.copy(mask_pattern)
    sym_pattern[int(y_sym-range[0]-1):int(y_sym+range[0]), int(x_sym-range[1]-1):int(x_sym+range[1])] = sym_cut_data #把對稱完成的部分補回去data中

    return sym_pattern #匯出rawdata對稱後的結果