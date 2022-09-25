import numpy as np
def pmap(wave_len,SD_distance,N1,N2):
    # resolution/pixel resolution計算
    # N1 #處理好後要解的data大小
    n=(N1-1)/2
    p=(75e-6) #pixel大小
    D=SD_distance #distance m
    L=wave_len #L=lamda
    theta=np.arctan(n*p/D)
    k=1/L
    Deltak=k*np.sin(theta)
    dx=1/(2*Deltak); # pixel resolution
    pixel_size = dx

    #delta_p[倒空間pixel resolution] = 1/(N[real space擷取大小]*delta_r[data解出來的pixel resolution])
    delta_p = 1/(N2*pixel_size)
    pmap = np.zeros([N2,N2])
    center = ((N2+1)/2)-1
    for row in range(N2):
        for col in range(N2):
            pmap[row,col] = (np.sqrt((row-center)**2+(col-center)**2))*delta_p
    return pmap,delta_p