import numpy as np
def CM(Array2D):
    size = np.shape(Array)
    row_index = np.zeros(size)
    col_index = np.zeros(size)
    for n in range(1,size[0]+1):
        row_index[n-1,:] = n
    for m in range(1,size[1]+1):
        col_index[:,m-1] = m

    row_center = round(np.sum(Array*row_index)/np.sum(Array))-1
    col_center = round(np.sum(Array*col_index)/np.sum(Array))-1


    return row_center,col_center