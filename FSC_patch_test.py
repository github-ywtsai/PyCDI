import FSC_patch as FSCp
import numpy as np
import matplotlib.pyplot as plt
A = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]])
B = np.array([[0,0,0,0,0],[1,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
[C,D] = FSCp.FSC_patch(A,B)
plt.figure()
plt.imshow(C)  
plt.figure()
plt.imshow(D)
plt.show()