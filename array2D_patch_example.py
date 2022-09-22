# import FSC_patch as FSCp
# import numpy as np
# import matplotlib.pyplot as plt
# A = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]])
# B = np.array([[0,0,0,0,0],[1,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
# [C,D] = FSCp.FSC_patch(A,B)
# plt.figure()
# plt.imshow(C)  
# plt.figure()
# plt.imshow(D)
# plt.show()
####################################################################################
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import FSC_patch as FSCp
#做一個IR sample
img = Image.open('IR.jpg')
Object = img.convert('L')
A = np.zeros([500,500])
A[0:128,0:136] = Object
B = np.zeros([500,500])
B[250:378,250:386] = Object
[C,D] = FSCp.FSC_patch(A,B)
plt.figure()
plt.imshow(A)  
plt.figure()
plt.imshow(B)
plt.figure()
plt.imshow(C)  
plt.figure()
plt.imshow(D)
plt.show()