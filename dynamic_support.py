# X-ray image reconstruction from a diffraction pattern alone
# page2 paragraph2
import numpy as np
import scipy.signal
from cupyx.scipy.signal import fftconvolve
import cupy as cp

def dynamic_supp(rho_p, threshold, std_deviation):
    size = np.shape(rho_p)
    x, y = np.meshgrid(np.linspace(-1,1,size[1]), np.linspace(-1,1,size[0]))
    d = np.sqrt(x*x+y*y)
    sigma, mu = std_deviation, 0.0
    Gaussion = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    Gaussion = cp.asarray(Gaussion)

    multi = fftconvolve(rho_p,Gaussion,mode = 'same')
    adj_multi = cp.full(np.shape(multi), False)
    adj_multi[multi > cp.amax(multi)*threshold] = True

    return adj_multi