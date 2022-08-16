import numpy as np
import scipy.signal


def dynamic_supp(last_Rspace, autocorr_support, threshold, std_deviation):
    size = np.shape(last_Rspace)
    x, y = np.meshgrid(np.linspace(-1,1,size[1]), np.linspace(-1,1,size[0]))
    d = np.sqrt(x*x+y*y)
    sigma, mu = std_deviation, 0.0
    Gaussion = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )

    multi = scipy.signal.fftconvolve(last_Rspace,Gaussion,mode = 'same')
    adj_multi = np.full(np.shape(multi), False)
    adj_multi[multi > np.max(multi)*threshold] = True
    # adj_multi[autocorr_support==0] = False

    # 以下在把切過的support再放大一點 避免切太深
    # b = np.ones((2,2))
    # support_conv = scipy.signal.fftconvolve(adj_multi,b,mode = 'same') 
    # adj_support = np.full(np.shape(adj_multi), False)
    # adj_support[support_conv>0] = True
    return adj_multi