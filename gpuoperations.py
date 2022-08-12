"""Module implementing nparray operations on GPU by using cupy."""


import numpy as np
import cupy as cp
import math

class GPUArrrayData(object):
    """Object that represents a set of 2D image arrays on GPU.
    
    :param shape: total shape of all images
    :param dl: list of dilations in the network
    :param nin: number of input images of network
    """