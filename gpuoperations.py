"""Module implementing nparray operations on GPU by using cupy."""


import numpy as np
import cupy as cp
import os
import math

class GPUArrrayData(object):
    """Object that represents a set of 2D image arrays on GPU.
    
    :basic class for using GPU devices
    
    """

    def __init__(self, GPU_id = None):

        # Initialize GPU devices
        # :param GPU_id: The number(id) of GPU that the user want to use. Default: GPU_id = "0"

        print("Initialize GPU Operations......\n")
        if GPU_id == None:
            self.GPU_id = "0"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.GPU_id
            print("GPU_id is None.\nDefault: 'CUDA_VISIBLE_DEVICES' = '0', is chosen\n")
        elif GPU_id != "0":
            self.GPU_id = GPU_id
            os.environ["CUDA_VISIBLE_DEVICES"] = self.GPU_id
            print("'CUDA_VISIBLE_DEVICES' = '%s' is chosen\n" %(self.GPU_id))

        print("GPU Operations OK!\n")
    

    # Modify the data type: double to single percision
    # The input data should be np_ or cp_array with float, complex, or bool type
    def D2S(self, *args):
        nargs = len(args)
        args = list(args)
        for i in range(nargs):
            data_type = args[i].dtype

            if (data_type != 'complex64' or data_type != 'float32'):
                if (data_type == 'complex128'):
                    args[i] = args[i].astype('complex64')
                    print("No. %d, data type (%s) error!\nChenge complex from double to single precision\n" %(i, data_type))
                elif (data_type == 'float64'):
                    args[i] = args[i].astype('float32')
                    print("No. %d, data type (%s) error!\nChenge float from double to single precision\n" %(i, data_type))
            
            if (data_type=='float32' or data_type=='complex64'):
                print("No. %d, data type: "%(i), data_type,"\n")
            
            if (data_type == bool):
                print("No. %d, data type: "%(i), data_type,"\n")
        
        if nargs == 1:
            args = args[0]
            return args
        elif nargs > 1:
            return args


    # Converts an object to array 
    def Np2Cp(self, *args):
        nargs = len(args)
        args = list(args)
        for i in range(nargs):
            args[i] = cp.asarray(args[i])

        print("Change array to cupy_array\n")
        if nargs == 1:
            args = args[0]
            return args
        elif nargs > 1:
            return args
    

    # Returns an array on the host memory from an arbitrary source array
    def Cp2Np(self, *args):
        nargs = len(args)
        args = list(args)
        for i in range(nargs):
            args[i] = cp.asnumpy(args[i])
            
        print("Change array to numpy_array\n")
        if nargs == 1:
            args = args[0]
            return args
        elif nargs > 1:
            return args