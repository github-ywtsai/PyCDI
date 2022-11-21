"""Module implementing nparray operations on GPU by using cupy."""


import numpy as np
import cupy as cp
import os
import math

class GPUArrayData(object):
    """Object that represents a set of 2D image arrays on GPU.
    
    :basic class for using GPU devices
    
    """

    def __init__(self, GPU_id = None, display = False):

        # Initialize GPU devices
        # :param GPU_id: The number(id) of GPU that the user want to use. Default: GPU_id = "0"
        # :param display: To show the print (description) or not. Default: display = False
        # Acceptable input: empty, number (e.g., 0, 2, etc.), arg (e.g., GPU_id = "0", or GPU_id = 1)
        # Examples for input:
            # Run_GPU = gpuoperations.GPUArrayData()
            # Run_GPU = gpuoperations.GPUArrayData(0)
            # Run_GPU = gpuoperations.GPUArrayData(GPU_id = "0")
            # Run_GPU = gpuoperations.GPUArrayData(GPU_id = 1)

        self.display = display
        self.GPU_id = GPU_id

        if self.display: print("Initialize GPU Operations......\n")
        try:
            if len(self.GPU_id) == 1: # TypeError check: int
                if type(int(self.GPU_id)) == int: # ValueError check: str
                    os.environ["CUDA_VISIBLE_DEVICES"] = self.GPU_id
                    if self.display: print("'CUDA_VISIBLE_DEVICES' = '%s' is chosen\n" %(self.GPU_id))
                    if self.display: print("GPU Operations OK!\n")
            
            elif len(self.GPU_id) != 1:
                raise

        except TypeError:
            if self.GPU_id == None:
                self.GPU_id = "0"
                os.environ["CUDA_VISIBLE_DEVICES"] = self.GPU_id
                if self.display: print("GPU_id is None.\nDefault: 'CUDA_VISIBLE_DEVICES' = '0', is chosen\n")
                if self.display: print("GPU Operations OK!\n")

            elif type(self.GPU_id) == bool:
                print(50*'='+' '+'Notice!!!'+' '+50*'='+'\n')
                print("Please input the true type of GPU id, e.g., GPU_id = '0' or GPU_id = 1\n")
                print('Fail to Initialize GPU Operations\n')
                print(111*'='+'\n')

            elif self.GPU_id == bool:
                print(50*'='+' '+'Notice!!!'+' '+50*'='+'\n')
                print("Please input the true type of GPU id, e.g., GPU_id = '0' or GPU_id = 1\n")
                print('Fail to Initialize GPU Operations\n')
                print(111*'='+'\n')

            else:
                self.GPU_id = str(self.GPU_id)
                os.environ["CUDA_VISIBLE_DEVICES"] = self.GPU_id
                if self.display: print("'CUDA_VISIBLE_DEVICES' = '%s' is chosen\n" %(self.GPU_id))
                if self.display: print("GPU Operations OK!\n")

        except ValueError:
            print(50*'='+' '+'Notice!!!'+' '+50*'='+'\n')
            print("Please input the true type of GPU id, e.g., GPU_id = '0' or GPU_id = 1\n")
            print('Fail to Initialize GPU Operations\n')
            print(111*'='+'\n')

        except:
            print(50*'='+' '+'Notice!!!'+' '+50*'='+'\n')
            print("Please input the true type of GPU id, e.g., GPU_id = '0' or GPU_id = 1\n")
            print('Fail to Initialize GPU Operations\n')
            print(111*'='+'\n')


    

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
                    if self.display: print("No. %d array, data type (%s) error!\nChenge complex from double to single precision\n" %(i, data_type))
                elif (data_type == 'float64'):
                    args[i] = args[i].astype('float32')
                    if self.display: print("No. %d array, data type (%s) error!\nChenge float from double to single precision\n" %(i, data_type))
            
            if (data_type=='float32' or data_type=='complex64'):
                if self.display: print("No. %d array, true data type: "%(i), data_type,"\n")
            
            if (data_type == bool):
                if self.display: print("No. %d array, true data type: "%(i), data_type,"\n")
        
        if nargs == 1:
            args = args[0]
            return args
        elif nargs > 1:
            return args
    

    # Modify the data type: single to double percision
    # The input data should be np_ or cp_array with float, complex, or bool type
    def S2D(self, *args):
        nargs = len(args)
        args = list(args)
        for i in range(nargs):
            data_type = args[i].dtype

            if (data_type != 'complex128' or data_type != 'float64'):
                if (data_type == 'complex64'):
                    args[i] = args[i].astype('complex128')
                    if self.display: print("No. %d array, data type (%s) error!\nChenge complex from single to double precision\n" %(i, data_type))
                elif (data_type == 'float32'):
                    args[i] = args[i].astype('float64')
                    if self.display: print("No. %d array, data type (%s) error!\nChenge float from single to double precision\n" %(i, data_type))
            
            if (data_type=='float64' or data_type=='complex128'):
                if self.display: print("No. %d array, true data type: "%(i), data_type,"\n")
            
            if (data_type == bool):
                if self.display: print("No. %d array, true data type: "%(i), data_type,"\n")
        
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
            if self.display: print("The class of No. %d array --> "%(i), type(args[i]))

        if self.display: print("Change array to cupy_array\n")
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
            if self.display: print("The class of No. %d array --> "%(i), type(args[i]))
            
        if self.display: print("Change array to numpy_array\n")
        if nargs == 1:
            args = args[0]
            return args
        elif nargs > 1:
            return args