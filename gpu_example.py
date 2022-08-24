""" This example demonstrates how the module, gpuoperations.py, be used. """

import numpy as np
import os, time
import matplotlib.pyplot as plt


# define FFT test function
def FFT_test(arr):
    global a_cpu, b_cpu, c_cpu, fake_a_cpu_result
    s = time.time()
    for i in range(10):
        ss = time.time()
        print(i)
        arr2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr)))
        phase = np.angle(arr2)
        A = np.absolute(arr2)
        R = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(arr2))))
        F_new = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr2)))
        A_new = np.absolute(arr2)
        ee = time.time()
        print(ee-ss)
    e = time.time()
    print(e-s)
    return arr2

# Test array on CPU or GPU
def RunCPUorGPU(GPU=False):
    global a_cpu, b_cpu, c_cpu, fake_a_cpu_result
    if GPU == False:

        print("CPU is used")
        # run FFT_test function
        fake_a_cpu_result = FFT_test(a_cpu)
        return print("CPU test finish")
    elif GPU == True:

        # Initialize GPU devices
        import gpuoperations
        r_gpu = gpuoperations.GPUArrayData(GPU_id = "0")

        # Converts data type to single percision
        a_cpu, b_cpu, c_cpu = r_gpu.D2S(a_cpu, b_cpu, c_cpu)

        # Converts an object to gpu array
        a_cpu, b_cpu, c_cpu = r_gpu.Np2Cp(a_cpu, b_cpu, c_cpu)

        # run FFT_test function
        fake_a_cpu_result = FFT_test(a_cpu)

        # Converts an object to cpu array
        a_cpu, b_cpu, c_cpu, fake_a_cpu_result = r_gpu.Cp2Np(a_cpu, b_cpu, c_cpu, fake_a_cpu_result)
        return print("GPU test finish")


# Define array
a_cpu = np.random.rand(4000,4000)+np.random.rand(4000,4000)*1j
b_cpu = np.random.rand(4000,4000).astype('float32')
c_cpu = np.zeros((4000,4000)).astype(bool)

# Choose CPU or GPU to be used: True for using
RunCPUorGPU(GPU=True)

# Note that only numpy array can be used in matplotlib
plt.imshow(np.log(np.abs(fake_a_cpu_result)))
plt.show()
