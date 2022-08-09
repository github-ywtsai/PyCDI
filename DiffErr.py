import numpy as np

def DiffErr(amplitude, measured_amplitude):
    diff_err = np.nansum(np.absolute(amplitude - measured_amplitude))/np.nansum(measured_amplitude)
    return diff_err