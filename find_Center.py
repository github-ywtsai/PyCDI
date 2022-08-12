import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from skimage.transform import resize, rotate
from skimage.feature import match_template
import cv2
import time


def Method_CV2(data_ndarray, mask_ndarray, method = 'CCORR_NORMED', diameter = 100):

    def dead_pixel(All_ProcessedData, Mask_ndarray):
        dead_pix = np.max(All_ProcessedData)
        print('The value of dead pixel: ', dead_pix)

        if np.isnan(dead_pix) == True:
            print('All dead pixels of original data have been depicted by Nan\n')
            return All_ProcessedData
        elif np.isnan(dead_pix) == False:
            All_ProcessedData[Mask_ndarray == False] = np.nan
            print('All dead pixels of original data were changed to Nan\n')
            return All_ProcessedData


    #@jit(nopython = True)
    def Intensity_normalization(normalization_ndarray):
        print('Start Intensity_normalization\n')
        Data_max = np.nanmax(normalization_ndarray)
        Data_min = np.nanmin(normalization_ndarray)

        # for y in range(0, normalization_ndarray.shape[0]):
        #     for x in range(0, normalization_ndarray.shape[1]):
        #         if  np.isnan(normalization_ndarray[y,x]) == False:
        #             normalization_ndarray[y,x] = (normalization_ndarray[y,x]-Data_min)/(Data_max-Data_min)
        #         elif np.isnan(normalization_ndarray[y,x]) == True:
        #             normalization_ndarray[y,x] = 0.0

        # Using array directly to improve speed
        normalization_ndarray = (normalization_ndarray- Data_min)/(Data_max-Data_min)
        normalization_ndarray[np.isnan(normalization_ndarray)] = 0.0
        
        #normalization_ndarray = normalization_ndarray.astype('float32')
        print('End of Intensity_normalization\n')
        return normalization_ndarray
    
    
    def def_mask(Mask_ndarray):
        # Note that 1.0 means "use this pixel" and 0.0 means "do not". 
        # If the value between 0.0 and 1.0, it represents the weight.
        print('Start def_mask\n')
        # for y in range(0, Mask_ndarray.shape[0]):
        #     for x in range(0, Mask_ndarray.shape[1]):
        #         if Mask_ndarray[y,x] == False:
        #             Mask_ndarray[y,x] = 0.0 
        #         elif Mask_ndarray[y,x] == True:
        #             Mask_ndarray[y,x] = 1.0
        Mask_ndarray[Mask_ndarray == False] = 0.0
        Mask_ndarray[Mask_ndarray == True] = 1.0
        #Mask_ndarray = Mask_ndarray.astype('float32')
        print('End of def_mask\n')
        return Mask_ndarray
    
    
    
    #@jit(nopython = True)
    def findCenterofMass(CM_ndarray):
        CenterofMass = []
        TotalMass = 0.0
        print('Start findCenterofMass\n')

        # for y in range(0, CM_ndarray.shape[0]):
        #     for x in range(0, CM_ndarray.shape[1]):
        #         if  np.isnan(CM_ndarray[y,x]) == False:
        #             CenterofMass.append([y*CM_ndarray[y,x], x*CM_ndarray[y,x]])
        #             TotalMass += CM_ndarray[y,x]
        #
        #
        # CenterofMass = np.array(CenterofMass)
        # CenterofMass = [int(np.sum(CenterofMass[:,0], axis = 0)/TotalMass), int(np.sum(CenterofMass[:,1], axis = 0)/TotalMass)]

        # Using array directly to improve speed
        y,x = np.mgrid[range(0, CM_ndarray.shape[0]),range(0, CM_ndarray.shape[1])]
        TotalMass = np.nansum(CM_ndarray)
        CenterofMass = [int(i) for i in ([np.nansum(y*CM_ndarray),np.nansum(x*CM_ndarray)]/TotalMass)] # Change type from float to int
        print('End of findCenterofMass\n')
        return CenterofMass

    # ================================[2022/08/08] to check this function that are run or not.==========================
    new_center = None # to record the new center calculated by this function. Type: list.
    plot_data = None # to record the raw data including targeted, rotated, original, and zooming data. Type: dictionary.
    # ==================================================================================================================
    TM_methods = {'SQDIFF':0, 'SQDIFF_NORMED':1, 'CCORR':2, 'CCORR_NORMED':3, 'CCOEFF':4, 'CCOEFF_NORMED':5}
    TM_method = TM_methods[method]
    DiameterInTM = diameter
    All_ProcessedData = np.copy(data_ndarray)
    Mask_ndarray = np.copy(mask_ndarray)

    All_ProcessedData = dead_pixel(All_ProcessedData, Mask_ndarray) # define the dead pixel
    
    #YCenterInTM, XCenterInTM = 1500,1500
    #YCenterInTM, XCenterInTM = np.where(All_ProcessedData == np.nanmax(All_ProcessedData))
    YCenterInTM, XCenterInTM = findCenterofMass(All_ProcessedData)


    Symmetric_ProcessedData = All_ProcessedData[int(int(YCenterInTM)-(DiameterInTM/2)):int(int(YCenterInTM)+(DiameterInTM/2)),int(int(XCenterInTM)-(DiameterInTM/2)):int(int(XCenterInTM)+(DiameterInTM/2))]
    Symmetric_rotate_ProcessedData = (rotate(Symmetric_ProcessedData, 180.0 ))
    

    Symmetric_rotate_ProcessedData_norm = Intensity_normalization(Symmetric_rotate_ProcessedData)
    Symmetric_rotate_ProcessedData_norm = Symmetric_rotate_ProcessedData_norm.astype('float32')
    
    All_ProcessedData_norm = Intensity_normalization(All_ProcessedData)
    All_ProcessedData_norm = All_ProcessedData_norm.astype('float32')

    
    F32_mask = def_mask(Mask_ndarray) # F32 means 'float32'
    F32_mask = F32_mask.astype('float32')
    F32_mask = F32_mask[int(int(YCenterInTM)-(DiameterInTM/2)):int(int(YCenterInTM)+(DiameterInTM/2)),int(int(XCenterInTM)-(DiameterInTM/2)):int(int(XCenterInTM)+(DiameterInTM/2))]       
    F32_mask = cv2.rotate(F32_mask, cv2.ROTATE_180)


    print('Start matchTemplate\n')
    results = cv2.matchTemplate(All_ProcessedData_norm, Symmetric_rotate_ProcessedData_norm, TM_method, mask = F32_mask)
    print('End of matchTemplate\n')

    Min_Max_Loc = cv2.minMaxLoc(results) # return (min_val, max_val, min_loc, max_loc)
    if TM_method == 0 or TM_method == 1:  
        Min_Max_Loc = Min_Max_Loc[2] # minimum location
    else:
        Min_Max_Loc = Min_Max_Loc[3] # maximum location


    left_up = Min_Max_Loc ## this point is left_up cornor
    x_pixel_posi, y_pixel_posi = left_up[0], left_up[1]
    right_bottom = (y_pixel_posi + Symmetric_rotate_ProcessedData_norm.shape[0], x_pixel_posi + Symmetric_rotate_ProcessedData_norm.shape[1])
    results_ProcessedData = All_ProcessedData[y_pixel_posi:right_bottom[0], x_pixel_posi:right_bottom[1]]


    result_center = [(right_bottom[0] + y_pixel_posi)/2.0, (right_bottom[1] + x_pixel_posi)/2.0]
    new_center = [(result_center[0] + int(YCenterInTM))/2, (result_center[1] + int(XCenterInTM))/2]
    plot_data = {"target":Symmetric_ProcessedData, "rotation":Symmetric_rotate_ProcessedData, "origin":All_ProcessedData, "result":results_ProcessedData, "match_points":left_up}
   
   
    print('result_center_y: ',  result_center[0], 'result_center_x: ', result_center[1])
    print('new_center_y: ',new_center[0] , 'new_center_x: ',new_center[1] )


    return new_center, plot_data


def plot_Center(plot_data):
    
    if type(plot_data) != dict:
        print("*"*70 + "\nError! data type does not match.\nThe input should be 'dictionary' in which there are five 'keys'.\nCheck function 'find_Center' that was truly run.\n" + "*"*70)
        return print("plot_center fail!")
    elif type(plot_data) == dict:
        get_keys = list(plot_data.keys())
        num_keys = len(plot_data)
        if (num_keys == 5):
            if (get_keys[0] == "target" and get_keys[1] == "rotation" and get_keys[2] == "origin" and get_keys[3] == "result" and get_keys[4] == "match_points" ):
                
                plt.subplot(141)
                plt.title('Target image')
                plt.imshow(np.log(plot_data[get_keys[0]]))

                plt.subplot(142)
                plt.title('Target "rotated image"')
                plt.imshow(np.log(plot_data[get_keys[1]]))

                plt.subplot(143)
                plt.title('Match in original image')
                plt.imshow(np.log((plot_data[get_keys[2]] )))
                rect = plt.Rectangle(plot_data[get_keys[4]], plot_data[get_keys[0]].shape[1], plot_data[get_keys[0]].shape[0], edgecolor='red', facecolor='none')
                plt.subplot(143).add_patch(rect)

                plt.subplot(144)
                plt.title('Zoom in of original image')
                plt.imshow(np.log(plot_data[get_keys[3]]))

                plt.show()

            elif (get_keys[0] != "target" or get_keys[1] != "rotation" or get_keys[2] != "origin" or get_keys[3] != "result" or get_keys[4] != "match_points" ):
                print("*"*70 + "\nError! data type does not match.\nThe input should be 'dictionary' in which there are five 'keys'.\nCheck function 'find_Center' that was truly run.\n" + "*"*70)
                return print("plot_center fail!")
        
        else:
            print("*"*70 + "\nError! data type does not match.\nThe input should be 'dictionary' in which there are five 'keys'.\nCheck function 'find_Center' that was truly run.\n" + "*"*70)
            return print("plot_center fail!")
        
