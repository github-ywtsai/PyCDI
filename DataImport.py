## usage:
## from PyEigerData import EigerData
## handle = EigerData() to create toe object handle

import h5py
import hdf5plugin
import os
import numpy as np
import pandas

class EigerBasic:
    # open(master_file): open and load a master file
    # readFrame(ReqSNs): read and return the requested data of frames in ReqSNs.
    # sumFrame(ReqSNs): return the sum of all the frames within ReqSNs
    # avgFrame(ReqSNs): return the average of all the frames within ReqSNs by the number of frames
    # normFrame(ReqSNs): return the normalized of all the frames within ReqSNs by the number of frames and the count times
    def __init__(self):
        self.Header = dict()
        self.Description = 'EigerBasic'

    def open(self,MasterFile): # master file information
        ## check file exist or not
        ## if file doesn't exist, return False
        if os.path.exists(MasterFile):
            self.Header['MasterFP'] = os.path.abspath(MasterFile)
            self.Header['MasterFF'], self.Header['MasterFN'] = os.path.split(self.Header['MasterFP'])          
        else:
            print('File does not exist.')
            return False

        ## read header
        self.__readHeader()

    def __readHeader(self):
        ## read basic information
        FO = h5py.File(self.Header['MasterFP'],'r') # open file object
        self.Header['BitDepthImage'] = FO['/entry/instrument/detector/bit_depth_image'][()]
        self.Header['XPixelsInDetector'] = FO['/entry/instrument/detector/detectorSpecific/x_pixels_in_detector'][()]
        self.Header['YPixelsInDetector'] = FO['/entry/instrument/detector/detectorSpecific/y_pixels_in_detector'][()]
        self.Header['CountTime'] = FO['/entry/instrument/detector/count_time'][()]
        self.Header['DetectorDistance'] = FO['/entry/instrument/detector/detector_distance'][()]
        self.Header['XPixelSize'] = FO['/entry/instrument/detector/x_pixel_size'][()]
        self.Header['YPixelSize'] = FO['/entry/instrument/detector/y_pixel_size'][()]
        self.Header['Wavelength'] = FO['/entry/instrument/beam/incident_wavelength'][()]*1E-10 # convert from A to meter
        self.Header['BeamCenterX'] = FO['/entry/instrument/detector/beam_center_x'][()]
        self.Header['BeamCenterY'] = FO['/entry/instrument/detector/beam_center_y'][()]
        self.Header['PixelMask'] = FO['/entry/instrument/detector/detectorSpecific/pixel_mask'][()].astype(bool) # convert the mask to logical array

        ## create link data information
        self.Header['LinkData'] = np.array([])
        self.Header['ContainFramesInLinkData'] = np.array([],dtype = 'int32')
        LinkDataFNList = np.array(FO['/entry/data']) # linked data file name list
        
        for SN in range(0,len(LinkDataFNList)): # check file exist or not
            FF = self.Header['MasterFF']
            FN = self.Header['MasterFN'].replace('_master','_'+LinkDataFNList[SN])
            FP = os.path.join(FF,FN)
            if os.path.exists(FP):
                self.Header['LinkData'] = np.append(self.Header['LinkData'],LinkDataFNList[SN])
                self.Header['ContainFramesInLinkData'] = np.append(self.Header['ContainFramesInLinkData'],FO['entry/data'][LinkDataFNList[SN]].shape[0]) # DataShape: (frame,x pixel, y pixel)
            else:
                break       
        self.Header['ContainFrames'] = sum(self.Header['ContainFramesInLinkData']) 

        FO.close() # close file object

    def __readSingleFrame(self,ReqSN):
        ## basic function for read single data 
        ## ReqSN: require frame SN
        # SN start from 1 and idx start from 0
        FO = h5py.File(self.Header['MasterFP'],'r')

        ## find ReqSN in links
        NLinkData = len(self.Header['LinkData'])
        NextLinkDataSN = 1
        for SN in range(0,NLinkData):
            StartSN = NextLinkDataSN; # Start SN in this linked data
            EndSN = StartSN + self.Header['ContainFramesInLinkData'][SN] - 1 # End SN in this linked data
            NextLinkDataSN = EndSN + 1 # Start SN in next linked data
            
            if (ReqSN >= StartSN) & (ReqSN <= EndSN):
                FrameSNInLinkData = ReqSN - StartSN + 1
                FrameIdxInLinkData = FrameSNInLinkData - 1
                break

        SingleFrameData = FO['entry/data/'+ self.Header['LinkData'][SN]][FrameIdxInLinkData]
        FO.close()
        return SingleFrameData
    
    def __createReqSNs(self,ReqSNs):
        # convert Request SN from int or list to np array
        if isinstance(ReqSNs, int):
            ReqSNs = [ReqSNs]
        ReqSNs = np.array(ReqSNs)
        
        # remove SN out of the sheet contained
        if np.any(ReqSNs > self.Header['ContainFrames']) or np.any(ReqSNs < 1):
            print('Request SNs must be within 1 to %d.'%(self.Header['ContainFrames']))
            print('SNs out of range have been removed.')
            ReqSNs = np.delete(ReqSNs,ReqSNs > self.Header['ContainFrames'])
            ReqSNs = np.delete(ReqSNs,ReqSNs < 1)
        
        if ReqSNs.size == 0:
            ReqSNs = False
            print('Request SNs is unavailable.')
        return ReqSNs
    
    def readFrame(self,ReqSNs):
        ReqSNs = self.__createReqSNs(ReqSNs)
        if ReqSNs is False:
            return

        NReqs = len(ReqSNs)
        DataBuffer = np.zeros([NReqs,self.Header['YPixelsInDetector'],self.Header['XPixelsInDetector']])

        for Idx in range(0,NReqs):
            ReqSN = ReqSNs[Idx]
            Data = self.__readSingleFrame(ReqSN)
            DataBuffer[Idx] = Data
        
        return DataBuffer
    
    def sumFrame(self,ReqSNs):
        ReqSNs = self.__createReqSNs(ReqSNs)
        if ReqSNs is False:
            return False

        DataBuffer = np.zeros([self.Header['YPixelsInDetector'],self.Header['XPixelsInDetector']])
        for SN in ReqSNs:
            DataBuffer = DataBuffer + self.__readSingleFrame(SN)
        
        return DataBuffer
    
    def avgFrame(self,ReqSNs):
        ReqSNs = self.__createReqSNs(ReqSNs)
        if ReqSNs is False:
            return False
        
        NReqs = len(ReqSNs)
        DataBuffer = self.sumFrame(ReqSNs)/NReqs
        return DataBuffer
    
    def normFrame(self,ReqSNs):
        ReqSNs = self.__createReqSNs(ReqSNs)
        if ReqSNs is False:
            return False

        DataBuffer = self.avgFrame(ReqSNs)/self.Header['CountTime']
        return DataBuffer
        
    # def __convCSV2ROI(self,CSVFP):
    #     # convert CSV from ImageJ to Boolean ROI
    #     if not os.path.exists(CSVFP):
    #         print('File does not exist.')
    #         return False
    #     else:
    #         CSVData = pandas.read_csv(CSVFP)
    #         X = CSVData['X']
    #         Y = CSVData['Y']
    #         Value = CSVData['Value']
    #         DataLength = CSVData.shape[0]
    #         BoolROI = np.zeros([self.Header['YPixelsInDetector'],self.Header['XPixelsInDetector']],dtype = bool)
    #         for Idx in range(0,DataLength):
    #             col = X[Idx]
    #             row = Y[Idx]
    #             BoolROI[row,col] = True
    #
    #         return BoolROI
