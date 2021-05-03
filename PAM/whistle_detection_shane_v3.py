# Description: New version of the whistle detection
#   1. python version
#   2. new SNR definition
#   3. modified narrow-band & long time duration filter 
#   4. use DBSCAN to deal with clustering


# Read through wave file and detect the following 
# features.
#   1. Whistle start fre
#   2. Whistle end fre
#   3. Whistle start time 
#   4. Whistle end time 
#   5. Whistle duration time 
#   6. Whistle total numbre count  
#   7. Whislte total duration 
######################################################

# basic package you definitly know
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os.path
import time

# package for read wave file 
import soundfile as sf

# package for signal processing
import scipy.signal
from scipy.fftpack import fft

# package for image processing
import cv2

# package for clustering
from sklearn.cluster import DBSCAN

class ShaneWhistleDetector():
    def __init__(self, waf_file_name):
        self.data, self.fs = sf.read(waf_file_name)
        if(len(self.data.shape)==2):
            self.data = self.data[:,0]
        print("Successfully get {} data:".format(waf_file_name))
        print("Total duration: {} (Sec)".format(len(self.data)/self.fs))
        print("Sampling rate: {} (S/s)".format(self.fs))       
    # Set all needed parameters    
    def onStartUp(self):
        # Parameters for removeClick function
        self.threshold_click_removal = 5
        self.power_click_removal = 6

        # Parameters for computeSTFT function
        self.fft_number = 1024
        self.overlab = 0.5
        self.window_type = "hann"
        self.sensitivity = 211
        
        # Parameters for bandPassWithSNRFilter function
        self.start_fre = 2500
        self.end_fre = 10000
        self.threshold_SNR = 2

        # Parameters for whistleFeatureFilter function
        self.frequency_width = 50
        self.time_duration = 0.02

        # Parameters for detectWhistle function
        self.whistle_detector_frame = 2
        self.time_start_index = 0

        # Count numbers for whistle detector features
        self.whistle_count = 0
        self.whistle_duration = 0

    # Print out all whislte detector parameters
    def printSetting(self):
        print("######################################################")
        print("#         Parameters for removeClick function        #")
        print("######################################################")
        print("     1. Threshold: \t{}".format(self.threshold_click_removal))
        print("     2. Power: \t\t{}".format(self.power_click_removal))
        print("")
        print("------------------------------------------------------")
        print("######################################################")
        print("#         Parameters for computeSTFT function        #")
        print("######################################################")
        print("     1. FFT Number: \t\t{}".format(self.fft_number))
        print("     2. Overlab: \t\t{}".format(self.overlab))
        print("     3. Window Type: \t\t{}".format(self.window_type))
        print("     4. Hydrophone Sensitivity: {}".format(self.overlab))
        print("")
        print("------------------------------------------------------")
        print("######################################################")
        print("#    Parameters for bandPassWithSNRFilter function   #")
        print("######################################################")
        print("     1. Start Frequency: \t{} Hz".format(self.start_fre))
        print("     2. End Frequency: \t\t{} Hz".format(self.end_fre))
        print("     3. SNR Threshold: \t\t{}".format(self.threshold_SNR))
        print("")
        print("------------------------------------------------------")
        print("######################################################")
        print("#    Parameters for whistleFeatureFilter function    #")
        print("######################################################")
        print("     1. Frequency Width: \t{} Hz".format(self.frequency_width))
        print("     2. Time duration: \t\t{} Sec".format(self.time_duration))
        print()
        print("------------------------------------------------------")
        print("######################################################")
        print("#        Parameters for detectWhistle function       #")
        print("######################################################")
        print("     1. Whistle Detector Window: {} Sec".format(self.whistle_detector_frame))
        print("")
        print("------------------------------------------------------")


    #################################################
    # first step, click removal through time series #
    #################################################
    # Parameters:
    #   1. threshold
    #   2. power
    def removeClick(self,x):
        x = np.array(x*100)
        m = np.mean(x)
        SD = np.std(x)
        w = 1.0/(1+((x-m)/self.threshold_click_removal*SD)**self.power_click_removal)
        return w*x/100   

    #############################
    # Second step, STFT process #
    #############################
    # Parameters:
    #   1. fft_number
    #   2. window_type
    #   3. overlap ratio
    #   4. hydrophone sensitivity
    def computeSTFT(self):
        PSD_clickRemoval = []
        hop_size = math.ceil(self.fft_number*(1-self.overlab))
        window = scipy.signal.get_window(self.window_type, self.fft_number, fftbins=True)
        fft_half_number = math.ceil((self.fft_number+1)/2)

        total_stft_frame = (self.whistle_detector_frame*self.fs-self.fft_number)//hop_size+1
        total_sample_count = (total_stft_frame-1)*hop_size+self.fft_number

        sample_time_series = self.data[self.time_start_index:self.time_start_index+total_sample_count]
        self.time_start_index += total_sample_count
        
        
        end_index = self.fft_number
        for i in range(total_stft_frame):
            weighted_data = self.removeClick(sample_time_series[end_index-self.fft_number:end_index])
            windowed_data = window*weighted_data
            z = fft(windowed_data)[:fft_half_number]/self.fft_number*2
            psd = 20*np.log10(abs(z)**2) + self.sensitivity
            PSD_clickRemoval.append(psd)
            end_index += hop_size
        PSD_clickRemoval = np.array(PSD_clickRemoval)
        return PSD_clickRemoval

    #############################
    # Third step, image bluring #
    #############################
    # applying median bluring



    #############################################
    # Fourth step, band passing & SNR threshold #
    #############################################
    # Parameters:
    #   1. start freq
    #   2. end freq
    #   3. SNR threshold
    def bandPassWithSNRFilter(self, median_blur):
        df = self.fs/self.fft_number
        start_index = math.floor((self.start_fre)/df)
        end_index = math.ceil((self.end_fre)/df)

        noise_start_fre = 15000
        noise_end_fre = 20000
        noise_start_index = math.floor((noise_start_fre)/df)
        noise_end_index = math.ceil((noise_end_fre)/df)

        median_f = np.median(median_blur[:,noise_start_index:noise_end_index],axis=1)
        median_f = median_f.reshape(len(median_f),1)
        SNR = 10*np.log10(median_blur[:,start_index:end_index]/median_f)
        high_SNR = SNR>self.threshold_SNR
        return high_SNR

    #####################################################
    # Fifth step, narrow bandwidth & long time duration #
    #####################################################
    # Parameters:
    #   1. frequency width
    #   2. time duration
    def whistleFeatureFilter(self, high_SNR):
        hop_size = math.ceil(self.fft_number*(1-self.overlab))
        dt = hop_size/self.fs
        df = self.fs/self.fft_number

        col_size = math.ceil(self.time_duration/dt)
        row_size = math.ceil(self.frequency_width/df)

        image_row = high_SNR.T.shape[0]
        image_col = high_SNR.T.shape[1]

        padding = np.zeros((image_row+row_size-1, image_col+col_size-1))
        padding[row_size//2:row_size//2+image_row,col_size//2:col_size//2+image_col] = high_SNR.T

        detection = np.zeros((image_row, image_col))
        for i in range(col_size):
            detection += padding[:,i:i+image_col]
        return detection==col_size

    def DBSCANCluster(self, detection, filename):
        hop_size = math.ceil(self.fft_number*(1-self.overlab))
        df = self.fs/self.fft_number
        dt = hop_size/self.fs

        (x,y) = np.nonzero(detection.T)
        if len(x):
            point = np.array([x,y]).T
            clustering=DBSCAN(eps=20,min_samples=10).fit(point)

            point_without_outlier = point[clustering.labels_!=-1]
            new_label = clustering.labels_[clustering.labels_!=-1]
            point_x = point_without_outlier.T[0]*dt+self.time_start_index/self.fs
            point_y = (point_without_outlier.T[1]*df+self.start_fre)/1000
            
            length = len(set(new_label))
            col = ['Start Time','End Time','Start Freq','End Freq','Duration']
            feature = []
            for i in range(length):
                cluster_x = point_x[new_label==i]
                cluster_y = point_y[new_label==i]
                if(cluster_x[-1]-cluster_x[0]>0.05):
                    feature.append([cluster_x[0], cluster_x[-1], cluster_y[0], cluster_y[-1], cluster_x[-1]-cluster_x[0]])
                    self.whistle_duration += cluster_x[-1]-cluster_x[0]
                    self.whistle_count += 1
            df = pd.DataFrame(feature, columns=col)
            if os.path.isfile(filename):
                df.to_csv(filename, mode='a', header=False, index=False)
            else:
                df.to_csv(filename, mode='w', header=True, index=False)  
    
    def detectWhistle(self, sec=0, csv_file='whislte_detection_outcome.csv'):
        wave_file_long = len(self.data)/self.fs
        if(sec>wave_file_long):
            print("Input parameter sec is too large, the max number is {}.".format(wave_file_long))
            return 
        elif(sec!=0):
            cut = wave_file_long-sec
            print("Detecting whistle in first {} seconds of the file ...\n".format(sec))
        else:
            print("Detecting whistle in full file ... \n")
            cut = 0
        self.onStartUp()
        self.printSetting()

        self.stft_with_click_removal_time = []
        self.median_blur_time = []
        self.high_SNR_time = []
        self.detection_time = []
        self.clustering_time = []
        
        start = time.time()
        while(self.time_start_index+self.whistle_detector_frame*self.fs<=len(self.data)-cut*self.fs):
            t_1 = time.time()
            PSD_clickRemoval = self.computeSTFT()
            t_2 = time.time()
            self.stft_with_click_removal_time.append(t_2-t_1)
            median_blur = cv2.medianBlur(PSD_clickRemoval.astype(np.float32),3)
            t_3 = time.time()
            self.median_blur_time.append(t_3-t_2)
            high_SNR = self.bandPassWithSNRFilter(median_blur)
            t_4 = time.time()
            self.high_SNR_time.append(t_4-t_3)
            detection = self.whistleFeatureFilter(high_SNR)
            t_5 = time.time()
            self.detection_time.append(t_5-t_4)
            self.DBSCANCluster(detection, csv_file)
            t_6 = time.time()
            self.clustering_time.append(t_6-t_5)
        end = time.time()
        print("elapsed time: {} Sec".format(end-start))