#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 16:53:41 2021

@author: Sid
"""

import pandas as pd 
import numpy as np 
from scipy import stats 
import matplotlib.pyplot as plt 


class preprocessing: 
    def extract_pt_IDs(input_dataframe): 
        # MUST input motor_change df as arg
        pt_IDs = pd.Series(input_dataframe['Pt ID'])
        return pt_IDs
    
    
    def extract_delta_FMs(input_dataframe): 
        # MUST input motor_change as arg
        pt_deltas = pd.Series(input_dataframe['Score'])
        return pt_deltas 
    
    
class dti_measures: 
    def calculate_DTI_change(pretherapy, posttherapy, list_IDs): 
        # extract ROIs and cast to numpy (string) array
        ROIs = np.asarray(pretherapy['Region'],dtype=str).reshape(len(pretherapy),1)
        # convert input data frames to numpy arrays for easier manipuation 
        pretherapy = np.asarray(pretherapy)
        posttherapy = np.asarray(posttherapy)
        # store number of pts in cohort (ignore first col which is list of ROIs)
        num_pts = np.shape(pretherapy)[1]
        # calculate difference in measure after therapy 
        difference = posttherapy[:, 1:num_pts] - pretherapy[:, 1:num_pts]
        # convert to pd.DataFrame
        difference_df = pd.DataFrame(data=difference, columns = list_IDs)
        # merge ROI series and difference dataframe
        difference_df.insert(0,"Region",ROIs)
        return difference_df
    
    
    def correlate_measures(dti_dataframe, list_motor_scores): 
        correlation = np.empty((len(dti_dataframe),2))
        for i in range(len(dti_dataframe)): 
            # extract current region DTI values into pd.Series 
            x = dti_dataframe.iloc[i,:] 
            # store current ROI 
            curr_ROI = x['Region']
            # remove ROI from pd.Series
            x= x.drop(['Region'])
            # calculate (spearman correlation, p val) bw (DTI measure, deltaFM)
            correlation[i,:] = stats.spearmanr(x,list_motor_scores)
            print(curr_ROI, 'Spearman Correlation, p value: ', correlation[i,:])
        return correlation 


class plot_dti_measures: 
    def plot_hemisphere_measures(dti_dataframe, list_motor_scores):
        # extract ROIs, convert to str array, in Ipsi/Contra/Ipsi/Contra order
        ROIs = np.asarray(dti_dataframe['Region'],dtype=str).reshape(2,2)
        # create figure and (2x2) axes object, all axes share x and y spacing
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        count = 0 
        # set axes titles 
        for i in range(int(len(dti_dataframe)/2)):
            for j in range(int(len(dti_dataframe)/2)): 
                # set scatterplot details 
                axes[i][j].set_title(ROIs[i][j]) # set titles to ROI names
                axes[i][j].set(ylim=(0,12)) # set y axis bounds
                # check for first column subplots 
                if (j == 0): 
                    # set shared y label
                    axes[i][j].set_ylabel("Change in UEFM") 
                # check for second row subplots 
                if (i == 1): 
                    # set shared x lable 
                    axes[i][j].set_xlabel("Log (Change in FA") 
                # extract dti values of current iteration 
                x = dti_dataframe.iloc[count,:]
                # remove region name 
                x = x[1:len(dti_dataframe.iloc[count,:])]
                # plot each (dti value, motor score) for each pt for each plot
                axes[i][j].scatter(x, list_motor_scores)
                count = count+1
        plt.show()
        return 0 
        
    def plot_cerebellar_measures(cbl_dataframe, list_motor_scores): 
        # extract ROIs, convert to str array, in Ipsi/Contra/Ipsi/Contra order
        ROIs = np.asarray(cbl_dataframe['Region'],dtype=str).reshape(14,2)
        return 0 
    
    
