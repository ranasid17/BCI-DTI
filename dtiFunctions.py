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
    
    
    def extract_motor_scores(input_dataframe): 
        # MUST input motor_change as arg
        series_motor_scores = pd.Series(input_dataframe['Score'])
        return series_motor_scores 
    
    
    def log_transform(input_dataframe, list_of_pts): 
        # calculate number of pts in cohort 
        num_pts = len(list_of_pts) + 1 
        # extract measured values, leaving out region names 
        extract_values = input_dataframe.iloc[:,1:num_pts]
        # convert dataframe to np array 
        extract_values = np.asarray(extract_values, dtype=float)
        # apply natural log to all indices 
        log_values = np.log(extract_values)
        return log_values 
    
    
    def extract_ROIs(input_dataframe): 
        # Extract ROIs from input df as series 
        ROIs = pd.Series(input_dataframe['Region'])
        return ROIs
    
    
    def reconstruct_df(input_array, list_of_cols, list_of_ROIs): 
        """ Use convert np array back to pandas dataframe. Apply this method 
            after converting df to np array for log transfrom to return back
            to df object. plot_dti_measures class only accepts df objects so 
            is helpful to have objects in correct structure. """ 
        # create df using pt IDs for columns 
        df = pd.DataFrame(data=input_array, columns=list_of_cols)
        # insert ROIs 
        df.insert(0, 'Region', list_of_ROIs)
        return df 
    
    
class feature_analysis: 
    def calculate_feature_change(pre_df, post_df, list_IDs): 
        """ First two input args MUST be pandas dataframes
            Method will not work if input args are numpy arrays """ 
        # extract ROIs and cast to numpy (string) array
        ROIs = np.asarray(pre_df['Region'],dtype=str).reshape(len(pre_df),1)
        # convert input data frames to numpy arrays for easier manipuation 
        pre_df = np.asarray(pre_df)
        post_df = np.asarray(post_df)
        # store number of pts in cohort (ignore first col which is list of ROIs)
        num_pts = np.shape(pre_df)[1]
        # calculate difference in measure after therapy 
        difference = post_df[:, 1:num_pts] - pre_df[:, 1:num_pts]
        # convert to pd.DataFrame
        difference_df = pd.DataFrame(data = difference, columns = list_IDs)
        # merge ROI series and difference dataframe
        difference_df.insert(0, "Region", ROIs)
        return difference_df
    
    
    def correlate_features(dti_dataframe, list_motor_scores, list_of_ROIs):
        # create empty numpy array to hold (correlations, significance) pairs
        correlation = np.empty((len(dti_dataframe),2))
        
        for i in range(len(dti_dataframe)): 
            # extract current region DTI values into pd.Series 
            x = dti_dataframe.iloc[i,:] 
            # store current ROI (only needed if wanting to print correlations)
            """ Must uncomment curr_ROI if seeking to uncomment print 
                statement at end of loop """ 
            # curr_ROI = x['Region']
            # remove ROI from pd.Series
            x= x.drop(['Region'])
            # calculate (spearman correlation, p val) bw (DTI measure, deltaFM)
            correlation[i,:] = stats.spearmanr(x,list_motor_scores)
            """ If uncommenting print statement then must also uncomment 
                curr_ROI initialization"""
            # print(curr_ROI, 'Spearman Correlation, p value: ', correlation[i,:])
        
        # initialize column names for upcoming dataframe 
        col_names = np.array(['Correlation', 'Significance'])
        # convert correlations array to df with col_names array as column names
        correlation_df = pd.DataFrame(correlation, columns=col_names)
        # insert ROI names into df 
        correlation_df.insert(0, "Region", list_of_ROIs)
        return correlation_df 
    
    
    def check_significance(corr_df):
        # create empty array of same size as input dataframe
        sig_corrs = np.zeros((corr_df.shape),dtype=object)
        
        # iterate across input dataframe 
        for i in range(len(corr_df)): 
            # check current row for statistical significance 
            if corr_df.iloc[i,2] < 0.05: 
                # iff significant then store in array 
                sig_corrs[i,:] = corr_df.iloc[i,:]
        
        # check for empty rows and remove them if exist
        sig_corrs_cleaned = sig_corrs[~np.all(sig_corrs == 0, axis=1)]
        # check that array is not empty (AKA: there are significant values)
        if sig_corrs_cleaned.size > 0: 
            print("Statistically significant correlation identified in: \n",
                  sig_corrs_cleaned[0,0], "\n"
                  "Spearman Correlation: ", sig_corrs_cleaned[0,1], "\n"
                  "p value: ", sig_corrs_cleaned[0,2])
            return sig_corrs_cleaned 
        else: # if no sig values then array must be empty 
            print("No statistically significant correlations found.")


class plot_features: 
    def plot_hemisphere_features(dti_dataframe, list_motor_scores):
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
                axes[i][j].set(ylim=(min(list_motor_scores),
                                     max(list_motor_scores))) # y axis bounds
                
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
        
    
    def plot_cerebellar_feasures(cbl_dataframe, list_motor_scores): 
        # extract ROIs, convert to str array, in Ipsi/Contra/Ipsi/Contra order
        ROIs = np.asarray(cbl_dataframe['Region'],dtype=str).reshape(14,2)
        return 0 


    
