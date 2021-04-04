#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 17:38:56 2021

@author: Sid
"""

import dtiFunctions as dti
import pandas as pd 

if __name__ == "__main__":
    
    # initialize variable to decide which data to load 
    prompt_user = 0 
    # import pretherapy and change in motor score as dataframes
    motor_initial = pd.read_excel(
        r'/Users/Sid/Documents/Leuthardt Lab/DTI/data/FM_initial.xlsx')
    motor_change = pd.read_excel(
        r'/Users/Sid/Documents/Leuthardt Lab/DTI/data/FM_change.xlsx')
    # extract list of patients from either dataframe 
    pt_IDs = dti.preprocessing.extract_pt_IDs(motor_initial)
    # convert motor scores to pandas series 
    FM_pre = dti.preprocessing.extract_motor_scores(motor_initial) 
    FM_delta = dti.preprocessing.extract_motor_scores(motor_change) 
    
    while (prompt_user != 1) and (prompt_user != 2):
        # ask user if want to study FA or MD for hemispheric ROIs
        prompt_user = int(input("Investigate FA or MD for hemisphere ROIs? \n" 
                               "Press 1 for FA or 2 for MD: "))
    # user selects to investigate FA 
    if prompt_user == 1:
        # import pre- and post-therapy FAs for hemisphere ROIs
        pre = pd.read_excel(r'/Users/Sid/Documents/Leuthardt Lab/DTI/data/pretherapy_hemisphere_fa.xlsx') 
        post = pd.read_excel(r'/Users/Sid/Documents/Leuthardt Lab/DTI/data/posttherapy_hemisphere_fa.xlsx') 
    # user selects to investigate MD
    if prompt_user == 2: 
        # import pre- and post-therapy MDs for hemisphere ROIs
        pre = pd.read_excel(r'/Users/Sid/Documents/Leuthardt Lab/DTI/data/pretherapy_hemisphere_md.xlsx')
        post = pd.read_excel(r'/Users/Sid/Documents/Leuthardt Lab/DTI/data/posttherapy_hemisphere_md.xlsx')
    
    # extract list of ROIs
    ROIs = dti.preprocessing.extract_ROIs(pre)
    # log transform FAs 
    """ Note: log_transforms() converts df objects to np arrays in 
        order to apply the np.log() function """ 
    pre = dti.preprocessing.log_transform(pre, pt_IDs)
    post = dti.preprocessing.log_transform(post, pt_IDs)
    # convert np arrays back to dataframe objects 
    pre = dti.preprocessing.reconstruct_df(pre, pt_IDs, ROIs)
    post = dti.preprocessing.reconstruct_df(post, pt_IDs, ROIs)
    # calculate change in FA for hemisphere ROIs
    delta = dti.feature_analysis.calculate_feature_change(pre, post, 
                                                                  pt_IDs)
    
    # calculate correlation (pretherapy FA, init FM) for hemispheres
    corr_pre_feature_pre_FM = dti.feature_analysis.correlate_features(
        pre, FM_pre, ROIs)
    # calculate correlation (pretherapy FA, delta FM) for hemsipheres
    corr_pre_feature_delta_FM = dti.feature_analysis.correlate_features(
        pre, FM_delta, ROIs)
    # calculate correlation (delta FA, delta FM) for hemispheres
    corr_delta_feature_delta_FM = dti.feature_analysis.correlate_features(
        delta, FM_delta, ROIs)
    
    # check (pretherapy FA, init FM) correlations for significance 
    sig_corr_pre_feature_pre_FM = dti.feature_analysis.check_significance(
        corr_pre_feature_pre_FM)
    # check (pretherapy FA, delta FM) correlations for significance 
    sig_corr_pre_feature_delta_FM = dti.feature_analysis.check_significance(
        corr_pre_feature_delta_FM)
    # check (delta FA, delta FM) correlations for significance 
    sig_corr_delta_feature_delta_FM = dti.feature_analysis.check_significance(
        corr_delta_feature_delta_FM)
    
    # plot (pretherapy FA, init FM) for hemispheres
    dti.plot_features.plot_hemisphere_features(pre, FM_pre)
    # plot (pretherapy FA, delta FM) for hemispheres
    dti.plot_features.plot_hemisphere_features(pre, FM_delta)
    # plot (delta FA, delta FM) for hemisphers
    dti.plot_features.plot_hemisphere_features(delta, FM_delta)
        