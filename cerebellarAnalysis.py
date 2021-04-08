#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 17:34:20 2021

@author: Sid
"""

import dtiFunctions as dti

if __name__ == "__main__": 
    
    """ 1) Data initialization """
    # load list of pt IDs, initial UEFM score, and change in UEFM score
    pt_IDs, FM_pre, FM_delta = dti.data_initialization.load_pt_data()
    # load  pre-, post-therapy hemispheric ROI data for FA or MD (user choice)
    pre_ROI, post_ROI, pre_TB, post_TB = dti.data_initialization.load_cerebellar_data()
    # extract list of ROIs (any ROI or TB df will pass as an arg)
    """ Note: The extracted series will contain ipsi-, contra-lesional, and
        vermis ROIs. """
    all_ROIs = dti.preprocessing.extract_ROIs(pre_ROI)
    
    """ 2) Data preprocessing """
    # log transform FAs 
    """ Note: log_transforms() converts df objects to np arrays in 
        order to apply the np.log() function """ 
    pre_ROI = dti.preprocessing.log_transform(pre_ROI, pt_IDs)
    post_ROI = dti.preprocessing.log_transform(post_ROI, pt_IDs)

    pre_TB = dti.preprocessing.log_transform(pre_TB, pt_IDs)
    post_TB = dti.preprocessing.log_transform(post_TB, pt_IDs)

    # convert np arrays back to dataframe objects 
    pre_ROI = dti.preprocessing.reconstruct_df(pre_ROI, pt_IDs, all_ROIs)
    post_ROI = dti.preprocessing.reconstruct_df(post_ROI, pt_IDs, all_ROIs)
    
    pre_TB = dti.preprocessing.reconstruct_df(pre_TB, pt_IDs, all_ROIs)
    post_TB = dti.preprocessing.reconstruct_df(post_TB, pt_IDs, all_ROIs)
    
    """ 3) Data anlysis """
    # calculate change in FA for cerebellar regions
    delta_ROI = dti.feature_analysis.calculate_feature_change(pre_ROI, 
                                                              post_ROI, pt_IDs)
    
    delta_TB = dti.feature_analysis.calculate_feature_change(pre_TB, 
                                                             post_TB, pt_IDs)
    
    # calculate correlation (pretherapy, init FM) for ROI and TB values
    corr_pre_ROI_pre_FM = dti.feature_analysis.correlate_features(pre_ROI, FM_pre, all_ROIs)
    corr_pre_TB_pre_FM = dti.feature_analysis.correlate_features(pre_TB, FM_pre, all_ROIs)
    # calculate correlation (pretherapy FA, delta FM) for ROI and TB values 
    corr_pre_ROI_delta_FM = dti.feature_analysis.correlate_features(pre_ROI, FM_delta, all_ROIs)
    corr_pre_TB_delta_FM = dti.feature_analysis.correlate_features(pre_TB, FM_delta, all_ROIs)
    # calculate correlation (delta FA, delta FM) for ROI and TB values 
    corr_delta_ROI_delta_FM = dti.feature_analysis.correlate_features(delta_ROI, FM_delta, all_ROIs)
    corr_delta_TB_delta_FM = dti.feature_analysis.correlate_features(delta_ROI, FM_delta, all_ROIs)
