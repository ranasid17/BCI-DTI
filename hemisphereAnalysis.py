#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 17:38:56 2021

@author: Sid
"""

import dtiFunctions as dti

if __name__ == "__main__":
    # load list of pt IDs, initial UEFM score, and change in UEFM score
    pt_IDs, FM_pre, FM_delta = dti.data_initialization.load_pt_data()
    # load  pre-, post-therapy hemispheric ROI data for FA or MD (user choice)
    pre, post = dti.data_initialization.load_hemispheric_data()

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
    delta = dti.feature_analysis.calculate_feature_change(pre, post, pt_IDs)
    
    # calculate correlation (pretherapy, init FM) for hemispheres
    corr_pre_feature_pre_FM = dti.feature_analysis.correlate_features(
        pre, FM_pre, ROIs)
    # calculate correlation (pretherapy, delta FM) for hemsipheres
    corr_pre_feature_delta_FM = dti.feature_analysis.correlate_features(
        pre, FM_delta, ROIs)
    # calculate correlation (delta, delta FM) for hemispheres
    corr_delta_feature_delta_FM = dti.feature_analysis.correlate_features(
        delta, FM_delta, ROIs)
    
    # check (pretherapy, init FM) correlations for significance 
    sig_corr_pre_feature_pre_FM = dti.feature_analysis.check_significance(
        corr_pre_feature_pre_FM)
    # check (pretherapy, delta FM) correlations for significance 
    sig_corr_pre_feature_delta_FM = dti.feature_analysis.check_significance(
        corr_pre_feature_delta_FM)
    # check (delta, delta FM) correlations for significance 
    sig_corr_delta_feature_delta_FM = dti.feature_analysis.check_significance(
        corr_delta_feature_delta_FM)
    
    # plot (pretherapy, init FM) for hemispheres
    dti.plot_features.plot_hemisphere_features(pre, FM_pre)
    # plot (pretherapy, delta FM) for hemispheres
    dti.plot_features.plot_hemisphere_features(pre, FM_delta)
    # plot (delta, delta FM) for hemisphers
    dti.plot_features.plot_hemisphere_features(delta, FM_delta)
        