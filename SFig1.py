#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 22:29:42 2020

@author: Sid
"""

import numpy as np 
import pandas as pd 
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
import seaborn as sns
import matplotlib.pyplot as plt 

# Goal: Confirm assumptions of MLR model 
# 1) Linear relationship between IVs and DV 
# 2) IVs are not highly correlated w each other (test via VIFs)
# 3) Residuals variance constant across IVs
# 4) Residuals are normally distributed 

# list of pt IDs 
subjID = np.array([201, 202, 205, 209, 210, 211, 212, 226, 235])
# dict containing change in UEFM and initial MDs
data = {'deltaFM': [7, 1, 4.5, 9.5, 5.5, 7.5, 9, 10.5, 9], 
        'lobV_ROI':  [-0.278444103, -0.297216312, -0.3237528, -0.247932336, 
                      -0.296667344, -0.263524615, -0.268707043, 
                      -0.26776712, -0.266517848],
        'lobV_Tract': [-0.271238092, -0.286282709, -0.280722589, 
                           -0.242411922, -0.291233332, -0.254711238, 
                           -0.262254898, -0.254633173, -0.268066514],
        'lobVIIb_ROI': [-0.207733013, -0.251167989, -0.228048188, -0.166458949, 
                            -0.231719395, -0.193120968, -0.21517096, 
                            -0.107674392, -0.197467769], 
        'lobVIIb_Tract': [-0.28137211, -0.28663884, -0.26947531, -0.251731326, 
                         -0.288067139, -0.262810539, -0.256433013, 
                         -0.245805377, -0.266829227],
        'lobVIIIa_ROI': [-0.228150258, -0.299290227, -0.264795657, -0.180483702, 
                         -0.281337245, -0.262727823, -0.239383032, 
                         -0.171478017, -0.278439155], 
        'lobVIIIa_Tract': [-0.282712377, -0.285168325, -0.263044455, 
                           -0.253216392, -0.286749771, -0.266264415, 
                           -0.256280978, -0.243455172, -0.26296806],
        'AccuracyV': ['T','T','T','T','T','T','T','T','T'],
        'AccuracyVIIb': ['F','T','F','T','T','T','T','T','T',],
        'Correct Grouping': ['T','T','F','T','T','T','T','T','T']
        }
# Building MLR 
class MLR(): 
    # constructor 
    def __init__(self, rawData): 
        self.input = rawData
    def setVars(rawData): 
        # Goal: Return X and Y variables for MLR 
        df = pd.DataFrame(rawData) # convert input to pd.DataFrame
        X = df[['lobV_ROI', 'lobVIIb_ROI', 'lobVIIIa_ROI']] # change this for respective models
        X = sm.add_constant(X) # add Y intercept  
        Y = df[['deltaFM']] 
        return X, Y, df
    def multipleRegression(X,Y): 
        # Goals: 
        #   1) Run MLR on input variables 
        #   2) Return predicted Y 
        model = sm.OLS(Y, X).fit() #
        predictedY = model.predict(X) # predictions of the model
        print_model = model.summary()
        print(print_model)
        return predictedY
    def residuals(df, predictedY): 
        # Goals: 
        #   1) calculate residuals for MLR 
        #   2) build pandas df of predicted change, actual change, residual
        yActual = df.deltaFM # extract change in UEFM from df 
        residuals = yActual - predictedY # calc residuals 
        inputData = pd.DataFrame({'prediction':predictedY, 'actual':yActual})
        return residuals, inputData

resultsV = {'Initial MD': [-0.278444103, -0.297216312, -0.3237528, 
                               -0.247932336, -0.296667344, -0.263524615, 
                               -0.268707043, -0.26776712, -0.266517848,
                               -0.278444103, -0.297216312, -0.3237528, 
                               -0.247932336, -0.296667344, -0.263524615, 
                               -0.268707043, -0.26776712, -0.266517848],
            'UEFM Change': [6.761533069944892, 4.538727518266251, 3.966885080266686, 
                           10.793401578069279,4.087257504033683, 8.965985846300937, 
                           8.02751185096163, 8.796874596024077, 7.561822956132486, 
                           7, 1, 4.5, 9.5, 5.5, 7.5, 9, 10.5, 9],
           'Source': ['Model', 'Model', 'Model', 'Model', 'Model', 'Model', 
               'Model', 'Model', 'Model', 'Actual', 'Actual','Actual','Actual',
               'Actual','Actual','Actual','Actual','Actual']
           }
resultsVIIb = {'Initial MD': [-0.278444103, -0.297216312, -0.3237528, 
                               -0.247932336, -0.296667344, -0.263524615, 
                               -0.268707043, -0.26776712, -0.266517848,
                               -0.278444103, -0.297216312, -0.3237528, 
                               -0.247932336, -0.296667344, -0.263524615, 
                               -0.268707043, -0.26776712, -0.266517848],
               'UEFM Change': [5.5948773933337055, 3.811485955843965, 
                               6.0428836165933735, 9.489143042627777, 
                               4.271946592446797, 7.693884439265474, 
                               7.5966614526903085,11.79545192860094,7.20366557859797,
                               7, 1, 4.5, 9.5, 5.5, 7.5, 9, 10.5, 9],
               'Source': ['Model', 'Model', 'Model', 'Model', 'Model', 'Model', 
                          'Model', 'Model', 'Model', 'Actual', 'Actual','Actual',
                          'Actual','Actual','Actual','Actual','Actual','Actual']
               }
resultsVIIIa = {'Initial MD': [-0.278444103, -0.297216312, -0.3237528, 
                               -0.247932336, -0.296667344, -0.263524615, 
                               -0.268707043, -0.26776712, -0.266517848,
                               -0.278444103, -0.297216312, -0.3237528, 
                               -0.247932336, -0.296667344, -0.263524615, 
                               -0.268707043, -0.26776712, -0.266517848],
                'UEFM Change': [	5.5948773933337055, 3.811485955843965, 
                                6.0428836165933735, 9.489143042627777, 
                                4.271946592446797, 7.693884439265474, 
                                7.5966614526903085, 11.79545192860094, 
                                7.20366557859797, 7, 1, 4.5, 9.5, 5.5, 7.5, 9, 
                                10.5, 9],
                'Source': ['Model', 'Model', 'Model', 'Model', 'Model', 'Model', 
                          'Model', 'Model', 'Model', 'Actual', 'Actual','Actual',
                          'Actual','Actual','Actual','Actual','Actual','Actual']
                }
resultsV = pd.DataFrame(resultsV)
resultsVIIb = pd.DataFrame(resultsVIIb)
resultsVIIIa = pd.DataFrame(resultsVIIIa)

residuals ={'residV': [0.23846693005510833, -3.538727518266251, 0.5331149197333138, 
            -1.293401578069279, 1.4127424959663166, -1.4659858463009368, 
            0.9724881490383694, 1.703125403975923, 1.4381770438675137],
            'residVIIb': [0.9844066809088812, -2.9048267479038294, -2.3691607373089756, 
            -0.512111029909061, 1.2665901710354923, 0.8746510776619232, 
            0.8310803477022493, -0.6555258189893394, 2.4848960568027696],
            'residVIIIa': [1.4051226066662945, -2.811485955843965, -1.5428836165933735, 
            0.010856957372222809, 1.2280534075532028, -0.19388443926547438, 
            1.4033385473096915, -1.29545192860094, 1.7963344214020296],
            'distV': [0,3,0,1,1,1,0,1,1],
            'distVIIb': [0,2,2,0,1,0,0,0,2],
            'distVIIIa': [1,2,1,0,1,0,1,1,1]
            }
residuals = pd.DataFrame(residuals)
# Visualization
fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15,10)) 
# Figure titles 
sns.set_context("poster")
fig.tight_layout()
axes[0].set_title('Lob V')
axes[1].set_title('Lob VIIb')
axes[2].set_title('Lob VIIIa')
sns.set_palette("colorblind")
sns.set_context("poster")
plt.xticks(np.arange(0,8))
#fig.tight_layout()
# Plot 1 
a = sns.scatterplot(ax=axes[0], data=residuals, x=residuals.index,
                    y='residV',hue = 'distV',legend=None,
                    palette="Paired_r")
a.set_xlabel("Pt ID")
a.set_ylabel("Residuals")
a.axhline(y=1, color='black', linestyle='--')
a.axhline(y=-1, color='black', linestyle='--')
a.axhline(y=2, color='black', linestyle=':')
a.axhline(y=-2, color='black', linestyle=':')
a.set(ylim=(-4,3))
# Plot 2 
b = sns.scatterplot(ax=axes[1], data=residuals, x=residuals.index,
                    y='residVIIb',hue='distVIIb',legend=None,
                    palette="Paired_r")
b.set_xlabel("Pt ID")
b.axhline(y=1, color='black', linestyle='--')
b.axhline(y=-1, color='black', linestyle='--')
b.axhline(y=2, color='black', linestyle=':')
b.axhline(y=-2, color='black', linestyle=':')
# Plot 3 
c = sns.scatterplot(ax=axes[2], data=residuals, x=residuals.index,
                    y='residVIIIa',hue='distVIIIa',legend=None,
                    palette="Paired_r")
c.set_xlabel("Pt ID")
c.axhline(y=1, color='black', linestyle='--')
c.axhline(y=-1, color='black', linestyle='--')
c.axhline(y=2, color='black', linestyle=':')
c.axhline(y=-2, color='black', linestyle=':')
