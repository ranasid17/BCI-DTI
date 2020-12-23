#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 23:59:57 2020

@author: Sid
"""

import pandas as pd
import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns 

# list of pt IDs
subjID = np.array([201, 202, 205, 209, 210, 211, 212, 226, 235])
# dict containing change in UEFM and initial MDs
data = {'deltaFM': [7, 2, 4.5, 9.5, 5.5, 7.5, 9, 10.5, 9], 
        'logIpsiCC': [-0.20017166, -0.252784018, -0.163620177, -0.208070169,
                      -0.218716504, -0.179308019, -0.207698681, -0.178451101, 
                      -0.131627742], 
        'logContraCC': [-0.254886936, -0.26535456, -0.200582939, -0.139186028, 
                        -0.255906603, -0.210783938, -0.207630726, -0.229499983, 
                        -0.24131515], 
        'logIpsiCBL': [-0.305742376, -0.315745465, -0.315984538, -0.228511, 
                       -0.30109862, -0.286364156, -0.305376377, -0.27252137, 
                       -0.304158604], 
        'logContraCBL': [-0.324798352, -0.320080847, -0.323611875, -0.294715448,
                         -0.314881047, -0.28572474, -0.327417297, -0.269812259, 
                         -0.31539608] }
# convert input to pd.dataFrame 
df = pd.DataFrame(data = data)
# calc Spearman Correl Coeffs (and p values)
correl_IpsiCC = stats.spearmanr(df.logIpsiCC, df.deltaFM)
correl_ContraCC = stats.spearmanr(df.logContraCC, df.deltaFM)
correl_IpsiCBL = stats.spearmanr(df.logIpsiCBL, df.deltaFM)
correl_ContraCBL = stats.spearmanr(df.logContraCBL, df.deltaFM)

# Visualization
# Figure params 
sns.set_context("poster")
fig, axes = plt.subplots(2, 2, sharex=True, sharey = True, figsize=(15,10) ) 
fig.tight_layout()
fig.subplots_adjust(top=0.95)
# Figure titles 
#fig.suptitle('Pre-Therapy Hemispheric MD and Change in UEFM Score', fontsize=16)
axes[0][0].set_title('Ipsilesional Cerebral Cortex')
axes[0][1].set_title('Contralesional Cerebral Cortex')
axes[1][0].set_title('Ipsilesional Cerebellum')
axes[1][1].set_title('Contralesional Cerebellum')
# Graph: 2.a
a = sns.scatterplot(ax = axes[0][0], data=df, x = 'logIpsiCC', y = 'deltaFM',
                    color="black")
a.set(ylim=(0,12)) 
a.set(xlabel=None)
a.set(ylabel="Change in UEFM Score")
a.text(-0.20,2,'$r=0.3263$')
a.text(-0.20,1,'$p=0.3913$')
# Graph: 2.b
b= sns.scatterplot(ax = axes[0][1], data = df, x = 'logContraCC', y = 'deltaFM',
                   color="black")
b.set(ylim=(0,12))
b.set(xlabel=None)
b.set(ylabel=None)
b.text(-0.20,2,'$r=0.4769$')
b.text(-0.20,1,'$p=0.1942$')
# Graph: 2.c
c = sns.regplot(ax = axes[1][0], data = df, x = 'logIpsiCBL', y = 'deltaFM', 
            color = "firebrick", ci = None)
c.set(ylim=(0,12))
c.set(xlabel="Log Initial Mean Diffusivity")
c.set(ylabel="Change in UEFM Score")
c.text(-0.20,2,'$r=0.7782$')
c.text(-0.20,1,'$p=0.0135$')
# Graph: 2.d
d= sns.scatterplot(ax = axes[1][1], data = df, x = 'logContraCBL', y = 'deltaFM',
                   color="black")
d.set(ylim=(0,12))
d.set(xlabel="Log Initial Mean Diffusivity")
d.set(ylabel=None)
d.text(-0.20,2,'$r=0.4519$')
d.text(-0.20,1,'$p=0.2220$')
