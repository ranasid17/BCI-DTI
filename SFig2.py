#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 18:12:23 2020

@author: Sid
"""
import pandas as pd 
import seaborn as sns 
from scipy import stats 
import matplotlib.pyplot as plt 

countLong ={'Pt ID': [1,2,3,4,6,7,8,9,1,2,3,4,6,7,8,9,1,2,3,4,6,7,8,9],
       'deltaFM': [7, 2, 4.5, 9.5, 7.5, 9, 10.5, 9,7, 2, 4.5, 9.5, 
                   7.5, 9, 10.5, 9,7, 2, 4.5, 9.5,  7.5, 9, 10.5, 9],
       'fiberCount': [19,5,58,8,80,9,121,48,0,0,1,19,23,4,9,5,
                    0,0,0,4,67,1,13,0],
       'Area': ['V','V','V','V','V','V','V','V','VIIB','VIIB','VIIB',
                'VIIB','VIIB','VIIB','VIIB','VIIB','VIIIa',
                'VIIIa','VIIIa','VIIIa','VIIIa','VIIIa','VIIIa',
                'VIIIa']}
countWide = {'Pt ID': [1,2,3,4,6,7,8,9],
            'deltaFM': [7, 2, 4.5, 9.5, 7.5, 9, 10.5, 9],
            'fiberCount_V': [19,5,58,8,80,9,124,48],
            'fiberCount_VIIb': [0,0,1,19,23,4,9,5],
            'fiberCount_VIIIa': [0,0,0,4,67,1,13,0]}
mdLong = {'Pt ID': [1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9],
          'deltaFM': [7, 2, 4.5, 9.5, 5.5, 7.5, 9, 10.5, 9,
                      7, 2, 4.5, 9.5, 5.5, 7.5, 9, 10.5, 9,
                      7, 2, 4.5, 9.5, 5.5, 7.5, 9, 10.5, 9],
          'MeanDiffusivity': [0.462413, 0.45117, 0.50913, 0.484565, 0.435113, 
                              0.477358, 0.501428, 0.493048, 0.505289,0, 0, 
                              0.503343, 0.529992, 0.446674, 0.458095, 0.540579, 
                              0.511424, 0.499599,0, 0, 0.527059, 0.490344, 
                              0.452297, 0.460207, 0.519877, 0.509279, 0.494319],
          'Area': ['V','V','V','V','V','V','V','V','V','VIIb','VIIb','VIIb',
                   'VIIb','VIIb','VIIb','VIIb','VIIb','VIIb','VIIIa','VIIIa',
                   'VIIIa','VIIIa','VIIIa','VIIIa','VIIIa','VIIIa','VIIIa',]
          }
mdWide = {'Pt ID': [1,2,3,4,5,6,7,8,9],
          'deltaFM': [7, 2, 4.5, 9.5, 5.5, 7.5, 9, 10.5, 9],
          'areaV': [0.462413, 0.45117, 0.50913, 0.484565, 0.435113, 0.477358, 
                    0.501428, 0.493048, 0.505289],
          'areaVIIb': [0, 0, 0.503343, 0.529992, 0.446674, 0.458095, 0.540579, 
                       0.511424, 0.499599],
          'areaVIIIa': [0, 0, 0.527059, 0.490344, 0.452297, 0.460207, 0.519877, 
                        0.509279, 0.494319]
          }

# Convert data dictionaries to DataFrames 
countLong = pd.DataFrame(data=countLong)
countWide = pd.DataFrame(data=countWide)
mdLong = pd.DataFrame(data=mdLong)
mdWide = pd.DataFrame(data=mdWide)
# Calculate correlations 
correl_V = stats.spearmanr(mdWide.areaV, mdWide.deltaFM)
correl_VIIb = stats.spearmanr(mdWide.areaVIIb, mdWide.deltaFM)
correl_VIIIa = stats.spearmanr(mdWide.areaVIIIa, mdWide.deltaFM)

# Visualization
fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(25,15))
axes[0].set_title('Tracts: V - Contralesional MC')
axes[1].set_title('Tracts: VIIb - Contralesional MC')
axes[2].set_title('Tracts: VIIIa - Contralesional MC')
a = sns.scatterplot(ax=axes[0], data=mdWide, x='areaV', y='deltaFM',
                    palette=["black"])
a.set_ylabel("Change in UEFM Score")
a.set(ylim=(0, 12))
a.set_xlabel("Initial Tract MD")
b = sns.scatterplot(ax=axes[1], data=mdWide, x='areaVIIb', y='deltaFM')
b.set_xlabel("Initial Tract MD")
c = sns.scatterplot(ax=axes[2], data=mdWide, x='areaVIIIa', y='deltaFM')
c.set_xlabel("Initial Tract MD")
