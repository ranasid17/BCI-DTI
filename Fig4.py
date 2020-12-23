#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:15:19 2020

@author: Sid
"""
import numpy as np 
import pandas as pd 
from scipy import stats 
from sklearn import linear_model
import statsmodels.api as sm
import seaborn as sns 
import matplotlib.pyplot as plt 

# list of pt IDs 
subjID = np.array([201, 202, 205, 209, 210, 211, 212, 226, 235])
# dict containing change in UEFM and initial MDs
data = {'deltaFM': [7, 2, 4.5, 9.5, 5.5, 7.5, 9, 10.5, 9], 
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
# Easier to plot first 6 subplots with ROI/Tract data combined 
data2 = {'deltaFM': [7, 2, 4.5, 9.5, 5.5, 7.5, 9, 10.5, 9,
                     7, 2, 4.5, 9.5, 5.5, 7.5, 9, 10.5, 9], 
        'lobV':  [-0.278444103, -0.297216312, -0.3237528, -0.247932336, 
                  -0.296667344, -0.263524615, -0.268707043, -0.26776712, 
                  -0.266517848, -0.271238092, -0.286282709, -0.280722589, 
                  -0.242411922, -0.291233332, -0.254711238, -0.262254898, 
                  -0.254633173, -0.268066514],
        'lobVIIb': [-0.207733013, -0.251167989, -0.228048188, -0.166458949, 
                    -0.231719395, -0.193120968, -0.21517096, -0.107674392, 
                    -0.197467769, -0.28137211, -0.28663884, -0.26947531, 
                    -0.251731326, -0.288067139, -0.262810539, -0.256433013, 
                    -0.245805377, -0.266829227],
        'lobVIIIa': [-0.228150258, -0.299290227, -0.264795657, -0.180483702, 
                     -0.281337245, -0.262727823, -0.239383032, -0.171478017, 
                     -0.278439155, -0.282712377, -0.285168325, -0.263044455, 
                     -0.253216392, -0.286749771, -0.266264415, -0.256280978, 
                     -0.243455172, -0.26296806],
        'Method': ["ROI","ROI","ROI","ROI","ROI","ROI","ROI","ROI","ROI",
                   "Tract","Tract","Tract","Tract","Tract","Tract","Tract",
                   "Tract","Tract"]
        }
# convert input to pd.dataFrame 
df = pd.DataFrame(data = data)
df2 = pd.DataFrame(data = data2)

# Regression model (SKLearn)
xLobV = df[['lobV_ROI','lobV_Tract']] 
xLobVIIb = df[['lobVIIb_ROI','lobVIIb_Tract']]
xLobVIIIa = df[['lobVIIIa_ROI', 'lobVIIIa_Tract']]
y = df['deltaFM']
regr = linear_model.LinearRegression()
regr.fit(xLobVIIIa, y)

# Predictions (from SKLearn MLR)
predLobV=44.338917678892685+(41.62415274*df.lobV_ROI)+(95.8102328*df.lobV_Tract)
predLobVIIb=36.997145695675165+(30.2327451*df.lobVIIb_ROI)+(89.28365029*df.lobVIIb_Tract)
predLobVIIIa=38.2670771224262+(26.46974791*df.lobVIIIa_ROI)+(92.71756782*df.lobVIIIa_Tract)
# Residuals (for each model)
residLobV=df.deltaFM-predLobV
residLobVIIb=df.deltaFM-predLobVIIb
residLobVIIIa=df.deltaFM-predLobVIIIa
# Spearman Correl Coeffs (and p values)
correl_V_ROI = stats.spearmanr(df.lobV_ROI, df.deltaFM)
correl_V_Tract = stats.spearmanr(df.lobV_Tract, df.deltaFM)
correl_VIIb_ROI = stats.spearmanr(df.lobVIIb_ROI, df.deltaFM)
correl_VIIb_Tract = stats.spearmanr(df.lobVIIb_Tract, df.deltaFM)
correl_VIIIa_ROI = stats.spearmanr(df.lobVIIIa_ROI, df.deltaFM)
correl_VIIIa_Tract = stats.spearmanr(df.lobVIIIa_Tract, df.deltaFM)

## Visualization
# Figure params 
fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(30,20)) 
# Figure titles 
sns.set_context("poster")
fig.tight_layout()
#fig.subplots_adjust(top=0.88)
#fig.suptitle('Pre-Therapy Cerebellar MD and Change in UEFM Score', fontsize=30)
axes[0][0].set_title('Correlation')
axes[0][1].set_title('Regression')
axes[0][2].set_title('Predicted Motor Recovery')
sns.color_palette("colorblind", 2)

# Graphs 
# Plot: 4.1.1
a11 = sns.scatterplot(ax=axes[0][0], data=df2, x='lobV', y='deltaFM',
                       hue='Method', style='Method', 
                       palette=["#0072B2", "#009E73"], legend=None)
a11.set_ylabel("Change in UEFM Score")
a11.set(ylim=(0, 12))
a11.text(-0.12,5,'$r=0.850$')
a11.text(-0.12,4,'$p=0.004$')
# Plot: 4.2.1
a21 = sns.scatterplot(ax=axes[1][0], data=df2, x='lobVIIb', y='deltaFM',
                       hue='Method', style='Method', 
                       palette=["#0072B2", "#009E73"], legend=None)
a21.set_ylabel("Change in UEFM Score")
a21.set(ylim=(0, 12))
a21.text(-0.12,5,'$r=0.850$')
a21.text(-0.12,4,'$p=0.004$')
# Plot: 4.3.1
a31 = sns.scatterplot(ax=axes[2][0], data=df2, x='lobVIIIa', y='deltaFM',
                    hue='Method', style='Method', 
                       palette=["#0072B2", "#009E73"])
a31.set_ylabel("Change in UEFM Score")
a31.set_xlabel("Log Initial MD")
a31.set(ylim=(0,12))
a31.legend(loc='lower right')
a31.text(-0.12,5,'$r=0.750$')
a31.text(-0.12,4,'$p=0.020$')

# Plot: 4.1.2
b12 = sns.regplot(ax=axes[0][1], data=df2, x="lobV", y="deltaFM",
                   color='firebrick')
b12.set(ylim=(0,12))
b12.set(xlabel=None)
b12.set(ylabel=None)
b12.text(-0.12,5,'$R^2=0.66$')
b12.text(-0.12,4,'$p=0.041$')
# Plot: 4.2.2
b22 = sns.regplot(ax=axes[1][1], data=df2, x="lobVIIb", y="deltaFM",
                   color='firebrick')
b22.set(ylim=(0,12))
b22.set(xlabel=None)
b22.set(ylabel=None)
b22.text(-0.12,5,'$R^2=0.71$')
b22.text(-0.12,4,'$p=0.041$')
# Plot: 4.3.2
b32 = sns.regplot(ax=axes[2][1], data=df2, x="lobVIIIa", y="deltaFM",
                   color='firebrick')
b32.set(ylim=(0,12))
b32.set_xlabel("Log Initial MD")
b32.set(ylabel=None)
b32.text(-0.12,5,'$R^2=0.65$')
b32.text(-0.12,4,'$p=0.043$')
# Plot: 4.1.3
c13 = sns.scatterplot(ax=axes[0][2], data=df, x='lobV_ROI', y=predLobV,
                      color='firebrick',legend=None)
c13.set(ylim=(0,12))
c13.set(xlabel=None)
c13.set(ylabel=None)
# Plot: 4.2.3
c23 = sns.scatterplot(ax=axes[1][2], data=df, x='lobVIIb_ROI', y=predLobVIIb,
                      hue='AccuracyVIIb', palette=["black", "firebrick"],
                      legend=None)
c23.set(ylim=(0,12))
c23.set(xlabel=None)
c23.set(ylabel=None)
# Plot: 4.3.3
c33 = sns.scatterplot(ax=axes[2][2], data=df, x='lobVIIIa_ROI', 
                      y=predLobVIIIa, hue='Correct Grouping', 
                      palette=["firebrick", "black"])

c33.set(ylim=(0,12))
c33.legend(loc='lower right')
c33.set_xlabel("Log Initial MD")
c33.set(ylabel=None)



