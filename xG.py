#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:15:19 2021

@author: mheinone

This files functions is based to calculate xG.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import statistics
from sklearn.cluster import KMeans
from matplotlib.colors import Normalize
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import streamlit as st
import ExtraFunctions as UF

@st.cache(allow_output_mutation=True)
def xG(df, all=0):
    with st.spinner('Calculating xG values...'):
        data = df.copy(deep=True)
        data = data.loc[(data.event.str.contains('Shot'))]
        # drop all another game phases off from dataframe

        if all == 0:
            delete_set_pieces = data[data.event_category.str.contains('Set_play')].index
            data.drop(delete_set_pieces , inplace=True)
    
        shots_model = pd.DataFrame(columns=['Goal','X','Y','Distance','Angle','Degree','Result','SP','Layer','gender'])

        # return empty dataframe
        if len(data) == 0:
            return shots_model

        a = [(65/2) - (7.32/2), 0] # Goal left post
        c = [(65/2) + (7.32/2), 0] # Goal right post
        different_results_of_shots = ['Shoot over', 'Goal', 'Blocked', 'Saved', 'Own Goal']
        for i, row in data.iterrows():
            shots_model.at[i,'X'] = row['xh']
            shots_model.at[i,'Y'] = row['yh']

            # Distance of the center spot
            shots_model.at[i,'C'] = abs(np.sqrt((row['x'] - 105/2)**2 + (row['y'] - 68/2)**2))

            # Distance in metres and shot angle in radians.
            shots_model.at[i,'Distance'] = np.sqrt((shots_model.at[i,'X'] - 65/2)**2 + (shots_model.at[i,'Y'])**2)
            b = [shots_model.at[i,'X'], shots_model.at[i,'Y']] # Shot location
            ang = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
            if ang < 0:
                ang = np.pi + ang
        
            result = different_results_of_shots[UF.resultOfShot(i, df)]
        
            shots_model.at[i,'Angle'] = ang
            shots_model.at[i,'Degree'] = math.degrees(ang)
            shots_model.at[i,'Result'] = result
            shots_model.at[i,'SP'] = row['event']
            shots_model.at[i,'BodyPart'] = row['bp']

            #if ( 'Team' in row.columns ):
            #    shots_model.at[i,'Gendre'] = 'Women' if 'Women' in row['Team'] else 'Men'

            # Was this shot a goal
            is_goal = 1 if result == 'Goal' else 0
            shots_model.at[i,'Goal'] = is_goal

        # Original model is made with Angle and Distance
        #model_variables = ['Angle','Distance','BodyPart','Gendre','SP'] #,'X','C']
        model_variables = ['Angle','Distance'] #,'X','C']
    
        # Fit the model
        #test_model = smf.glm(formula="Goal ~ " + ' + '.join(model_variables), data=shots_model,family=sm.families.Binomial()).fit()
        #print(test_model.summary())
        #print(test_model.params)
        #b = test_model.params

        b = [ 3, -3, 0 ]

        #Add an xG to my dataframe
        xG = shots_model.apply(calculate_xG, axis=1, args=(model_variables, b))

        shots_model = shots_model.assign(xG=xG)
        df = df.assign(xG=xG)
        df = df.assign(Goal=shots_model.Goal)

        return df

@st.cache
def xG_heatmap(shots_model):
    # Two dimensional histogram
    H_Shot = np.histogram2d(shots_model['event_y'], shots_model['event_x'], bins=50, range=[[0, 65], [0, 65]]) # range=[[0, 65], [0, 65]]
    goals_only = shots_model[shots_model['Goal'] == 1]
    H_Goal = np.histogram2d(goals_only['event_y'], goals_only['event_x'], bins=50, range=[[0, 65], [0, 65]])

    #Plot the number of shots from different points
    (fig,ax) = UF.createGoalMouth()
    pos = ax.imshow(H_Shot[0], extent=[0,65,105,0],  aspect='auto', cmap=plt.cm.Reds)
    fig.colorbar(pos, ax = ax)
    ax.set_title('Number of shots')
    plt.xlim((-1,65))
    plt.ylim((-3,35))
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    #Plot the number of GOALS from different points
    (fig,ax) = UF.createGoalMouth()
    pos=ax.imshow(H_Goal[0], extent=[0,65,105,0], aspect='auto',cmap=plt.cm.Reds)
    fig.colorbar(pos, ax=ax)
    ax.set_title('Number of goals')
    plt.xlim((-1,65))
    plt.ylim((-3,35))
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    #Plot the probability of scoring from different points
    (fig,ax) = UF.createGoalMouth()
    pos = ax.imshow(H_Goal[0] / H_Shot[0], extent=[0,65,105,0], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=0.8)
    fig.colorbar(pos, ax=ax)
    ax.set_title('Proportion of shots resulting in a goal')
    plt.xlim((-1,65))
    plt.ylim((-3,35))
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    return()

@st.cache
def calculate_xG(sh, model_variables, b):
    bsum = b[0]
    for i,v in enumerate(model_variables):

        bsum = bsum + b[i + 1] * sh[v]

    if ( sh['SP'] == 'Penalty kick' ):
        xG = 0.75
    else:
        xG = 1 / (1 + np.exp(bsum))

    return xG