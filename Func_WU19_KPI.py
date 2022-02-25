#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 01:51:36 2020

@author: mheinone
"""

import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.patches import Arc
import matplotlib.cm as cmap
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4
import os
import json
import re
import math
from io import BytesIO
import pymysql
from soccerplots.radar_chart import Radar
import statistics
from sklearn.cluster import KMeans
from matplotlib.colors import Normalize
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

# Global variables
# Software_action variables
# 1 = add to database
# 2 = make visualisation from all games (you have to select one team)
# 3 = make visualisation from one game
software_action = 2

# Team name is needed if you use action 2
#search_team_games = "FIN -05"
search_team_games = "FIN"

# Match id is needed if you use action 3
matchId = [
    #"220210218",
    "20210218",
    #"20210204",
    #"20210127",
    #"20200309",
    #"20191219",
    #"20201115",
    #"20200227",
    #"20200616",
    #"20200723",
    #"20200822",
    #"20200905",
    ]

# File name is needed if action is 1
file_name = [
    #"20201126 FIN-02 - SalPa-0405.json",
    "20210218 FIN-KuPS.json",
    #220210218 - FIN-05 - HPS.json",
    #"20210218 FIN-KuPS.json",
    #"20210204 Match.json",
    #"20210127 FIN - PK-35.json", 
    #20200309 Iceland-Germany.json",
    #"20191219 Hammarby - HFA.json",
    #"20201115 HFA-TFA.json",
    #"20200227 HFA-KäPa United-06.json",
    #"20200616 HFA-FC Espoo 05.json",
    #"20200723 KJP P15-HFA.json",
    #"20200822 Gnistan-HFA.json",
    #"20200905 KäPa City05-HFA.json",
    ] 

description = {
        'Saved' : 'o',
        'Goal' : 'o',
        'Shoot over' : 'o',
        'Blocked' : 's',
        '' : 'o'
    }

ball_missed = 5
scoring_chance = 20
ball_possession = 15
ball_missed_color = 'white'
scoring_chance_color = 'blue'
ball_possession_color = 'black'
other_color = 'yellow'

def createPitch(length,width, unity, linecolor, pitch=0): # in meters
    # Code by @JPJ_dejong

    """
    creates a plot in which the 'length' is the length of the pitch (goal to goal).
    And 'width' is the width of the pitch (sideline to sideline). 
    Fill in the unity in meters or in yards.

    """
    #Set unity
    if unity == "meters":
        # Set boundaries
        if length >= 120.5 or width >= 75.5:
            return(str("Field dimensions are too big for meters as unity, didn't you mean yards as unity?\
                       Otherwise the maximum length is 120 meters and the maximum width is 75 meters. Please try again"))
        #Run program if unity and boundaries are accepted
        else:
            #Create figure
            fig=plt.figure()
            #fig.set_size_inches(7, 5)
            ax=fig.add_subplot(1,1,1)
           
            #Pitch Outline & Centre Line
            plt.plot([0,0],[0,width], color=linecolor)
            plt.plot([0,length],[width,width], color=linecolor)
            plt.plot([length,length],[width,0], color=linecolor)
            plt.plot([length,0],[0,0], color=linecolor)
            plt.plot([length/2,length/2],[0,width], color=linecolor)
            
            #Left Penalty Area
            plt.plot([16.5 ,16.5],[(width/2 +16.5),(width/2-16.5)],color=linecolor)
            plt.plot([0,16.5],[(width/2 +16.5),(width/2 +16.5)],color=linecolor)
            plt.plot([16.5,0],[(width/2 -16.5),(width/2 -16.5)],color=linecolor)
            
            #Right Penalty Area
            plt.plot([(length-16.5),length],[(width/2 +16.5),(width/2 +16.5)],color=linecolor)
            plt.plot([(length-16.5), (length-16.5)],[(width/2 +16.5),(width/2-16.5)],color=linecolor)
            plt.plot([(length-16.5),length],[(width/2 -16.5),(width/2 -16.5)],color=linecolor)
            
            #Left 5-meters Box
            plt.plot([0,5.5],[(width/2+7.32/2+5.5),(width/2+7.32/2+5.5)],color=linecolor)
            plt.plot([5.5,5.5],[(width/2+7.32/2+5.5),(width/2-7.32/2-5.5)],color=linecolor)
            plt.plot([5.5,0.5],[(width/2-7.32/2-5.5),(width/2-7.32/2-5.5)],color=linecolor)
            
            #Right 5 -eters Box
            plt.plot([length,length-5.5],[(width/2+7.32/2+5.5),(width/2+7.32/2+5.5)],color=linecolor)
            plt.plot([length-5.5,length-5.5],[(width/2+7.32/2+5.5),width/2-7.32/2-5.5],color=linecolor)
            plt.plot([length-5.5,length],[width/2-7.32/2-5.5,width/2-7.32/2-5.5],color=linecolor)
            
            #Prepare Circles
            centreCircle = plt.Circle((length/2,width/2),9.15,color=linecolor,fill=False, zorder=5)
            centreSpot = plt.Circle((length/2,width/2),0.8,color=linecolor, zorder=5)
            leftPenSpot = plt.Circle((11,width/2),0.8,color=linecolor, zorder=5)
            rightPenSpot = plt.Circle((length-11,width/2),0.8,color=linecolor, zorder=5)
            
            # Draw goal
            # First left side goal > Left post and after that right post [x1, x2], [y1, y2]
            plt.plot([0, -1.3], [width/2 + (7.3/2),width/2 + (7.3/2)], alpha=1,color=linecolor, zorder=5)
            plt.plot([0, -1.3], [width/2 - (7.3/2),width/2 - (7.3/2)], alpha=1,color=linecolor, zorder=5)
            plt.plot([-1.3, -1.3], [width/2 - (7.3/2), width/2 + (7.3/2)], alpha=1,color=linecolor, zorder=5)
            
            plt.plot([length, length + 1.3], [width/2 + (7.3/2),width/2 + (7.3/2)], alpha=1,color=linecolor, zorder=5)
            plt.plot([length, length + 1.3], [width/2 - (7.3/2),width/2 - (7.3/2)], alpha=1,color=linecolor, zorder=5)
            plt.plot([length + 1.3, length + 1.3], [width/2 - (7.3/2), width/2 + (7.3/2)], alpha=1,color=linecolor, zorder=5)

            
            #Draw Circles
            ax.add_patch(centreCircle)
            ax.add_patch(centreSpot)
            ax.add_patch(leftPenSpot)
            ax.add_patch(rightPenSpot)
            
            #Prepare Arcs
            leftArc = Arc((11,width/2),height=18.3,width=18.3,angle=0,theta1=308,theta2=52,color=linecolor, zorder=5)
            rightArc = Arc((length-11,width/2),height=18.3,width=18.3,angle=0,theta1=128,theta2=232,color=linecolor, zorder=5)
            
            #Draw Arcs
            ax.add_patch(leftArc)
            ax.add_patch(rightArc)
            
            ## Pitch rectangle
            if ( pitch != 0 ):
                field = plt.Rectangle((-1, -1), length + 2, width + 2,ls='-',color=pitch, zorder=1,alpha=1)
                ax.add_artist(field)

            #Axis titles

    #check unity again
    elif unity == "yards":
        #check boundaries again
        if length <= 95:
            return(str("Didn't you mean meters as unity?"))
        elif length >= 131 or width >= 101:
            return(str("Field dimensions are too big. Maximum length is 130, maximum width is 100"))
        #Run program if unity and boundaries are accepted
        else:
            #Create figure
            fig=plt.figure()
            #fig.set_size_inches(7, 5)
            ax=fig.add_subplot(1,1,1)
            
            #Pitch Outline & Centre Line
            plt.plot([0,0],[0,width], color=linecolor)
            plt.plot([0,length],[width,width], color=linecolor)
            plt.plot([length,length],[width,0], color=linecolor)
            plt.plot([length,0],[0,0], color=linecolor)
            plt.plot([length/2,length/2],[0,width], color=linecolor)
            
            #Left Penalty Area
            plt.plot([18 ,18],[(width/2 +18),(width/2-18)],color=linecolor)
            plt.plot([0,18],[(width/2 +18),(width/2 +18)],color=linecolor)
            plt.plot([18,0],[(width/2 -18),(width/2 -18)],color=linecolor)
            
            #Right Penalty Area
            plt.plot([(length-18),length],[(width/2 +18),(width/2 +18)],color=linecolor)
            plt.plot([(length-18), (length-18)],[(width/2 +18),(width/2-18)],color=linecolor)
            plt.plot([(length-18),length],[(width/2 -18),(width/2 -18)],color=linecolor)
            
            #Left 6-yard Box
            plt.plot([0,6],[(width/2+7.32/2+6),(width/2+7.32/2+6)],color=linecolor)
            plt.plot([6,6],[(width/2+7.32/2+6),(width/2-7.32/2-6)],color=linecolor)
            plt.plot([6,0],[(width/2-7.32/2-6),(width/2-7.32/2-6)],color=linecolor)
            
            #Right 6-yard Box
            plt.plot([length,length-6],[(width/2+7.32/2+6),(width/2+7.32/2+6)],color=linecolor)
            plt.plot([length-6,length-6],[(width/2+7.32/2+6),width/2-7.32/2-6],color=linecolor)
            plt.plot([length-6,length],[(width/2-7.32/2-6),width/2-7.32/2-6],color=linecolor)
            
            #Prepare Circles; 10 yards distance. penalty on 12 yards
            centreCircle = plt.Circle((length/2,width/2),10,color=linecolor,fill=False,zorder=5)
            centreSpot = plt.Circle((length/2,width/2),0.8,color=linecolor,zorder=5)
            leftPenSpot = plt.Circle((12,width/2),0.8,color=linecolor,zorder=5)
            rightPenSpot = plt.Circle((length-12,width/2),0.8,color=linecolor,zorder=5)
            
            # Draw goal
            # First left side goal > Left post and after that right post [x1, x2], [y1, y2]
            plt.plot([0, -1.3], [width/2 + (7.3/2),width/2 + (7.3/2)], alpha=1,color=linecolor, zorder=5)
            plt.plot([0, -1.3], [width/2 - (7.3/2),width/2 - (7.3/2)], alpha=1,color=linecolor, zorder=5)
            plt.plot([-1.3, -1.3], [width/2 - (7.3/2), width/2 + (7.3/2)], alpha=1,color=linecolor, zorder=5)
            
            plt.plot([length, length + 1.3], [width/2 + (7.3/2),width/2 + (7.3/2)], alpha=1,color=linecolor, zorder=5)
            plt.plot([length, length + 1.3], [width/2 - (7.3/2),width/2 - (7.3/2)], alpha=1,color=linecolor, zorder=5)
            plt.plot([length + 1.3, length + 1.3], [width/2 - (7.3/2), width/2 + (7.3/2)], alpha=1,color=linecolor, zorder=5)

            #Draw Circles
            ax.add_patch(centreCircle)
            ax.add_patch(centreSpot)
            ax.add_patch(leftPenSpot)
            ax.add_patch(rightPenSpot)
            
            #Prepare Arcs
            leftArc = Arc((11,width/2),height=20,width=20,angle=0,theta1=312,theta2=48,color=linecolor, zorder=5)
            rightArc = Arc((length-11,width/2),height=20,width=20,angle=0,theta1=130,theta2=230,color=linecolor, zorder=5)
            
            #Draw Arcs
            ax.add_patch(leftArc)
            ax.add_patch(rightArc)

            ## Pitch rectangle
            if ( pitch != 0 ):
                field = plt.Rectangle((-2, -2), length + 4, width + 4,ls='-',color=pitch, zorder=1,alpha=1)
                ax.add_artist(field)
            
                
    #Tidy Axes
    plt.axis('off')
    
    return fig,ax

# Load json file data, and convert it to dataframe
def loadJsonFile(filename, normalize=1):  
    if ( not os.path.isfile(filename) ):
        print("Filename {} is not exist.".format(filename))
        return()
    
    if ( normalize ):
        with open(filename) as train_file:
            dict_train = json.load(train_file)

        train = pd.json_normalize(dict_train)
    else:
        train = pd.read_json(filename, orient='values')

    df = parseJSONFile(train)
    
    return(df)

def parseJSONFile(df):
    parsedFile = pd.DataFrame()
    
    i = 0
    for key, value in df.iterrows():
        for row in value['rows']:
            for highlights in row['highlights']:
                parsedFile.at[i, 'actioncode'] = row['name']
                parsedFile.at[i, 'start'] = highlights['start']
                parsedFile.at[i, 'end'] = highlights['end']
                parsedFile.at[i, 'duration'] = highlights['end'] - highlights['start']
                for event in highlights['events']: 
                    pref = event['name'].split(':')
                    if ( len(pref) > 1 ):
                        pref_key = pref[0]                            
                        pref_value = pref[1]
                        #if ( pref_value.isdigit() ):
                        if ( re.match('^[0-9\.]*$', pref_value) and (('x' in pref_key) or ('y' in pref_key)) ):                            
                            pref_value = float(pref_value)
                        
                        if ( pref_key in ['Open','Penetration','Result','SP'] and pref_key in parsedFile.columns ):
                            if ( not pd.isnull(parsedFile[pref_key].iloc[i]) ):
                                pref_value = pref_value + ":" + parsedFile.at[i, pref_key]
                        
                        parsedFile.at[i, pref_key] = pref_value
                    else:
                        parsedFile.at[i, pref[0]] = True
                i += 1
    
    parsedFile = parsedFile.replace(np.nan, '', regex=True)
    parsedFile = parsedFile.rename(columns={'\u2028' : 'NL'})
    
    home_team = np.unique(parsedFile['Home'])
    home_team = np.delete(home_team, np.where(home_team == ''))[0]
    away_team = np.unique(parsedFile['Away'])
    away_team = np.delete(away_team, np.where(away_team == ''))[0]

    conditions = [
        (parsedFile['actioncode'].str.contains(home_team)),
        (parsedFile['actioncode'].str.contains(away_team))
        ]

    values = [home_team, away_team]
    parsedFile['Team'] = np.select(conditions, values)
    parsedFile['action'] = parsedFile['actioncode'].str.replace('{} - '.format(home_team), '')
    parsedFile['action'] = parsedFile['action'].str.replace('{} - '.format(away_team), '')
    parsedFile = parsedFile.sort_values(by=['start'])
    parsedFile.index = np.arange(0, len(parsedFile))

    return(parsedFile, home_team, away_team)

def howManySecondsToNextOppAction(df, row_index):
    row = df.iloc[row_index]
    team = row['Team']
    start_time = row['start']
    end_time = 0
    time = 0
    last_own_action = ""
    
    if ( ('Shoot' in row['Penetration']) or ('Goal' in row['Result']) ):
        last_own_action = row['Penetration']
        time = 10
    else:
        actions = df.loc[(df['start'] > start_time)]
        for i, action in actions.iterrows():
            if ( action['Team'] == team ):
                end_time = action['start']
                time = end_time - start_time
                last_own_action = action['Penetration']
                if ( last_own_action ):
                    break
            else:
                break
        
    return(time,last_own_action)

def whatColorAction(time, action):
    color = other_color
    if ( ((time <= ball_missed) and (action != 'Shoot')) ):
        color = ball_missed_color
    elif ( ((time <= scoring_chance) and ( ('Shoot' in action) or ('Goal' in action) ) ) ):
        color = scoring_chance_color
    elif ( time >= ball_possession ):
        color = ball_possession_color

    return(color)

def getAllEventsDuration(df, event, team):
    duration = 0.0
    event = df.loc[((df['action'] == event) & (df['Team'] == team))]
    duration = sum(event['duration'])
    return(duration)

def drawIntercept(df, team='', caption='', subcaption='', opp_team='', field_height=105, field_width=68, save_pic=0):
    df_copy = df.copy()
    index_bp = df_copy[ df_copy['action'].str.contains('BP ') ].index
    
    df_copy.drop(index_bp, inplace = True) 
    df_copy.index = np.arange(0, len(df_copy))
    
    (fig,ax) = createPitch(field_height, field_width, 'yards', 'white', '#80B860')

    # Add opponent ball possession chart to figure
    bp_ration = {}
    whole_bp = 0
    for third in ['BP 1-1', 'BP 1-2', 'BP 1-3', 'BP 2-1', 'BP 2-2', 'BP 2-3', 'BP 3-1', 'BP 3-2', 'BP 3-3']:
        bp_ration[third] = getAllEventsDuration(df, third, opp_team)
        whole_bp += bp_ration[third]
    
    # Make chart
    box_height = field_height / 3
    box_width = field_width / 3
    for key in range(3, 0, -1):
        x = box_height * (3 - key)
        box1 = mpatches.Rectangle((x, box_width*2), box_height, box_width, ls='-', color='white', zorder=9, alpha=bp_ration['BP ' + str(key) + '-3']/whole_bp)
        box2 = mpatches.Rectangle((x, box_width), box_height, box_width, ls='-', color='white', zorder=9,alpha=bp_ration['BP ' + str(key) + '-2']/whole_bp)
        box3 = mpatches.Rectangle((x, 0), box_height, box_width, ls='-', color='white', zorder=9,alpha=bp_ration['BP ' + str(key) + '-1']/whole_bp)
                
        ax.add_patch(box1)
        ax.add_patch(box2)
        ax.add_patch(box3)
            
        
    # Add team intercept information plots
    intercept = df_copy.index[((df_copy['action'] == 'Interception') & (df_copy['Team'] == team))].tolist()
    for intercept_index in intercept:
        row = df_copy.iloc[intercept_index]

        if ( len(df_copy) - 1 == intercept_index ):
            plt.scatter(row['x'],row['y'], s=50, edgecolors='white', color='white', alpha=0.7, zorder=8)  
        else:
            (sec, action) = howManySecondsToNextOppAction(df_copy, intercept_index)
            #print("({},{}) - {} - {}".format(row['x'], row['y'], sec, action))
            action_color = whatColorAction(sec, action)
            plt.scatter(row['x'],row['y'], s=50, edgecolors='white', color=action_color, alpha=0.7, zorder=8)  
    
    fig.set_size_inches(field_height/10, field_width/10)
    if ( caption ):
        #fig.text(0.05, 1, "{}\n".format(caption), fontsize=16, fontweight="bold")
        fig.suptitle("{}\n".format(caption), fontsize=16, fontweight="bold")        
    if ( subcaption ):
        fig.text(0.05, 1, "{}".format(subcaption), fontsize=10, fontweight="normal")
        #ax.set_title("{}".format(subcaption), fontsize=10, fontweight="normal")


    # Draw info box
    plt.text(0,-1, 'Ball possession lost in {} seconds'.format(ball_missed), color=ball_missed_color, fontsize=7, va='center', ha='left', zorder=8)
    plt.text(field_height/2-2,-1, 'Ball possession continued for {} to {} seconds'.format(ball_missed, ball_possession), color=other_color, fontsize=7, va='center', ha='right', zorder=8)
    plt.text(field_height/2+2,-1, 'Ball possession continued for at least {} seconds'.format(ball_possession), color=ball_possession_color, fontsize=7, va='center', ha='left', zorder=8)
    plt.text(field_height,-1, 'Scoring chance created in {} seconds'.format(scoring_chance), color=scoring_chance_color, fontsize=7, va='center', ha='right', zorder=8)

    plt.arrow(field_height/5*2, field_width+0.5, field_height/5, 0, head_width=0.7, head_length=0.7, color='black', alpha=0.5, length_includes_head=True, zorder=5)

    plt.tight_layout()

    intercept = make_image_to_buffer()
    if ( save_pic == 1 ):
        file_name = team + "_interceptions.jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()
        
    plt.close()
    return(intercept)

def drawThrowIns(df, team='', caption='', subcaption='', opp_team='', field_height=105, field_width=68, save_pic=0):
    df_copy = df.copy()
    
    (fig,ax) = createPitch(field_height, field_width, 'yards', 'white', '#80B860')
        
    # Add team intercept information plots
    intercept = df_copy.index[((df_copy['SP'].str.contains('Throw in')) & (df_copy['Team'] == team))].tolist()
    for intercept_index in intercept:
        row = df_copy.iloc[intercept_index]
        if ( len(df_copy) - 1 == intercept_index ):
            if ( row['y'] < 10 ):
                row['y'] = 0
            if ( row['y'] > 60 ):
                row['y'] = 68
                
            plt.scatter(row['x'],row['y'], s=50, edgecolors='white', color='white', alpha=0.7, zorder=8)  
            dx = row['x_dest'] - row['x']
            dy = row['y_dest'] - row['y']
            plt.arrow(row['x'], row['y'], dx, dy, head_width=1, head_length=1, color='white', alpha=0.5, length_includes_head=True, zorder=5)
        else:
            (sec, action) = howManySecondsToNextOppAction(df_copy, intercept_index)
            action_color = whatColorAction(sec, action)

            if ( row['y'] < 10 ):
                row['y'] = 0
            if ( row['y'] > 60 ):
                row['y'] = 68
            
            plt.scatter(row['x'],row['y'], s=50, edgecolors='white', color=action_color, alpha=0.7, zorder=8)  
            dx = row['x_dest'] - row['x']
            dy = row['y_dest'] - row['y']
            plt.arrow(row['x'], row['y'], dx, dy, head_width=1, head_length=1, color=action_color, alpha=0.5, length_includes_head=True, zorder=5)
    
    fig.set_size_inches(field_height/10, field_width/10)
    if ( caption ):
        #fig.text(0.05, 1, "{}\n".format(caption), fontsize=16, fontweight="bold")
        fig.suptitle("{}\n".format(caption), fontsize=16, fontweight="bold")        
    if ( subcaption ):
        fig.text(0.05, 1, "{}".format(subcaption), fontsize=10, fontweight="normal")
        #ax.set_title("{}".format(subcaption), fontsize=10, fontweight="normal")


    # Draw info box
    plt.text(0,-1, 'Ball possession lost in {} seconds'.format(ball_missed), color=ball_missed_color, fontsize=7, va='center', ha='left', zorder=8)
    plt.text(field_height/2-2,-1, 'Ball possession continued for {} to {} seconds'.format(ball_missed, ball_possession), color=other_color, fontsize=7, va='center', ha='right', zorder=8)
    plt.text(field_height/2+2,-1, 'Ball possession continued for at least {} seconds'.format(ball_possession), color=ball_possession_color, fontsize=7, va='center', ha='left', zorder=8)
    plt.text(field_height,-1, 'Scoring chance created in {} seconds'.format(scoring_chance), color=scoring_chance_color, fontsize=7, va='center', ha='right', zorder=8)

    plt.arrow(field_height/5*2, field_width+0.5, field_height/5, 0, head_width=0.7, head_length=0.7, color='black', alpha=0.5, length_includes_head=True, zorder=5)

    plt.tight_layout()

    intercept = make_image_to_buffer()
    if ( save_pic == 1 ):
        file_name = team + "_throw_ins.jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()
        
    plt.close()
    return(intercept)

def drawGoalPasses(df, team='', caption='', subcaption='', opp_team='', field_height=105, field_width=68, save_pic=0):
    (fig,ax) = createPitch(field_height, field_width, 'yards', 'white', '#80B860')
   
    height_of_corridor = field_width / 5
    width_of_line = field_height / 6
    x = 0
    y = 0
    count_of_passes = np.zeros((6,6))
    
    intercept = df.index[((df['Result'].str.contains('Goal')) & (df['Team'] == team))].tolist()
    
    # Add team intercept information plots
    for intercept_index in intercept:
        row = df.iloc[intercept_index]
        if ( row['x_pass'] == 0 and row['y_pass'] == 0 ):
            continue
        
        if ( len(df) - 1 == intercept_index ):
            plt.scatter(row['x_pass'],row['y_pass'], s=50, edgecolors='white', color='white', alpha=0.7, zorder=8)  
        else:
            (sec, action) = howManySecondsToNextOppAction(df, intercept_index)
            action_color = whatColorAction(sec, action)
            plt.scatter(row['x_pass'],row['y_pass'], s=50, edgecolors='white', color=action_color, alpha=0.7, zorder=8)  
            column = int(row['x_pass'] / width_of_line)
            corridor = int(row['y_pass'] / height_of_corridor)
            count_of_passes[column][corridor] += 1

    for x in range(6):
        min_x = width_of_line * x
        for y in range(5):
            min_y = height_of_corridor * y

            x_coord = min_x + width_of_line / 2
            y_coord = min_y + height_of_corridor / 2
            plt.scatter(x_coord,y_coord, s=1000, edgecolors='white', color='white', alpha=0.3, zorder=10)
            plt.text(x_coord, y_coord, int(count_of_passes[x][y]), color='white', fontweight='bold', alpha=0.7, fontsize=12, va='center', ha='center', zorder=11)
            y += 1
        x += 1

    
    fig.set_size_inches(field_height/10, field_width/10)
    if ( caption ):
        #fig.text(0.05, 1, "{}\n".format(caption), fontsize=16, fontweight="bold")
        fig.suptitle("{}\n".format(caption), fontsize=16, fontweight="bold")        
    if ( subcaption ):
        fig.text(0.05, 1, "{}".format(subcaption), fontsize=10, fontweight="normal")
        #ax.set_title("{}".format(subcaption), fontsize=10, fontweight="normal")




    # Draw info box
    plt.text(0,-1, 'Ball possession lost in {} seconds'.format(ball_missed), color=ball_missed_color, fontsize=7, va='center', ha='left', zorder=8)
    plt.text(field_height/2-2,-1, 'Ball possession continued for {} to {} seconds'.format(ball_missed, ball_possession), color=other_color, fontsize=7, va='center', ha='right', zorder=8)
    plt.text(field_height/2+2,-1, 'Ball possession continued for at least {} seconds'.format(ball_possession), color=ball_possession_color, fontsize=7, va='center', ha='left', zorder=8)
    plt.text(field_height,-1, 'Scoring chance created in {} seconds'.format(scoring_chance), color=scoring_chance_color, fontsize=7, va='center', ha='right', zorder=8)

    plt.arrow(field_height/5*2, field_width+0.5, field_height/5, 0, head_width=0.7, head_length=0.7, color='black', alpha=0.5, length_includes_head=True, zorder=5)

    plt.tight_layout()

    intercept = make_image_to_buffer()
    if ( save_pic == 1 ):
        file_name = team + "_passes_to_goal.jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()
        
    plt.close()
    return(intercept)

def plot_4_4_2(press='m',mark='o',size=100,fcolor='grey',lcolor='grey',zo=12):
    if (press.lower().startswith("h")):
        goalie_press = -15
        dline_press = -30
        mline_press = -20
        sline_press = -15
    elif (press.lower().startswith("l")):
        goalie_press = 7
        dline_press = 10
        mline_press = 20
        sline_press = 25
    else:
        goalie_press = 0
        dline_press = 0
        mline_press = 0
        sline_press = 0
        
    g = [97+goalie_press, 34]
    lfb = [81.5+dline_press, 54.4]
    lcb = [81.5+dline_press, 40.8]
    rcb = [81.5+dline_press, 27.2]
    rfb = [81.5+dline_press, 13.6]
    lw = [52+mline_press, 54.4]
    lcm = [52+mline_press, 40.8]
    rcm = [52+mline_press, 27.2]
    rw = [52+mline_press, 13.6]
    ls = [29.5+sline_press, 40.8]
    rs = [29.5+sline_press, 27.2]
        
    plt.scatter(g[0],g[1],marker=mark,s=size,color=fcolor,edgecolors=lcolor, zorder=zo) # Goalkeeper
    plt.scatter(lfb[0],lfb[1],marker=mark,s=size,color=fcolor,edgecolors=lcolor, zorder=zo) # Left full back
    plt.scatter(lcb[0],lcb[1],marker=mark,s=size,color=fcolor,edgecolors=lcolor, zorder=zo) # Left center back
    plt.scatter(rcb[0],rcb[1],marker=mark,s=size,color=fcolor,edgecolors=lcolor, zorder=zo) # Right center back
    plt.scatter(rfb[0],rfb[1],marker=mark,s=size,color=fcolor,edgecolors=lcolor, zorder=zo) # Right full back
    plt.scatter(lw[0],lw[1],marker=mark,s=size,color=fcolor,edgecolors=lcolor, zorder=zo) # Left winger
    plt.scatter(lcm[0],lcm[1],marker=mark,s=size,color=fcolor,edgecolors=lcolor, zorder=zo) # Left center midfield
    plt.scatter(rcm[0],rcm[1],marker=mark,s=size,color=fcolor,edgecolors=lcolor, zorder=zo) # Right center midfield
    plt.scatter(rw[0],rw[1],marker=mark,s=size,color=fcolor,edgecolors=lcolor, zorder=zo) # Right winger
    plt.scatter(ls[0],ls[1],marker=mark,s=size,color=fcolor,edgecolors=lcolor, zorder=zo) # Left striker
    plt.scatter(rs[0],rs[1],marker=mark,s=size,color=fcolor,edgecolors=lcolor, zorder=zo) # Right striker

def passesToBox(df, team, caption='', subcaption='', save_pic=0):
    field_height = 65
    field_width = 50
    (fig, ax) = createGoalMouth('white', '#80B860')
    played_to_box = df.index[(df['Penetration'].str.contains('Cross') | df['Penetration'].str.contains('Penetration') | df['Penetration'].str.contains('Key pass')) & (df['Team'] == team)].tolist()
    
    for pbl_index in played_to_box:
        row = df.iloc[pbl_index]
        if ( not row['x_tb'] ):
            continue
        
        if ( len(df) -1 == pbl_index ):
            action_color='white'
        else:
            (sec, action) = howManySecondsToNextOppAction(df, pbl_index)
            action_color = whatColorAction(sec, action)
        
        if ( row['x_tb'] and row['y_tb'] and row['x_tb_dest'] and row['y_tb_dest'] ):
            if ( action_color != scoring_chance_color ):
                action_color = 'white'
            
            plt.scatter(row['x_tb'],row['y_tb'], s=50, edgecolors='white', color=action_color, alpha=0.5, zorder=5)  
            dx = row['x_tb_dest'] - row['x_tb']
            dy = row['y_tb_dest'] - row['y_tb']
            plt.arrow(row['x_tb'], row['y_tb'], dx, dy, head_width=1, head_length=1, color=action_color, alpha=0.5, length_includes_head=True, zorder=5)
    
    fig.set_size_inches(field_height/10, field_width/10)
    if ( caption ):
        fig.suptitle("{}\n".format(caption), fontsize=16, fontweight="bold")
    if ( subcaption ):
        fig.text(0.05, 1, "{}".format(subcaption), fontsize=10, fontweight="normal")
        
    # Draw info box
    plt.text(0,-1, 'Scoring chance did not created in {} seconds'.format(scoring_chance), color=ball_missed_color, fontsize=7, va='center', ha='left', zorder=8)
    plt.text(field_height,-1, 'Scoring chance created in {} seconds'.format(scoring_chance), color=scoring_chance_color, fontsize=7, va='center', ha='right', zorder=8)

    plt.tight_layout()

    pass_to_box = make_image_to_buffer()
    if ( save_pic == 1 ):
        file_name = team + "_passes_to_the_box.jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()

    plt.close()
    return(pass_to_box)

def plotCorners(df, team, caption='', subcaption='', save_pic=0):
    field_height = 65
    field_width = 50
    (fig, ax) = createGoalMouth('white', '#80B860')
    played_to_box = df.index[(df['SP'].str.contains('Corner')) & (df['Team'] == team)].tolist()
    
    for pbl_index in played_to_box:
        row = df.iloc[pbl_index]
        if ( not row['x_tb'] ):
            continue
        
        if ( len(df) -1 == pbl_index ):
            action_color='white'
        else:
            if ( row['Result'] ):
                if ( row['Result'] == 'Shoot over' ):
                    action_color = 'yellow'
                elif ( row['Result'] == 'Blocked' ):
                    action_color = 'black'
                elif ( row['Result'] == 'Saved' ):
                    action_color = 'blue'
                else:
                    action_color = 'white'
            else:
                (sec, action) = howManySecondsToNextOppAction(df, pbl_index)
                action_color = whatColorAction(sec, action)
        
        if ( row['x_tb'] and row['y_tb'] and row['x_tb_dest'] and row['y_tb_dest'] ):
            #if ( action_color != scoring_chance_color ):
            #    action_color = 'white'
            
            plt.scatter(row['x_tb'],row['y_tb'], s=50, edgecolors='white', color=action_color, alpha=0.5, zorder=5)  
            dx = row['x_tb_dest'] - row['x_tb']
            dy = row['y_tb_dest'] - row['y_tb']
            plt.arrow(row['x_tb'], row['y_tb'], dx, dy, head_width=1, head_length=1, color=action_color, alpha=0.5, length_includes_head=True, zorder=5)
    
    fig.set_size_inches(field_height/10, field_width/10)
    if ( caption ):
        fig.suptitle("{}\n".format(caption), fontsize=16, fontweight="bold")
    if ( subcaption ):
        fig.text(0.05, 1, "{}".format(subcaption), fontsize=10, fontweight="normal")
        
    # Draw info box
    plt.text(0,-1, 'Scoring chance did not created'.format(scoring_chance), color='white', fontsize=7, va='center', ha='left', zorder=8)
    plt.text(field_height/3,-1, 'Shoot over'.format(scoring_chance), color='yellow', fontsize=7, va='center', ha='left', zorder=8)
    plt.text(field_height/3*2,-1, 'Blocked'.format(scoring_chance), color='black', fontsize=7, va='center', ha='left', zorder=8)
    plt.text(field_height,-1, 'Scoring chance'.format(scoring_chance), color=scoring_chance_color, fontsize=7, va='center', ha='right', zorder=8)

    plt.tight_layout()
    plt.show()
    
    pass_to_box = make_image_to_buffer()
    if ( save_pic == 1 ):
        file_name = team + "_corners.jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()

    plt.close()
    return(pass_to_box)

def plotAttackFreeKicks(df, team, caption='', subcaption='', save_pic=0):
    field_height = 65
    field_width = 50
    (fig, ax) = createGoalMouth('white', '#80B860')
    played_to_box = df.index[(df['SP'].str.contains('FK')) & (df['Team'] == team)].tolist()
    
    for pbl_index in played_to_box:
        row = df.iloc[pbl_index]
        if ( not row['x_tb'] ):
            continue
        
        if ( len(df) -1 == pbl_index ):
            action_color='white'
        else:
            if ( row['Result'] != '' ):
                if ( row['Result'] == 'Shoot over' ):
                    action_color = 'red'
                elif ( row['Result'] == 'Blocked' ):
                    action_color = 'black'
                elif ( row['Result'] == 'Saved' ):
                    action_color = 'blue'
                else:
                    action_color = 'white'
            else:
                (sec, action) = howManySecondsToNextOppAction(df, pbl_index)
                action_color = whatColorAction(sec, action)
                if ( action_color != scoring_chance_color ):
                    action_color = 'white'

        
        if ( row['x_tb'] and row['y_tb'] ):
            
            plt.scatter(row['x_tb'],row['y_tb'], s=50, edgecolors='white', color=action_color, alpha=0.5, zorder=5)  
            if ( row['x_tb_dest'] and row['y_tb_dest'] ):
                dx = row['x_tb_dest'] - row['x_tb']
                dy = row['y_tb_dest'] - row['y_tb']
                plt.arrow(row['x_tb'], row['y_tb'], dx, dy, head_width=1, head_length=1, color=action_color, alpha=0.5, length_includes_head=True, zorder=5)
    
    fig.set_size_inches(field_height/10, field_width/10)
    if ( caption ):
        fig.suptitle("{}\n".format(caption), fontsize=16, fontweight="bold")
    if ( subcaption ):
        fig.text(0.05, 1, "{}".format(subcaption), fontsize=10, fontweight="normal")
        
    # Draw info box
    plt.text(0,-1, 'Scoring chance did not created'.format(scoring_chance), color='white', fontsize=7, va='center', ha='left', zorder=8)
    plt.text(field_height/3,-1, 'Shoot over'.format(scoring_chance), color='red', fontsize=7, va='center', ha='left', zorder=8)
    plt.text(field_height/3*2,-1, 'Blocked'.format(scoring_chance), color='black', fontsize=7, va='center', ha='left', zorder=8)
    plt.text(field_height,-1, 'Scoring chance'.format(scoring_chance), color=scoring_chance_color, fontsize=7, va='center', ha='right', zorder=8)

    plt.tight_layout()
    plt.show()
    
    pass_to_box = make_image_to_buffer()
    if ( save_pic == 1 ):
        file_name = team + "_fk.jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()

    plt.close()
    return(pass_to_box)

def passesInBoxHeatMap(df, team, caption='', subcaption='', save_pic=0):
    field_height = 65
    field_width = 50
    (fig, ax) = createGoalMouth('white', '#80B860')
    played_to_box = df.index[(df['Penetration'].str.contains('Cross') | df['Penetration'].str.contains('Penetration') | df['Penetration'].str.contains('Key pass')) & (df['Team'] == team)].tolist()
    x = []
    y = []
    result = []
    
    for pbl_index in played_to_box:
        row = df.iloc[pbl_index]
        if ( not row['x_tb'] ):
            continue
            
        if ( row['x_tb'] and row['y_tb'] and row['x_tb_dest'] and row['y_tb_dest'] ):
            (sec, action) = howManySecondsToNextOppAction(df, pbl_index)
            action_color = whatColorAction(sec, action)

            if ( action_color == scoring_chance_color ):
                result.append(1)
            else:
                result.append(1)
                
            x.append(row['x_tb_dest'])
            y.append(row['y_tb_dest'])
  
    # Set threspass value for several hexbins
    nbins = 15
    pos = ax.hexbin(x, y, gridsize=nbins, C=result, ec='black', cmap=plt.cm.Greys, alpha=0.7, linewidths=0.2, reduce_C_function=np.sum, zorder=10)
    fig.colorbar(pos, orientation="horizontal", ax=ax, pad=0.01, aspect=50)
    
    plt.xlim((-1,field_height + 1))
    plt.ylim((-3,field_width + 10))
 
    fig.set_size_inches((field_height+5)/10, (field_width + 10) /10)
    if ( caption ):
        fig.suptitle("{}\n".format(caption), fontsize=16, fontweight="bold")
    if ( subcaption ):
        fig.text(0.05, 1, "{}".format(subcaption), fontsize=10, fontweight="normal")
        
    plt.tight_layout()
    box_heat = make_image_to_buffer()
    
    if ( save_pic == 1 ):
        file_name = team + "_passes_in_the_box_hm.jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()

    plt.close()
    return(box_heat)

def passesToBoxHeatMap(df, team, caption='', subcaption='', save_pic=0):
    field_height = 65
    field_width = 50
    (fig, ax) = createGoalMouth('white', '#80B860')
    played_to_box = df.index[(df['Penetration'].str.contains('Cross') | df['Penetration'].str.contains('Penetration') | df['Penetration'].str.contains('Key pass')) & (df['Team'] == team)].tolist()
    x = []
    y = []
    result = []
    
    for pbl_index in played_to_box:
        row = df.iloc[pbl_index]
        if ( not row['x_tb'] ):
            continue
            
        if ( row['x_tb'] and row['y_tb'] and row['x_tb_dest'] and row['y_tb_dest'] ):
            (sec, action) = howManySecondsToNextOppAction(df, pbl_index)
            action_color = whatColorAction(sec, action)

            if ( action_color == scoring_chance_color ):
                result.append(1)
            else:
                result.append(1)
                
            x.append(row['x_tb'])
            y.append(row['y_tb'])
  
    # Set threspass value for several hexbins
    nbins = 15
    pos = ax.hexbin(x, y, gridsize=nbins, C=result, ec='black', cmap=plt.cm.Greys, alpha=0.7, linewidths=0.2, reduce_C_function=np.sum, zorder=10)
    fig.colorbar(pos, orientation="horizontal", ax=ax, pad=0.01, aspect=50)

    plt.xlim((-1,field_height + 1))
    plt.ylim((-3,field_width + 10))
 
    fig.set_size_inches((field_height+5)/10, (field_width + 10) /10)
    if ( caption ):
        fig.suptitle("{}\n".format(caption), fontsize=16, fontweight="bold")
    if ( subcaption ):
        fig.text(0.05, 1, "{}".format(subcaption), fontsize=10, fontweight="normal")
        
    plt.tight_layout()

    image_field = make_image_to_buffer()
    if ( save_pic == 1 ):
        file_name = team + "_passes_to_the_box_hm.jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()

    plt.close()
    return(image_field)

def penetratingToBox(df, team, caption='', subcaption='', save_pic=0):
    width = 65
    height = 50
    (fig, ax) = createGoalMouth('white', '#80B860')
    played_to_box = df.index[(df['Penetration'].str.contains('1v1') | df['Penetration'].str.contains('Cross') | df['Penetration'].str.contains('Penetration') | df['Penetration'].str.contains('Key pass')) & (df['Team'] == team)].tolist()
    x = [] # 0,0,0,0,65,65,65,65,
    y = [] # 0,0,0,0,50,50,50,50,
    
    for pbl_index in played_to_box:
        row = df.iloc[pbl_index]
        if ( row['x_tb'] and row['y_tb'] ):
            x.append(row['x_tb'])
            y.append(row['y_tb'])
        else:
            continue
    
    # Set threspass value for several hexbins
    penetrated = np.histogram2d(y, x, bins=25, range=[[0, width], [0, width]])
    pos = ax.imshow(penetrated[0], extent=[0,width,0,height], origin='lower', aspect='auto', cmap=plt.cm.Blues, alpha=0.6, zorder=9)

    fig.colorbar(pos, orientation="horizontal", ax=ax, pad=0.01, aspect=50)
    plt.gca().set_aspect('equal', adjustable='box')
        
    plt.xlim((-1,width + 1))
    plt.ylim((-3,height + 10))
 
    fig.set_size_inches((width+10)/10, (height + 20) /10)
    if ( caption ):
        fig.suptitle("{}\n".format(caption), fontsize=16, fontweight="bold")
    if ( subcaption ):
        fig.text(0.05, 1, "{}".format(subcaption), fontsize=10, fontweight="normal")
        
    plt.tight_layout()

    image_field = make_image_to_buffer()
    if ( save_pic == 1 ):
        file_name = team + "_passes_to_the_box_from_areas.jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()

    plt.close()
    return(image_field)    

def playedBetweenLines(df, team='', caption='', subcaption='', field_height=105, field_width=68, press_line='', key_pass=0, save_pic=0):
    (fig,ax) = createPitch(field_height,field_width,'yards','white','#80B860')
    if ( press_line ):
        plot_4_4_2(press_line,'o',600,'grey','black')
    
    if ( key_pass == 1 ):
        played_between_lines = df.index[(df['Penetration'].str.contains('Key pass')) & (df['Team'] == team)].tolist()
    else:
        played_between_lines = df.index[(df['Open'].str.contains('Pocket') | df['Open'].str.contains('10-place')) & (df['Team'] == team)].tolist()
    
    #print(df[['start','actioncode','x','y','Open']].iloc[played_between_lines])
    
    for pbl_index in played_between_lines:
        row = df.iloc[pbl_index]
        if ( len(df) -1 == pbl_index ):
            action_color='white'
        else:
            (sec, action) = howManySecondsToNextOppAction(df, pbl_index)
            action_color = whatColorAction(sec, action)
        
        if ( key_pass == 1 ):
            if ( row['x'] and row['y'] and row['x_dest'] and row['y_dest'] ):
                # print("({},{}) - ({},{})".format(row['x'], row['y'], row['x_dest'], row['y_dest']))
                plt.scatter(row['x'],row['y'], s=50, edgecolors='white', color=action_color, alpha=0.5, zorder=5)  
                dx = row['x_dest'] - row['x']
                dy = row['y_dest'] - row['y']
                plt.arrow(row['x'], row['y'], dx, dy, head_width=1, head_length=1, color=action_color, alpha=0.5, length_includes_head=True, zorder=5)
        else:
            if ( row['Press'] == press_line ):
                if ( row['x'] and row['y'] and row['x_dest'] and row['y_dest'] ):
                    plt.scatter(row['x'],row['y'], s=50, edgecolors='white', color=action_color, alpha=0.5, zorder=5)  
                    dx = row['x_dest'] - row['x']
                    dy = row['y_dest'] - row['y']
                    plt.arrow(row['x'], row['y'], dx, dy, head_width=1, head_length=1, color=action_color, alpha=0.5, length_includes_head=True, zorder=5)
            else:
                continue
    
    fig.set_size_inches(field_height/10, field_width/10)
    if ( caption ):
        #fig.text(0.05, 1, "{}\n".format(caption), fontsize=16, fontweight="bold")
        fig.suptitle("{}\n".format(caption), fontsize=16, fontweight="bold")
    if ( subcaption ):
        fig.text(0.05, 1, "{}".format(subcaption), fontsize=10, fontweight="normal")
        #ax.set_title("{}".format(subcaption), fontsize=10, fontweight="normal")
        
    # Draw info box
    plt.text(0,-1, 'Ball possession lost in {} seconds'.format(ball_missed), color=ball_missed_color, fontsize=7, va='center', ha='left', zorder=8)
    plt.text(field_height/2-2,-1, 'Ball possession continued for {} to {} seconds'.format(ball_missed, ball_possession), color=other_color, fontsize=7, va='center', ha='right', zorder=8)
    plt.text(field_height/2+2,-1, 'Ball possession continued for at least {} seconds'.format(ball_possession), color=ball_possession_color, fontsize=7, va='center', ha='left', zorder=8)
    plt.text(field_height,-1, 'Scoring chance created in {} seconds'.format(scoring_chance), color=scoring_chance_color, fontsize=7, va='center', ha='right', zorder=8)

    plt.arrow(field_height/5*2, field_width+0.5, field_height/5, 0, head_width=0.7, head_length=0.7, color='black', alpha=0.5, length_includes_head=True, zorder=5)

    plt.tight_layout()
    image_field = make_image_to_buffer()
    
    if ( save_pic == 1 ):
        file = team.replace("/","")
        file_name = file + "_played_between_lines_" + press_line + ".jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()
        
    plt.close()
    return(image_field)

def buildUpPhase(df, team='', caption='', subcaption='', field_height=105, field_width=68, press_line='', key_pass=0, save_pic=0):
    (fig,ax) = createPitch(field_height,field_width,'yards','white','#80B860')
    if ( press_line ):
        plot_4_4_2(press_line,'o',600,'grey','black')
    
    if ( key_pass == 1 ):
        played_between_lines = df.index[(df['Penetration'].str.contains('Key pass')) & (df['Team'] == team)].tolist()
    else:
        played_between_lines = df.index[(df['Open'].str.contains('Pocket') | df['Open'].str.contains('10-place') | df['Open'].str.contains('6-place')) & (df['Team'] == team)].tolist()
    
    for pbl_index in played_between_lines:
        row = df.iloc[pbl_index]
        if ( len(df) -1 == pbl_index ):
            action_color='white'
        else:
            (sec, action) = howManySecondsToNextOppAction(df, pbl_index)
            action_color = whatColorAction(sec, action)
        
        if ( key_pass == 1 ):
            if ( row['x'] and row['y'] and row['x_dest'] and row['y_dest'] ):
                # print("({},{}) - ({},{})".format(row['x'], row['y'], row['x_dest'], row['y_dest']))
                plt.scatter(row['x'],row['y'], s=50, edgecolors='white', color=action_color, alpha=0.5, zorder=5)  
                dx = row['x_dest'] - row['x']
                dy = row['y_dest'] - row['y']
                plt.arrow(row['x'], row['y'], dx, dy, head_width=1, head_length=1, color=action_color, alpha=0.5, length_includes_head=True, zorder=5)
        else:
            if ( row['Press'] == press_line ):
                if ( row['x'] and row['y'] and row['x_dest'] and row['y_dest'] ):
                    plt.scatter(row['x'],row['y'], s=50, edgecolors='white', color=action_color, alpha=0.5, zorder=5)  
                    dx = row['x_dest'] - row['x']
                    dy = row['y_dest'] - row['y']
                    plt.arrow(row['x'], row['y'], dx, dy, head_width=1, head_length=1, color=action_color, alpha=0.5, length_includes_head=True, zorder=5)
            else:
                continue
    
    fig.set_size_inches(field_height/10, field_width/10)
    if ( caption ):
        #fig.text(0.05, 1, "{}\n".format(caption), fontsize=16, fontweight="bold")
        fig.suptitle("{}\n".format(caption), fontsize=16, fontweight="bold")
    if ( subcaption ):
        fig.text(0.05, 1, "{}".format(subcaption), fontsize=10, fontweight="normal")
        #ax.set_title("{}".format(subcaption), fontsize=10, fontweight="normal")
        
    # Draw info box
    plt.text(0,-1, 'Ball possession lost in {} seconds'.format(ball_missed), color=ball_missed_color, fontsize=7, va='center', ha='left', zorder=8)
    plt.text(field_height/2-2,-1, 'Ball possession continued for {} to {} seconds'.format(ball_missed, ball_possession), color=other_color, fontsize=7, va='center', ha='right', zorder=8)
    plt.text(field_height/2+2,-1, 'Ball possession continued for at least {} seconds'.format(ball_possession), color=ball_possession_color, fontsize=7, va='center', ha='left', zorder=8)
    plt.text(field_height,-1, 'Scoring chance created in {} seconds'.format(scoring_chance), color=scoring_chance_color, fontsize=7, va='center', ha='right', zorder=8)

    plt.arrow(field_height/5*2, field_width+0.5, field_height/5, 0, head_width=0.7, head_length=0.7, color='black', alpha=0.5, length_includes_head=True, zorder=5)

    plt.tight_layout()
    image_field = make_image_to_buffer()
    
    if ( save_pic == 1 ):
        file = team.replace("/","")
        file_name = file + "_buil_up_phase_" + press_line + ".jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()
        
    plt.close()
    return(image_field)

def played6area(df, team='', caption='', subcaption='', field_height=105, field_width=68, press_line='', key_pass=0, save_pic=0):
    (fig,ax) = createPitch(field_height,field_width,'yards','white','#80B860')
    if ( press_line ):
        plot_4_4_2(press_line,'o',600,'grey','black')
    
    played_between_lines = df.index[(df['Open'].str.contains('6-place')) & (df['Team'] == team)].tolist()
    
    for pbl_index in played_between_lines:
        row = df.iloc[pbl_index]
        if ( len(df) -1 == pbl_index ):
            action_color='white'
        else:
            (sec, action) = howManySecondsToNextOppAction(df, pbl_index)
            action_color = whatColorAction(sec, action)
        
        if ( key_pass == 1 ):
            if ( row['x'] and row['y'] and row['x_dest'] and row['y_dest'] ):
                # print("({},{}) - ({},{})".format(row['x'], row['y'], row['x_dest'], row['y_dest']))
                plt.scatter(row['x'],row['y'], s=50, edgecolors='white', color=action_color, alpha=0.5, zorder=5)  
                dx = row['x_dest'] - row['x']
                dy = row['y_dest'] - row['y']
                plt.arrow(row['x'], row['y'], dx, dy, head_width=1, head_length=1, color=action_color, alpha=0.5, length_includes_head=True, zorder=5)
        else:
            if ( row['Press'] == press_line ):
                if ( row['x'] and row['y'] and row['x_dest'] and row['y_dest'] ):
                    plt.scatter(row['x'],row['y'], s=50, edgecolors='white', color=action_color, alpha=0.5, zorder=5)  
                    dx = row['x_dest'] - row['x']
                    dy = row['y_dest'] - row['y']
                    plt.arrow(row['x'], row['y'], dx, dy, head_width=1, head_length=1, color=action_color, alpha=0.5, length_includes_head=True, zorder=5)
            else:
                continue
    
    fig.set_size_inches(field_height/10, field_width/10)
    if ( caption ):
        #fig.text(0.05, 1, "{}\n".format(caption), fontsize=16, fontweight="bold")
        fig.suptitle("{}\n".format(caption), fontsize=16, fontweight="bold")
    if ( subcaption ):
        fig.text(0.05, 1, "{}".format(subcaption), fontsize=10, fontweight="normal")
        #ax.set_title("{}".format(subcaption), fontsize=10, fontweight="normal")
        
    # Draw info box
    plt.text(0,-1, 'Ball possession lost in {} seconds'.format(ball_missed), color=ball_missed_color, fontsize=7, va='center', ha='left', zorder=8)
    plt.text(field_height/2-2,-1, 'Ball possession continued for {} to {} seconds'.format(ball_missed, ball_possession), color=other_color, fontsize=7, va='center', ha='right', zorder=8)
    plt.text(field_height/2+2,-1, 'Ball possession continued for at least {} seconds'.format(ball_possession), color=ball_possession_color, fontsize=7, va='center', ha='left', zorder=8)
    plt.text(field_height,-1, 'Scoring chance created in {} seconds'.format(scoring_chance), color=scoring_chance_color, fontsize=7, va='center', ha='right', zorder=8)

    plt.arrow(field_height/5*2, field_width+0.5, field_height/5, 0, head_width=0.7, head_length=0.7, color='black', alpha=0.5, length_includes_head=True, zorder=5)

    plt.tight_layout()
    image_field = make_image_to_buffer()
    
    if ( save_pic == 1 ):
        file = team.replace("/","")
        file_name = file + "_6-area_" + press_line + ".jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()
        
    plt.close()
    return(image_field)

def xG(data, team, save_pic=0):
    if ( team != '' ):
        shots = data.loc[((data['Penetration'].str.contains('Shoot') | (data['Result'] == 'Goal')) & (data['Team'] == team))]
    else:
        shots = data.loc[((data['Penetration'].str.contains('Shoot') | (data['Result'] == 'Goal')))]
    
    # Delete set pieces
    """
    for set_piece in ['corner','kick']:
        delete_set_pieces = shots[shots['att_type'].str.contains(set_piece)].index
        shots.drop(delete_set_pieces , inplace=True)
    """
    shots_model = pd.DataFrame(columns=['Goal','X','Y'])
    ration_x = 65 / 68
    ration_y = 50 / 105
    a = [(65/2) - (7.32/2), 0] # Goal left post
    c = [(65/2) + (7.32/2), 0] # Goal right post
    
    # Only open play shots and goals
    for i,shot in shots.iterrows():
        if ( not shot['x'] or not shot['y'] ):
            continue
        
        shots_model.at[i,'X'] = shot['y'] * ration_x
        shots_model.at[i,'Y'] = 105 - shot['x'] # 105 / 2 - shot['x'] * ration_y # + 3.2 # 6
        shots_model.at[i,'C'] = abs(np.sqrt((shot['x'] - 105/2)**2 + (shot['y'] - 68/2)**2)) # Distance of the center
        
        #if ( 'Goal' in shot['Result'] ):
        #    print("{} ({}): {} > {} ({})".format(team, shot['SP'], shot['x'], 105 / 2 - shot['x'] * ration_y + 3, shot['Result']))

        # att_type.append(shot['att_type'])

        # Distance in metres and shot angle in radians.
        shots_model.at[i,'Distance'] = np.sqrt((shots_model.at[i,'X'] - 65/2)**2 + (shots_model.at[i,'Y'])**2)
        b = [shots_model.at[i,'X'], shots_model.at[i,'Y']] # Shot location
        ang = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
        if ( ang < 0 ):
            ang = np.pi + ang
        
        
        results = shot['Result'].split(':')
        result = results[0]
        for r in ['Saved', 'Goal', 'Shoot over', 'Blocked']:
            if ( r in results ):
                result = r
            
        shots_model.at[i,'Angle'] = ang
        shots_model.at[i,'Degree'] = math.degrees(ang)
        shots_model.at[i,'Result'] = result
        shots_model.at[i,'SP'] = shot['SP']
        shots_model.at[i,'Team'] = shot['Team']
        shots_model.at[i,'Home'] = 1 if shot['Team'] == shot['Home'] else 0
        shots_model.at[i,'Opponent'] = shot['Away'] if shot['Team'] == shot['Home'] else shot['Home']
        
        # Was this shot a goal
        is_goal = 1 if 'Goal' in shot['Result'] else 0
        shots_model.at[i,'Goal'] = is_goal
        # print("Is it goal {} > {}?".format(shot['actioncode'],is_goal))

    model_variables = ['Angle','Distance'] #,'X','C']
    model = ' + '.join(model_variables)
    #b = { 'Intercept' : 3.571782, 'Angle' : -1.821250, 'Distance' : -0.064344 }
    # b = [ 3.571782, -1.821250, -0.064344 ]
    b = [ 3, -3, 0 ]
    
    #Add an xG to my dataframe
    xG = shots_model.apply(calculate_xG, axis=1, args=(model_variables, b)) 
    shots_model = shots_model.assign(xG = xG)
    
    if ( save_pic == 1 ):
        xG_heatmap(shots_model, team)
 
    return(shots_model)

def xG_heatmap(shots_model, team):
    # Two dimensional histogram
    H_Shot = np.histogram2d(shots_model['Y'], shots_model['X'], bins=50, range=[[0, 65], [0, 65]]) # range=[[0, 65], [0, 65]]
    goals_only = shots_model[shots_model['Goal'] == 1]
    H_Goal = np.histogram2d(goals_only['Y'], goals_only['X'], bins=50, range=[[0, 65], [0, 65]])

    #Plot the number of shots from different points
    (fig,ax) = createGoalMouth()
    pos = ax.imshow(H_Shot[0], extent=[0,65,105,0],  aspect='auto', cmap=plt.cm.Reds)
    fig.colorbar(pos, ax = ax)
    ax.set_title('Number of shots')
    plt.xlim((-1,65))
    plt.ylim((-3,35))
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    file = team.replace("/","")
    file_name = file + "_number_of_shots.jpg"
    fig.savefig(file_name, dpi=200)
    plt.show()

    #Plot the number of GOALS from different points
    (fig,ax) = createGoalMouth()
    pos=ax.imshow(H_Goal[0], extent=[0,65,105,0], aspect='auto',cmap=plt.cm.Reds)
    fig.colorbar(pos, ax=ax)
    ax.set_title('Number of goals')
    plt.xlim((-1,65))
    plt.ylim((-3,35))
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    file_name = file + "number_of_GOALS.jpg"
    fig.savefig(file_name, dpi=200)
    plt.show()

    #Plot the probability of scoring from different points
    (fig,ax) = createGoalMouth()
    pos = ax.imshow(H_Goal[0] / H_Shot[0], extent=[0,65,105,0], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=0.8)
    fig.colorbar(pos, ax=ax)
    ax.set_title('Proportion of shots resulting in a goal')
    plt.xlim((-1,65))
    plt.ylim((-3,35))
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')    
    file_name = file + "_probability_of_scoring.jpg"
    fig.savefig(file_name, dpi=200)
    plt.show()

    return()

def calculate_xG(sh, model_variables, b):
    bsum = b[0]
    for i,v in enumerate(model_variables):
        
        bsum = bsum + b[i + 1] * sh[v]
    
    if ( sh['SP'] == 'Penalty kick' ):
        xG = 0.75
    else:
        xG = 1 / (1 + np.exp(bsum)) 
        
    return xG  

def createGoalMouth(linecolor='black',pitch=''):
    #Adopted from FC Python
    #Create figure
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)

    #Pitch Outline & Centre Line
    plt.plot([0,65],[0,0], color=linecolor,zorder=2)
    plt.plot([65,65],[50,0], color=linecolor,zorder=2)
    plt.plot([0,0],[50,0], color=linecolor,zorder=2)
    
    #Left Penalty Area
    plt.plot([12.5,52.5],[16.5,16.5],color=linecolor,zorder=2)
    plt.plot([52.5,52.5],[16.5,0],color=linecolor,zorder=2)
    plt.plot([12.5,12.5],[0,16.5],color=linecolor,zorder=2)
    
    #Left 6-yard Box
    plt.plot([41.5,41.5],[5.5,0],color=linecolor,zorder=2)
    plt.plot([23.5,41.5],[5.5,5.5],color=linecolor,zorder=2)
    plt.plot([23.5,23.5],[0,5.5],color=linecolor,zorder=2)
    
    #Goal
    plt.plot([41.5-5.34,41.5-5.34],[-2,0],color=linecolor,zorder=2)
    plt.plot([23.5+5.34,41.5-5.34],[-2,-2],color=linecolor,zorder=2)
    plt.plot([23.5+5.34,23.5+5.34],[0,-2],color=linecolor,zorder=2)
    
    #Prepare Circles
    leftPenSpot = plt.Circle((65/2,11),0.8,color=linecolor,zorder=2)
    
    #Draw Circles
    ax.add_patch(leftPenSpot)
    
    #Prepare Arcs
    leftArc = Arc((32.5,11),height=18.3,width=18.3,angle=0,theta1=38,theta2=142,color=linecolor,zorder=2)
    
    #Draw Arcs
    ax.add_patch(leftArc)
    
    ## Pitch rectangle
    if ( pitch != '' ):
        field = plt.Rectangle((-1, -2), 67, 53,ls='-',color=pitch, zorder=1,alpha=1)
        ax.add_artist(field)

    #Tidy Axes
    plt.axis('off')
    
    return fig,ax

def plotShotsAndxG(calculated_xg, description, caption, subcaption, team, save_pic=0):
    (fig, ax) = createGoalMouth('white', '#80B860')

    for i, row in calculated_xg.iterrows():
        if ( row['Result'] == 'Shoot over' ):
            color = 'red'
            text_color = 'white'
        elif ( row['Result'] == 'Goal' ):
            color = 'yellow'
            text_color = 'black'
        else:
            color = 'black'
            text_color = 'white'
      
        ax.scatter(row['X'], row['Y'], edgecolors='black', s=450, color=color, marker=description[row['Result']], label=row['Result'], alpha=0.6, zorder=5)
        ax.text(row['X'], row['Y'], round(row['xG'],2), size=10, verticalalignment='center', color=text_color, horizontalalignment='center', zorder=7)

    ax.text(1.2, 49.5, "Sum of xG: {}".format(round(sum(calculated_xg['xG']), 2)), size=12, verticalalignment='center', horizontalalignment='left')
    ax.text(1.2, 48, "xG/shot: {}".format(round(np.mean(calculated_xg['xG']), 2)), size=12, verticalalignment='center', horizontalalignment='left')
    i = 1
    for key in description:
        if ( key == '' ):
            continue 
        
        key_text = key
        if ( key == 'Shoot over' ):
            color = 'red'
            key_text = 'Shot over/wide'
        elif ( key == 'Goal' ):
            color = 'yellow'
            key_text = 'Goal'
        else:
            color = 'black'

        ax.scatter(2, 48 - 2*i, edgecolors='black', s=200, color=color, marker=description[key], alpha=0.6, zorder=5)
        ax.text(4, 48 - 2*i, key_text, size=12, verticalalignment='center', horizontalalignment='left')
        i += 1


    fig.set_size_inches(10, 7)
    plt.tight_layout()
   
    if ( caption ):
        fig.suptitle("{}\n".format(caption), fontsize=16, fontweight="bold")
    if ( subcaption ):
        fig.text(0.05, 1, "{}".format(subcaption), fontsize=10, fontweight="normal")

    image_field = make_image_to_buffer()
    
    if ( save_pic == 1 ):
        file = team.replace("/","")
        file_name = file + "_shots-xG.jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()
    
    plt.close()
    return(image_field)

def plotShotsHex(calculated_xg, description, caption, subcaption, team, save_pic=0): 
    bg_color = '#171716'
    multiply_size = 400
    color = []
    for i, row in calculated_xg.iterrows():
        if ( row['Goal'] == 1 ):
            color.append('#4925e8')
        else:
            color.append('#e82525')

    (fig, ax) = createGoalMouth('white', bg_color)
    xGs = np.arange(0.1, 0.6, 0.1)
    xGs = xGs * multiply_size

    sizes = calculated_xg.xG * multiply_size
    
    scatter = ax.scatter(calculated_xg['X'], calculated_xg['Y'], c=color, edgecolors='white', s=sizes, marker='o', label=calculated_xg['Result'], alpha=0.6, zorder=5)
    
    # information box about xG size
    info_y = 55
    info_x = 47
    
    plt.text(info_x - 7, info_y, "xG values", fontsize=9, color="white", verticalalignment="center", horizontalalignment="left")
    for i in range(0,5):
        x = info_x + 3.5 * i
        x_text = x + 1
        plt.scatter(x, info_y, edgecolor='white', s=xGs[i], marker='o', zorder=8, c=bg_color, alpha=0.7)
        plt.text(x_text, info_y, round(xGs[i] / multiply_size, 1), fontsize=9, color="white", verticalalignment="center", horizontalalignment="left")

    # information box about goal or not
    plt.text(info_x - 7, info_y - 2, "Goal?", fontsize=9, color="white", verticalalignment="center", horizontalalignment="left")
    for i in range(0,2):
        x = info_x + 5 * i
        x_text = x + 1
        color = '#4925e8' if i == 0 else '#e82525'
        text = 'TRUE' if i == 0 else 'FALSE'
        plt.scatter(x, info_y - 2, edgecolor=color, s=xGs[0], marker='o', zorder=8, c=color, alpha=0.7)
        plt.text(x_text, info_y - 2, text, fontsize=9, color="white", verticalalignment="center", horizontalalignment="left")

    # Statistical information
    stats_info_caption = "Shots\nGoals\nxG Sum\nxG per shot\nShots per goal"
    stats_info_value = str(len(calculated_xg)) + "\n" + str(len(calculated_xg.loc[(calculated_xg.Goal == 1)])) +  "\n" +  str(round(sum(calculated_xg.xG), 2)) + "\n" +  str(round(statistics.mean(calculated_xg.xG), 2)) + "\n" + str(round(len(calculated_xg) / len(calculated_xg.loc[(calculated_xg.Goal == 1)]), 2))
    plt.text(61, 8, stats_info_caption, fontsize=9, color='white', verticalalignment='top', horizontalalignment='right')
    plt.text(62, 8, stats_info_value, fontsize=9, color='white', verticalalignment='top', horizontalalignment='left')

    fig.set_size_inches(10.5, 6.8)
    plt.tight_layout()
    
    fig.patch.set_facecolor(bg_color)
    if ( caption ):
        plt.text(0.05, info_y, "{}".format(caption), fontsize=16, fontweight="bold", horizontalalignment="left", verticalalignment="center", color="white")
    if ( subcaption ):
        plt.text(0.05, info_y - 2, "{}".format(subcaption), fontsize=10, fontweight="normal", horizontalalignment="left", verticalalignment="center", color="white")
  
    plt.ylim((-3, 60))
    image_field = make_image_to_buffer()
    if ( save_pic == 1 ):
        file = team.replace("/","")
        file_name = file + "_shots-xG_hm.jpg"
        fig.savefig(file_name, dpi=200, facecolor=bg_color)
        plt.show()
    
    plt.close()
    return(image_field)

def whenGoalsAreMade(df, ht, at, caption, subcaption, save_pic=0):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(frame_on=True)
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='x', colors='black')
    # ax.tick_params(axis='y', colors='black')
    ax.spines['bottom'].set_color('black')
        
    end_of_match = 95

    plt.xlim(0,end_of_match)
    plt.xticks(np.arange(0,end_of_match+1, 5))
    
    # Add goals to line
    i = 0
    max = 0
    for time in range(0, end_of_match, 5):
        minimi = time * 60
        maximi = (time + 5) * 60
        own_goal = len(df.loc[((df['start'].between(minimi,maximi)) & (df['Team'] == ht) & (df['Result'].str.contains('Goal')))])
        opp_goal = len(df.loc[((df['start'].between(minimi,maximi)) & (df['Team'] == at) & (df['Result'].str.contains('Goal')))])

        if ( max < own_goal ):
            max = own_goal
            
        if ( max < opp_goal ):
            max = opp_goal

        x = time + 1.25         
        plt.bar(x, own_goal, width=2.5, color="blue", alpha=0.3, zorder=4)
        plt.bar(x + 2.5, opp_goal, width=2.5, color="red", alpha=0.3, zorder=3)
        i += 1

    plt.ylim((0,max))
    #plt.yticks(np.arange(0,max + 0.5, 1))
    #ax.set_ylabel('Goals', fontsize=12, fontweight='bold')
    ax.set_xlabel('Minutes', fontsize=12, fontweight='bold')
    ax.set_facecolor('ghostwhite')
    fig.set_size_inches(20, 10)
    
    if ( caption ):
        fig.suptitle("{}".format(caption), fontsize=16, fontweight="bold")
        
    if ( subcaption ):
        ax.set_title("{}".format(subcaption))
    
    Image = make_image_to_buffer()
    if ( save_pic == 1 ):
        file_name = "When_goals_are_made.jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()
    
    #plt.tight_layout()
    plt.close()
    return(Image)

def makeChartDonut(data, color, caption='', subcaption='', team='', save_pic=0):
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    # create donut plots
    startingRadius = 0.7 + (0.3* (len(data)-1))
    for index, row in data.iterrows():
        scenario = row["name"]
        percentage = row["percentage"]
        all_txt = row["all"]
        if ( scenario == "Ball Possession" ):
            textLabel = scenario + ' ' + str(round(percentage,2)) + '% (' + str(round(all_txt,2)) + ' min)'
        else:
            textLabel = scenario + ' ' + str(round(percentage,2)) + '% (' + str(round(all_txt,2)) + ' kpl)'
            
        remainingPie = 100 - percentage
    
        donut_sizes = [remainingPie, percentage]
    
        plt.text(0.01, startingRadius - 0.07, textLabel, horizontalalignment='center', fontsize=16, fontweight='bold', verticalalignment='center')
        plt.pie(donut_sizes, radius=startingRadius, startangle=90, colors=[color[0], color[1]],
                wedgeprops={"edgecolor": "white", 'linewidth': 1})
    
        startingRadius-=0.3
    
    # equal ensures pie chart is drawn as a circle (equal aspect ratio)
    plt.axis('equal')
    
    fig.set_size_inches(10, 10)
    
    # create circle and place onto pie chart
    circle = plt.Circle(xy=(0, 0), radius=0.35, facecolor='white')
    plt.gca().add_artist(circle)
    plt.tight_layout()

    if ( caption ):
        #fig.text(0.05, 1, "{}\n".format(caption), fontsize=16, fontweight="bold")
        fig.suptitle("{}\n".format(caption), fontsize=16, fontweight="bold")
    if ( subcaption ):
        fig.text(0.05, 1, "{}".format(subcaption), fontsize=10, fontweight="normal")
        #ax.set_title("{}".format(subcaption), fontsize=10, fontweight="normal")
    
    Image = make_image_to_buffer()
    if ( save_pic == 1 ):
        file = team.replace("/","")
        file_name = file + "_donuts.jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()

    plt.close()
    return(Image)    

def makeBasicStats(df, team):
    event = df.loc[((df['action'].str.contains('BP')) & (df['Team'] == team))]
    team_duration = sum(event['duration'])
    event = df.loc[(df['action'].str.contains('BP'))]
    all_duration = sum(event['duration'])    
    xg_counted = xG(df, team)
    
    basic_stats = {
        'yellow' : len(df.loc[((df['Team'] == team) & (df['action'] == 'Yellow Card'))]),
        'red' : len(df.loc[((df['Team'] == team) & (df['action'] == 'Red Card'))]),
        'corner' : len(df.loc[((df['Team'] == team) & (df['SP'] == 'Corner'))]),
        'fk' : len(df.loc[((df['Team'] == team) & (df['SP'].str.contains('FK')))]),
        'saved_our_gk' : len(df.loc[((df['Team'] != team) &  (df['Result'].str.contains('Saved')))]),
        'saved' : len(df.loc[((df['Team'] == team) &  (df['Result'].str.contains('Saved')))]),
        'shots' : len(df.loc[((df['Team'] == team) & (df['Penetration'].str.contains('Shoot')))]),
        'block' : len(df.loc[((df['Team'] == team) & (df['Result'].str.contains('Blocked')))]),
        'goals' : len(df.loc[((df['Team'] == team) & (df['Result'].str.contains('Goal')))]),
        'penalty' : len(df.loc[((df['Team'] == team) & (df['SP'].str.contains('Penalty kick')))]),
        'throw_in' : len(df.loc[((df['Team'] == team) & (df['SP'].str.contains('Throw in')))]),
        'bp_time' : round(team_duration / 60, 2),
        'bp_p' : round(div_zero(team_duration, all_duration) * 100,2),
        'success_pass' : len(df.loc[((df['Team'] == team) & (df['action'] == 'Pass') & (df['AR'] == 'Success'))]),
        'failure_pass' : len(df.loc[((df['Team'] == team) & (df['action'] == 'Pass') & (df['AR'] == 'Failure'))]),
        'accurate_pass' : round(div_zero(len(df.loc[((df['Team'] == team) & (df['action'] == 'Pass') & (df['AR'] == 'Success'))]), len(df.loc[((df['Team'] == team) & (df['action'] == 'Pass') & (df['AR'] == 'Success'))]) + len(df.loc[((df['Team'] == team) & (df['action'] == 'Pass') & (df['AR'] == 'Failure'))])) * 100, 2),
        'all_pass' : len(df.loc[((df['Team'] == team) & (df['action'] == 'Pass') & (df['AR'] == 'Success'))]) + len(df.loc[((df['Team'] == team) & (df['action'] == 'Pass') & (df['AR'] == 'Failure'))]),
        'pass_freq_a' : round(( len(df.loc[((df['Team'] == team) & (df['action'] == 'Pass') & (df['AR'] == 'Success'))]) + len(df.loc[((df['Team'] == team) & (df['action'] == 'Pass') & (df['AR'] == 'Failure'))]) ) / (team_duration / 60), 1),  #+ " pass/min"
        'pass_freq_s' : round(len(df.loc[((df['Team'] == team) & (df['action'] == 'Pass') & (df['AR'] == 'Success'))]) / (all_duration / 60), 2),
        'pass_freq_f' : round(len(df.loc[((df['Team'] == team) & (df['action'] == 'Pass') & (df['AR'] == 'Failure'))]) / (all_duration / 60), 2),
        'xG' : round(sum(xg_counted['xG']), 2),
        'shot_accurate' : div_zero(len(df.loc[((df['Team'] == team) &  (df['Result'].str.contains('Saved')))]) + len(df.loc[((df['Team'] == team) & (df['Result'].str.contains('Goal')))]) +  len(df.loc[((df['Team'] == team) & (df['Result'].str.contains('Blocked')))]), len(df.loc[((df['Team'] == team) & (df['Penetration'].str.contains('Shoot')))])) * 100,
        }
    
    return(basic_stats)

def makeBasicStatsAverage(df, team, num_of_matches=1):
    event = df.loc[((df['action'].str.contains('BP')) & (df['Team'] == team))]
    team_duration = sum(event['duration'])
    event = df.loc[(df['action'].str.contains('BP'))]
    all_duration = sum(event['duration'])
    xg_counted = xG(df, team)
    success_pass = len(df.loc[((df['Team'] == team) & (df['action'] == 'Pass') & (df['AR'] == 'Success'))])
    failure_pass = len(df.loc[((df['Team'] == team) & (df['action'] == 'Pass') & (df['AR'] == 'Failure'))])
    all_passes = len(df.loc[((df['Team'] == team) & (df['action'] == 'Pass') & (df['AR'] == 'Success'))]) + len(df.loc[((df['Team'] == team) & (df['action'] == 'Pass') & (df['AR'] == 'Failure'))])

    basic_stats = {
        'yellow' : round(len(df.loc[((df['Team'] == team) & (df['action'] == 'Yellow Card'))]) / num_of_matches, 2),
        'red' : round(len(df.loc[((df['Team'] == team) & (df['action'] == 'Red Card'))]) / num_of_matches, 2),
        'corner' : round(len(df.loc[((df['Team'] == team) & (df['SP'] == 'Corner'))]) / num_of_matches, 2),
        'fk' : round(len(df.loc[((df['Team'] == team) & (df['SP'].str.contains('FK')))]) / num_of_matches, 2),
        'saved_our_gk' : round(len(df.loc[((df['Team'] != team) &  (df['Result'].str.contains('Saved')))]) / num_of_matches, 2),
        'saved' : round(len(df.loc[((df['Team'] == team) &  (df['Result'].str.contains('Saved')))]) / num_of_matches, 2),
        'shots' : round(len(df.loc[((df['Team'] == team) & (df['Penetration'].str.contains('Shoot')))]) / num_of_matches, 2),
        'block' : round(len(df.loc[((df['Team'] == team) & (df['Result'].str.contains('Blocked')))]) / num_of_matches, 2),
        'goals' : round(len(df.loc[((df['Team'] == team) & (df['Result'].str.contains('Goal')))]) / num_of_matches, 2),
        'penalty' : round(len(df.loc[((df['Team'] == team) & (df['SP'].str.contains('Penalty kick')))]) / num_of_matches, 2),
        'throw_in' : round(len(df.loc[((df['Team'] == team) & (df['SP'].str.contains('Throw in')))]) / num_of_matches, 2),
        'bp_time' : round(((team_duration / num_of_matches) / 60), 2),
        'bp_p' : round(div_zero(team_duration, all_duration) * 100,2),
        'success_pass' : round(success_pass / num_of_matches, 2),
        'failure_pass' : round(failure_pass / num_of_matches, 2),
        'accurate_pass' : round(div_zero(success_pass, all_passes) * 100, 2),
        'all_pass' : round(all_passes / num_of_matches, 2),
        'pass_freq_a' : round(all_passes / (team_duration / 60), 2),
        'pass_freq_s' : round(success_pass / (all_duration / 60), 2),
        'pass_freq_f' : round(failure_pass / (all_duration / 60), 2),
        'xG' : round(sum(xg_counted['xG']) / num_of_matches, 2),
        'shot_accurate' : round(div_zero(len(df.loc[((df['Team'] == team) &  (df['Result'].str.contains('Saved')))]) + len(df.loc[((df['Team'] == team) & (df['Result'].str.contains('Goal')))]) +  len(df.loc[((df['Team'] == team) & (df['Result'].str.contains('Blocked')))]), len(df.loc[((df['Team'] == team) & (df['Penetration'].str.contains('Shoot')))])) * 100,2),
        }
    
    return(basic_stats)

def plotBasicStats(stats, save_pic=0):
    basic_stats = { 'xG' : 'xG',
                    'yellow' : 'Yellow Cards',                    
                    'red' : 'Red Cards',
                    'corner' : 'Corners',
                    'fk' : 'Free kicks',
                    'penalty' : 'Penalty kicks',
                    'throw_in' : 'Throw in',                    
                    'accurate_pass' : 'Pass accurate %',
                    'pass_freq_a' : 'Pass/min',
                    'bp_time' : 'Ball possession',                    
                    'block' : 'Blocked shots',                    
                    'shots' : 'Shots',
                    'saved_our_gk' : 'Saves',
                   }
    
    y_ticks_label = []
    data = []
    for stat in basic_stats:
        y_ticks_label.append(basic_stats[stat])
        home = stats['home'][stat]
        away = stats['away'][stat]
            
        data.append([home, away])
    
    likert_colors = ['white','blue', 'red']
    dummy = pd.DataFrame(data,
                         columns=["home", "away"],
                         index=y_ticks_label)
    
    middle_bar = 25
    middles = dummy["home"]
    longest = middles.max()
    complete_longest = int(dummy.sum(axis=1).max()) + middle_bar
    dummy.insert(0, '', (middles - longest).abs())

    bar_height = 0.9
    fig, ax = plt.subplots(figsize=(18,13))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlim(0, complete_longest)
    
    for i, row in dummy.iterrows():
        ax.barh(i, row[''], left=0, height=bar_height, label=i, color=likert_colors[0], alpha=0.3)
        ax.barh(i, row['home'], left=row[''], height=bar_height, label=i, color=likert_colors[1], alpha=0.3)
        ax.barh(i, middle_bar, left=row[''] + row['home'], height=bar_height, label=i, color='black', alpha=0.3)
        ax.barh(i, row['away'], left=row[''] + row['home'] + middle_bar, height=bar_height, label=i, color=likert_colors[2], alpha=0.3)

        if ( i in ['Pass accurate %'] ):
            home_value = str(row['home']) + '%'
            away_value = str(row['away']) + '%'
        elif ( i in ['Ball possession'] ):
            home_value = str(row['home']) + ' min'
            away_value = str(row['away']) + ' min'            
        elif ( i in ['xG', 'Pass/min'] ):
            home_value = str(row['home'])
            away_value = str(row['away'])
        else:
            home_value = str(row['home'])
            away_value = str(row['away'])
            
        ax.text(row[''] - 2, i, home_value, ha='right', va='center', color='black', fontsize=15, fontweight='bold')
        ax.text(row[''] + row['home'] + middle_bar/2, i, str(i), ha='center', va='center', color='black', fontsize=12, fontweight='bold')
        ax.text(row[''] + middle_bar + row['home'] + row['away'] + 2, i, away_value, ha='left', va='center', color='black', fontsize=15, fontweight='bold')
    
    ax.set(frame_on=False)
    
    fig.set_size_inches(18, 13)
    #plt.tight_layout()
    Stats_image = make_image_to_buffer()
    if ( save_pic == 1 ):
        file_name = "Basic_stats_from_match.jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()

    plt.close()
    return(Stats_image)

def drawGameActions(data, teams, caption, subcaption, save_pic=0):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')
    ax.spines['bottom'].set_color('gray')
    ax.spines['bottom'].set_position('center')
    
    # Add actions to line
    actions = data.loc[((data['action'] == 'Set pieces') | (data['Penetration'].str.contains('Shoot')) | (data['Penetration'].str.contains('Key pass')) )]
    y = 0
    max_y = 0
    min_y = 0
    line_x = [0]
    line_y = [0]
    last_action_time = 0
    for index, action in actions.iterrows():
        add = 0
        
        if ( 'Shoot' in action['Penetration'] ):
            if ( 'Goal' in action['Result'] ):
                add = 1
            else:
                add = 0.5
        else:
            if ( 'Key pass' in action['Penetration'] ):
                add = 0.5
                
            if ( action['SP'] != 'Throw in' ):
                add = 0.5
            
        if ( teams[0] == action['Team'] ):
            y += add
            text_y = y + 0.5
            color = 'blue'
            align = 'bottom'
        else:
            y -= add
            text_y = y - 0.5          
            color = 'red'
            align = 'top' 

        if ( action['action'] == 'Set pieces' ):
            action_text = action['SP']
        else:
            action_text = 'Shot > ' + action['Result']
            
        x = action['start'] / 60
        last_action_time = action['end'] / 60
        
        plt.plot(x, y, 'ro-', color=color, zorder=9)
        line_x.append(x)
        line_y.append(y)
        
        max_y = y if max_y < y else max_y
        min_y = y if min_y > y else min_y
        
        ax.text(x, text_y, action_text, rotation=90, fontsize=6, color='grey', horizontalalignment='center', fontweight='bold', verticalalignment=align, zorder=7)

    plt.plot(line_x, line_y, '-', color='grey', zorder=5, alpha=.6)
    plt.xlim(0,last_action_time)
    plt.plot([0,last_action_time],[0, 0], "-", color='k', alpha=.7, zorder=8)

    fig.set_size_inches(20, 5)
    #plt.tight_layout()
    plt.ylim((min_y - 5, max_y + 5))
    
    # Home team background
    home_bg = plt.Rectangle((0, 0), last_action_time, max_y + 5,ls='-', color='blue', zorder=1, alpha=.05)
    away_bg = plt.Rectangle((0, min_y - 5), last_action_time, abs(min_y) + 5,ls='-', color='red', zorder=1, alpha=.05)
    ax.add_artist(home_bg)
    ax.add_artist(away_bg)
    
    if ( caption ):
        fig.suptitle("{}".format(caption), fontsize=16, fontweight="bold")
        
    if ( subcaption ):
        ax.set_title("{}".format(subcaption))

    Action_image = make_image_to_buffer()
    if ( save_pic == 1 ):
        file_name = "Game_actions_in_match.jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()
    
    plt.close()
    return(Action_image)

def div_zero(a, b):
    if (b == 0):
        return(0.0)
    else:
        return(a/b)

def make_image_to_buffer():
    imgdata = BytesIO()
    plt.savefig(imgdata, format='png', dpi=200)
    imgdata.seek(0)  # rewind the data
    
    Image = ImageReader(imgdata)
    return(Image)    

def addIconsToPDF(c, home, start_x = 2.5, start_y = 25):
    icons = [ "yellow", "red", "corner", "fk", "saved_our_gk", "penalty", "throw_in", "pass_freq_a" ]
    path = "Icons"
    
    i = 0
    # Make basic statistic icons to pdf file
    c.setFillColor("black")
    c.setFont('Helvetica', 9)
    for icon in icons:
        x_coordinate = start_x + (1.1*i)
        aspect = 1 / 0.6
        
        if ( icon == "yellow" ):
            c.setFillColor("yellow")
            c.rect(x_coordinate*cm,start_y*cm,0.6*cm,1*cm, stroke=0, fill=1)
        elif ( icon == "red" ):
            c.setFillColor("red")
            c.rect(x_coordinate*cm,start_y*cm,0.6*cm,1*cm, stroke=0, fill=1)
        else:
            img = ImageReader(path + "/" + icon + ".png")
            iw, ih = img.getSize()
            aspect = ih / float(iw)
            c.drawImage(img, x_coordinate*cm, start_y*cm, width=(1/aspect)*cm, height=1*cm, mask='auto') # Home team
        
        c.setFillColor("black")
        text_x = x_coordinate + ((1/aspect)/2)
        icon_text = str(home[icon])
        if ( icon == 'pass_freq_a' ):
            icon_text = str(home[icon]) + " pass/min"
        c.drawCentredString(text_x*cm, (start_y-0.3)*cm, icon_text)
        i += 1
       
def addHeaderToPDF(c, ottelu, info, logo, hgoal, agoal, logo_x=15, logo_y=0.4):
    c.setLineWidth(.3)
    # Make info score and teams
    c.setFont('Helvetica', 20)
    c.drawCentredString(10.5*cm, 28.0*cm, str(ottelu))
    c.line(2*cm, 27.7*cm, 19*cm, 27.7*cm)
    c.setFillColor("grey")
    c.rect(2*cm,27*cm,2 *cm,2*cm, stroke=0, fill=1)
    c.rect(17*cm,27*cm,2 *cm,2*cm, stroke=0, fill=1)    

    # Info which match
    c.setFont('Helvetica', 8)
    c.drawCentredString(10.5*cm, 27.3*cm, str(info))
    c.setFillColor("white")
    c.setFont('Helvetica-Bold', 22)
    c.drawCentredString(3*cm, 27.7 *cm, str(hgoal))
    c.drawCentredString(18*cm, 27.7 *cm, str(agoal))
    
    if ( os.path.isfile(logo) ):
        img = ImageReader(logo)
        iw, ih = img.getSize()
        aspect = ih / float(iw) 
        c.drawImage(img, logo_x*cm, logo_y*cm, width=30*cm, height=(30*aspect)*cm, mask='auto')
        
    c.setAuthor("Mauri Heinonen")
    c.setTitle("Match report is generated from MyVideoanalyser data which is made by author.")
    c.setSubject("Tagged fooball match statistical report")

def getPassesCount(df, third, team):
    areas = df.loc[((df['action'] == third) & (df['Team'] == team))]
    passes = 0
    
    for index, area in areas.iterrows():
        bp_start = area['start']
        bp_end = area['end']
        right_time = df.loc[(df['Team'] == team)]
        right_time = right_time.loc[(right_time['start'].between(bp_start, bp_end) | right_time['end'].between(bp_start, bp_end))]
        passes += len(right_time.loc[(right_time['action'] == 'Pass')])
        
    #print("{} - {}".format(passes,third))
    return(passes)
    
def passesInThirds(df, team='', caption='', subcaption='', field_height=105, field_width=68, save_pic=0):
    file = team.replace("/","")
    file_name = file + "passess_in_thirds.jpg"
    (fig,ax) = createPitch(field_height,field_width,'yards','white','#80B860')
        
    # Add opponent ball possession chart to figure
    passes_model = pd.DataFrame(columns=['Result','X','Y'])
    for x in range(1,4): #['BP 1-1', 'BP 1-2', 'BP 1-3', 'BP 2-1', 'BP 2-2', 'BP 2-3', 'BP 3-1', 'BP 3-2', 'BP 3-3']:
        for y in range(1,4):
            third = 'BP ' + str(x) + '-' + str(y) 
            areas = df.loc[((df['action'] == third) & (df['Team'] == team))]
    
            for index, area in areas.iterrows():
                bp_start = area['start'] - 0.5
                bp_end = area['end'] + 0.5
                right_time = df.loc[(df['Team'] == team)]
                right_time = right_time.loc[(right_time['start'].between(bp_start, bp_end) | right_time['end'].between(bp_start, bp_end))]
                right_time = right_time.loc[(right_time['action'] == 'Pass')]
            
                for i,pas in right_time.iterrows():
                    passes_model.at[i,'X'] = x
                    passes_model.at[i,'Y'] = y
                    passes_model.at[i,'Result'] = pas['AR']
    
    # Make chart
    box_height = field_height / 3
    box_width = field_width / 3
    for key in range(1, 4, 1):
        x = box_height * (key - 1)
        box1 = mpatches.Rectangle((x, box_width*2), box_height, box_width, ls='-', color='white', zorder=9, alpha=div_zero(len(passes_model.loc[((passes_model['X'] == key) & (passes_model['Y'] == 1))]),len(passes_model)))
        box2 = mpatches.Rectangle((x, box_width), box_height, box_width, ls='-', color='white', zorder=9,alpha=div_zero(len(passes_model.loc[((passes_model['X'] == key) & (passes_model['Y'] == 2))]),len(passes_model)))
        box3 = mpatches.Rectangle((x, 0), box_height, box_width, ls='-', color='white', zorder=9,alpha=div_zero(len(passes_model.loc[((passes_model['X'] == key) & (passes_model['Y'] == 3))]),len(passes_model)))
                
        ax.add_patch(box1)
        ax.add_patch(box2)
        ax.add_patch(box3)
        
        text_x = x + box_height / 2
        plt.text(text_x, box_width*2 + box_width / 2, '{} passes'.format(len(passes_model.loc[((passes_model['X'] == key) & (passes_model['Y'] == 1))])), color='black', fontsize=12, fontweight='bold', va='center', ha='center', zorder=10)
        plt.text(text_x, box_width + box_width / 2, '{} passes'.format(len(passes_model.loc[((passes_model['X'] == key) & (passes_model['Y'] == 2))])), color='black', fontsize=12, fontweight='bold', va='center', ha='center', zorder=10)
        plt.text(text_x, box_width / 2, '{} passes'.format(len(passes_model.loc[((passes_model['X'] == key) & (passes_model['Y'] == 3))])), color='black', fontsize=12, fontweight='bold', va='center', ha='center', zorder=10)
    
    fig.set_size_inches(field_height/10, field_width/10)
    if ( caption ):
        fig.suptitle("{}\n".format(caption), fontsize=16, fontweight="bold")
    if ( subcaption ):
        fig.text(0.05, 1, "{}".format(subcaption), fontsize=10, fontweight="normal")
                
    plt.tight_layout()
    image_field = make_image_to_buffer()
    if ( save_pic == 1 ):
        file = team.replace("/","")
        file_name = file + "_passes_in_thirds.jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()

    plt.close()
    return(image_field)

def plotFlanks(df, team='', caption='', subcaption='', image_width=50, image_height=65, save_pic=1):
    num_of_matches = len(df['match_id'].unique())
    data = {}
    data['Left'] = len(df.loc[((df['corrider'].str.contains('Left')) & (df['Team'] == team))]) / num_of_matches
    data['Center'] = len(df.loc[((df['corrider'].str.contains('Center')) & (df['Team'] == team))]) / num_of_matches
    data['Right'] = len(df.loc[((df['corrider'].str.contains('Right')) & (df['Team'] == team))]) / num_of_matches
    
    data['Left_act'] = len(df.loc[(((df['Penetration'].str.contains('Cross')) | (df['Penetration'].str.contains('Shoot')) | (df['Penetration'].str.contains('Key pass'))) & (df['corrider'].str.contains('Left')) & (df['Team'] == team))]) / num_of_matches
    data['Center_act'] = len(df.loc[(((df['Penetration'].str.contains('Cross')) | (df['Penetration'].str.contains('Shoot')) | (df['Penetration'].str.contains('Key pass'))) & (df['corrider'].str.contains('Center')) & (df['Team'] == team))]) / num_of_matches
    data['Right_act'] = len(df.loc[(((df['Penetration'].str.contains('Cross')) | (df['Penetration'].str.contains('Shoot')) | (df['Penetration'].str.contains('Key pass'))) & (df['corrider'].str.contains('Right')) & (df['Team'] == team))]) / num_of_matches
    data['max'] = max( data['Left'], data['Center'], data['Right'])    
        
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)

    ax.set(frame_on=False)
    #ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.ylim((0,image_width + 5))

    column = 1
    align_text = 'center'
    r = image_width / data['max']
    
    for corridor in ['Left', 'Center', 'Right']:
        start_point = data[corridor + '_act'] * r
        all_act = data[corridor] * r
        act_pro = data[corridor + '_act'] * r
        ration = round(data[corridor + '_act'] / data[corridor] * 100, 1)
                
        plt.bar(column, all_act - start_point, bottom=start_point, color="blue", alpha=0.5, zorder=5)
        plt.bar(column, act_pro, color="black", alpha=0.5, zorder=6)

        # Add text for total and for percentual
        plt.text(column, all_act, "{} pcs".format(round(data[corridor],1)), fontsize=8, horizontalalignment=align_text, fontweight='bold', color='black', verticalalignment='bottom', zorder=7)
        plt.text(column, start_point / 2, "{}%".format(ration), fontsize=8, horizontalalignment='center', verticalalignment='center', fontweight='bold', color='white', zorder=7)
        
        column += 1
    
    ax.axes.xaxis.set_ticklabels(['Left', 'Center', 'Right'])
    plt.xticks([1,2,3])
    if ( caption ):
        fig.suptitle("{}\n".format(caption), fontsize=16, fontweight="bold")
    if ( subcaption ):
        #fig.text(0.05, 1, "{}".format(subcaption), fontsize=10, fontweight="normal")
        ax.set_title("{}".format(subcaption), fontsize=10, fontweight="normal")
        
    if ( save_pic == 1 ):
        file = team.replace("/","")
        file_name = file + "_attacks_from_flanks.jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()
    
    plt.close()
    return()

def deleteFromDB(table, column, column_value, db):
    # Check if this match is already in db
    sql = "select * from {} where {} = {}".format(table, column, column_value)
    print(sql)
    match_info = pd.read_sql( sql, db)
    print("\n\nIs this {} already in database? - {}".format(column_value, not match_info.empty))
    
    # Delete rows from database if match id is already in database
    if ( not match_info.empty ):
        mycursor = db.cursor()
        sql_del = "delete from {} where {} = {}".format(table, column, column_value)
        mycursor.execute(sql_del)
        db.commit()
        print(mycursor.rowcount, "record(s) deleted")
    return()

def makePDF(loaded_data, home_team, away_team, str_date, info_txt, home_goals, away_goals, save_pic=0):
    count = 1
    # Make PDF document, paper size is A4
    c = canvas.Canvas( "./" + str(str_date) + " - " + str(home_team) + "-" + str(away_team) + ".pdf", pagesize=A4)

    #print("Adding header and logos")
    addHeaderToPDF(c, home_team + " - " + away_team, info_txt, "", home_goals, away_goals, -2, -2)
    stats = {}
    stats['home'] = makeBasicStats(loaded_data, home_team)
    stats['away'] = makeBasicStats(loaded_data, away_team)
        
    basic_stats = plotBasicStats(stats, save_pic)
    c.drawImage(basic_stats, 0.5*cm, 13*cm, width=20*cm, height=14.44*cm, mask=[255,255,255,255,255,255])
        
    # When goals are made in this match
    time_when_goal = whenGoalsAreMade(loaded_data, home_team, away_team, "When goals are made", "", save_pic)
    c.drawImage(time_when_goal, 2.5*cm, 6*cm, width=16*cm, height=8*cm, mask=[255,255,255,255,255,255])

    game_actions = drawGameActions(loaded_data, [home_team, away_team], '', '', save_pic)
    c.drawImage(game_actions, 0.5*cm, 0.5*cm, width=20*cm, height=5*cm, mask=[255,255,255,255,255,255])
            
    c.showPage()

    for team in [home_team, away_team]:
        other_team = away_team if team == home_team else home_team
                
        # Draw basic information about game like ball possession, passes and shoots
        team_stats = makeBasicStats(loaded_data, team)
        team_donats = { "name" : ["Ball Possession", "Pass Accuracy", "Shot Accuracy"],
                       "percentage" : [ team_stats['bp_p'], team_stats['accurate_pass'], team_stats['shot_accurate']],
                       "all" : [ team_stats['bp_time'], team_stats['success_pass'] + team_stats['failure_pass'], team_stats['shots'] ]}    

        #print("Adding header and logos")
        addHeaderToPDF(c, home_team + " - " + away_team, info_txt, "Logos/" + team +".png", home_goals, away_goals, -2, -2)#13, 1)    
        addIconsToPDF(c, team_stats, 6, 25.5)
                
        #passes = passesInThirds(loaded_data, team, 'How many passes is made in thirds', save_pic)
        passes = passesToBox(loaded_data, team, '{} entries to the box'.format(team), save_pic)
        c.drawImage(passes, 10*cm, 0.5*cm, width=9*cm, height=6.12*cm, mask=[255,255,255,255,255,255])

        penetratingToBox(loaded_data, team, '{} entries to the box'.format(team), '', save_pic)

        plotCorners(loaded_data, team, '{} corners'.format(team), '', save_pic)
        plotAttackFreeKicks(loaded_data, team, '{} attack free kicks'.format(team), '', save_pic)
        passesInBoxHeatMap(loaded_data, team, '{} target of passes in the box'.format(team), '', save_pic)

        #flanks = plotFlanks(loaded_data, team, '','', save_pic)
        #c.drawImage(flanks, 10*cm, 0.5*cm, width=9*cm, height=6.12*cm, mask=[255,255,255,255,255,255])
        
        team_data = pd.DataFrame(data=team_donats)
        donuts = makeChartDonut(team_data, ['#d5f6da', '#5cdb6f'], 'Basic statistics for {}'.format(team), '', team, save_pic)
        c.drawImage(donuts, 2*cm, 18.5*cm, width=6*cm, height=6*cm, mask=[255,255,255,255,255,255])

        # Draw interceptions with color codes
        interceptions = drawIntercept(loaded_data, team, "Interceptions for {} against opponent ball possession".format(team), '', other_team, 105, 68, save_pic)
        drawThrowIns(loaded_data, team, "Throw Ins for {}".format(team), '', other_team, 105, 68, save_pic)
        drawGoalPasses(loaded_data, team, "{}'s passes which lead to score".format(team), '', other_team, 105, 68, save_pic)
        c.drawImage(interceptions, 1*cm, 12.5*cm, width=9*cm, height=6.12*cm, mask=[255,255,255,255,255,255])
        
        # Draw scooring chance positions and calculate xG's
        calculated_team_xg = xG(loaded_data, team, 0)
        plot_xG = plotShotsAndxG(calculated_team_xg, description, "Scoring chances for {}".format(team), '', team, save_pic)
        c.drawImage(plot_xG, 1*cm, 6.5*cm, width=9*cm, height=6.12*cm, mask=[255,255,255,255,255,255])
        plotShotsHex(calculated_team_xg, description, "Shots by {}".format(team), '', team, save_pic)
    
        # High / mid / low block
        high = playedBetweenLines(loaded_data, team, "{} passes between the lines against high block".format(team), "", 105, 68, "h", 0, save_pic)
        mid = playedBetweenLines(loaded_data, team, "{} passes between the lines against mid block".format(team), "", 105, 68, "m", 0, save_pic)
        low = playedBetweenLines(loaded_data, team, "{} passes between the lines against low block".format(team), "", 105, 68, "l", 0, save_pic)

        played6area(loaded_data, team, "{} passes to 6-area against high block".format(team), "", 105, 68, "h", 0, save_pic)
        played6area(loaded_data, team, "{} passes to 6-area against mid block".format(team), "", 105, 68, "m", 0, save_pic)
        played6area(loaded_data, team, "{} passes to 6-area against low block".format(team), "", 105, 68, "l", 0, save_pic)

        buildUpPhase(loaded_data, team, "{} passes to 6-area against high block".format(team), "", 105, 68, "h", 0, save_pic)
        buildUpPhase(loaded_data, team, "{} passes to 6-area against mid block".format(team), "", 105, 68, "m", 0, save_pic)
        buildUpPhase(loaded_data, team, "{} passes to 6-area against low block".format(team), "", 105, 68, "l", 0, save_pic)

        c.drawImage(high, 10*cm, 18.5*cm, width=9*cm, height=6.12*cm, mask=[255,255,255,255,255,255])
        c.drawImage(mid, 10*cm, 12.5*cm, width=9*cm, height=6.12*cm, mask=[255,255,255,255,255,255])
        c.drawImage(low, 10*cm, 6.5*cm, width=9*cm, height=6.12*cm, mask=[255,255,255,255,255,255])
    
        # Draw key passes beging and destination
        key_passes = playedBetweenLines(loaded_data, team, "{} key passes".format(team), "", 105, 68, "", 1, save_pic)
        c.drawImage(key_passes, 1*cm, 0.5*cm, width=9*cm, height=6.12*cm, mask=[255,255,255,255,255,255])
    
        if ( count < 2 ):
            c.showPage()
            count += 1   

    c.save()
    return()

def makeAllMatches(loaded_data, home_team, away_team, str_date, info_txt, home_goals, away_goals, save_pic=1):
    num_of_matches = len(loaded_data['match_id'].unique())

    stats = {}
    stats['home'] = makeBasicStatsAverage(loaded_data, home_team, num_of_matches)
    stats['away'] = makeBasicStatsAverage(loaded_data, away_team, num_of_matches)
    
    plotRadar(loaded_data, stats, home_team, away_team)
    
    plotBasicStats(stats, save_pic)
    whenGoalsAreMade(loaded_data, home_team, away_team, "When goals are made", "", save_pic)
    #drawGameActions(loaded_data, [home_team, away_team], '', '', save_pic)
    shots_model = xG(loaded_data, '', save_pic)
    plotAngleDistance(shots_model, 'Compare shots between angle and distance', 'Data from season 2020', save_pic=1)
         
    for team in [home_team, away_team]:
        other_team = away_team if team == home_team else home_team
                
        # Draw basic information about game like ball possession, passes and shoots
        team_stats = makeBasicStatsAverage(loaded_data, team, num_of_matches)
        team_donats = { "name" : ["Ball Possession", "Pass Accuracy", "Shot Accuracy"],
                       "percentage" : [ team_stats['bp_p'], team_stats['accurate_pass'], team_stats['shot_accurate']],
                       "all" : [ team_stats['bp_time'], team_stats['success_pass'] + team_stats['failure_pass'], team_stats['shots'] ]}    

        passesInThirds(loaded_data, team, 'How many passes is made in thirds', '', 105, 68, save_pic)
        passesToBoxHeatMap(loaded_data, team, '{} entries to the box'.format(team), '', save_pic)
        passesInBoxHeatMap(loaded_data, team, '{} entries of passes in the box'.format(team), '', save_pic)
        passesToBox(loaded_data, team, '{} entries to the box'.format(team), '', save_pic)
        plotFlanks(loaded_data, team, '{} attacks from flanks'.format(team),'Average per match', save_pic)
        penetratingToBox(loaded_data, team, '{} entries to the box'.format(team), '', save_pic)
        
        team_data = pd.DataFrame(data=team_donats)     
        makeChartDonut(team_data, ['#d5f6da', '#5cdb6f'], 'Basic statistics for {}'.format(team), '', team, save_pic)

        # Draw interceptions with color codes
        drawIntercept(loaded_data, team, "Interceptions for {} against opponent ball possession".format(team), '', other_team, 105, 68, save_pic)
        drawThrowIns(loaded_data, team, "Throw Ins for {}".format(team), '', other_team, 105, 68, save_pic)
        drawGoalPasses(loaded_data, team, "{}'s passes which lead to score".format(team), '', other_team, 105, 68, save_pic)

        # Draw scooring chance positions and calculate xG's
        calculated_team_xg = xG(loaded_data, team, 0)
        plotShotsAndxG(calculated_team_xg, description, "Scoring chances for {}".format(team), '', team, save_pic)
        plotShotsHex(calculated_team_xg, description, "Shots by {}".format(team), 'Summary of {} matches'.format(num_of_matches), team, save_pic)
        
        # High / mid / low block
        playedBetweenLines(loaded_data, team, "{} passes between the lines against high block".format(team), "", 105, 68, "h", 0, save_pic)
        playedBetweenLines(loaded_data, team, "{} passes between the lines against mid block".format(team), "", 105, 68, "m", 0, save_pic)
        playedBetweenLines(loaded_data, team, "{} passes between the lines against low block".format(team), "", 105, 68, "l", 0, save_pic)
        
        played6area(loaded_data, team, "{} passes to 6-area against high block".format(team), "", 105, 68, "h", 0, save_pic)
        played6area(loaded_data, team, "{} passes to 6-area against mid block".format(team), "", 105, 68, "m", 0, save_pic)
        played6area(loaded_data, team, "{} passes to 6-area against low block".format(team), "", 105, 68, "l", 0, save_pic)

        buildUpPhase(loaded_data, team, "{} passes to 6-area and between the lines against high block".format(team), "", 105, 68, "h", 0, save_pic)
        buildUpPhase(loaded_data, team, "{} passes to 6-area and between the lines against mid block".format(team), "", 105, 68, "m", 0, save_pic)
        buildUpPhase(loaded_data, team, "{} passes to 6-area and between the lines against low block".format(team), "", 105, 68, "l", 0, save_pic)
    
        # Draw key passes beging and destination
        playedBetweenLines(loaded_data, team, "{} key passes".format(team), "", 105, 68, "", 1, save_pic)

    return()

def plotRadar(df, stats, home_team, away_team, caption='', sup_caption='', save_pic=1):
    num_of_matches = len(df['match_id'].unique())
    ## parameter names
    params = ['Interceptions in last third', 'Interceptions', 'Set pieces', 'xG', 'Passes', 'Shots', 'Blocks', 'Goals', 'Pass / min']

    values = [
                [
                    round(len(df.loc[((df['action'] == 'Interception') & (df['Team'] == home_team) & (df['x'] > 70))]) / num_of_matches, 2), # Intercepts in last thirds
                    round(len(df.loc[((df['action'] == 'Interception') & (df['Team'] == home_team))]) / num_of_matches, 2), # Intercepts
                    round(len(df.loc[((df['action'] == 'Set pieces') & (df['Team'] == home_team))]) / num_of_matches, 2), # Set pieces
                    stats['home']['xG'], # xG
                    stats['home']['all_pass'],
                    stats['home']['shots'],
                    stats['home']['block'],
                    stats['home']['goals'],
                    stats['home']['pass_freq_a']                 
                ], # for home team
                [
                    round(len(df.loc[((df['action'] == 'Interception') & (df['Team'] == away_team) & (df['x'] > 70))]) / num_of_matches, 2), # Intercepts in last thirds
                    round(len(df.loc[((df['action'] == 'Interception') & (df['Team'] == away_team))]) / num_of_matches, 2), # Intercepts
                    round(len(df.loc[((df['action'] == 'Set pieces') & (df['Team'] == away_team))]) / num_of_matches, 2), # Set pieces
                    stats['away']['xG'], # xG
                    stats['away']['all_pass'],
                    stats['away']['shots'],
                    stats['away']['block'],
                    stats['away']['goals'],
                    stats['away']['pass_freq_a']
                ] # for away team
            ]
            
    ## range values
    ranges = [
                (min([values[0][0], values[1][0]])-5, max([values[0][0], values[1][0]])),
                (min([values[0][1], values[1][1]])-5, max([values[0][1], values[1][1]])),
                (min([values[0][2], values[1][2]])-5, max([values[0][2], values[1][2]])),
                (min([values[0][3], values[1][3]])-0.2, max([values[0][3], values[1][3]])),
                (min([values[0][4], values[1][4]])-5, max([values[0][4], values[1][4]])),
                (min([values[0][5], values[1][5]])-5, max([values[0][5], values[1][5]])),
                (min([values[0][6], values[1][6]])-5, max([values[0][6], values[1][6]])),
                (min([values[0][7], values[1][7]])-5, max([values[0][7], values[1][7]])),
                (min([values[0][8], values[1][8]])-5, max([values[0][8], values[1][8]]))
            ]

    ## instantiate object
    ## title
    title = dict(
        title_name = home_team,
        title_color='#9B3647',
        title_name_2 = away_team,
        title_color_2='#3282b8',
        title_fontsize=18
        )

    ## endnote 
    endnote = "Visualization made by: Mauri Heinonen"

    ## instantiate object 
    radar = Radar(background_color="#ffffff", patch_color="#f1f1f1", label_color="#000000", range_color="#000000")
              
    ## plot radar              
    fig, ax = radar.plot_radar(ranges=ranges, params=params, values=values, 
                               radar_color=['#9B3647', '#3282b8'], 
                               title=title, endnote=endnote,
                               alphas=[0.55, 0.5], compare=True)
    
    plt.tight_layout()
    if ( save_pic == 1 ):
        file_name = "Compare_radar.jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()
    
    plt.close()

def build_model(model_variables):
    model=''
    for v in model_variables[:-1]:
        model = model  + v + ' + '
    model = model + model_variables[-1]
    return model

def plotAngleDistance(shots_model, caption='', subcaption='', save_pic=0, bg_color='#171716', txt_color='white'):
    clusters = 6
    features = shots_model[['Degree', 'Distance', 'Goal']]
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)
    
    fit_model = LinearRegression(fit_intercept=True)
    x = shots_model['Degree']
    y = shots_model['Distance']

    fit_model.fit(np.vstack([x]).T, y)
    yf_cubic = fit_model.predict(np.vstack([x]).T)
    model = KMeans(n_clusters=clusters, random_state=0).fit(features)
    shots_model['Predicted_Group'] = model.labels_

    my_cmap = cmap.get_cmap('jet')
    my_norm = Normalize(vmin=0, vmax=clusters + 2)
    x_max_value = max(shots_model['Distance']) + 5
    y_max_value = max(shots_model['Degree']) + 5
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(frame_on=True)
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.spines['top'].set_color(txt_color)
    ax.spines['right'].set_color(txt_color)
    ax.spines['left'].set_color(txt_color)
    ax.spines['bottom'].set_color(txt_color)
    ax.tick_params(colors=txt_color)
    
    plt.xticks(np.arange(0,x_max_value, 5))
    plt.yticks(np.arange(0,y_max_value, 5))
    
    ax.set_facecolor(bg_color)
    ax.set_xlabel('Distance (m)', color=txt_color)
    ax.set_ylabel('Angle (degrees)', color=txt_color)
    
    plt.xlim(0, x_max_value)
    plt.ylim(0, y_max_value)

    data = shots_model[['Goal','Degree','Distance']].dropna()
    data.index = np.arange(0, len(data))

    # model_fit = smf.ols(formula='Goal ~ Degree + Distance', data=data).fit()
    model_fit = smf.ols(formula='Degree ~ Distance', data=data).fit()
    #print(model_fit.summary()) 
    #print(model_fit.params)

    index = 2
    for i in range(clusters):
        group_data = shots_model[shots_model['Predicted_Group'] == i]
        ax.scatter(group_data['Distance'], group_data['Degree'], edgecolors=my_cmap(my_norm(index)), color=bg_color, label=i, zorder=10)
            
        goal_data = group_data[group_data['Goal'] == 1]
        ax.scatter(goal_data['Distance'], goal_data['Degree'], edgecolors=my_cmap(my_norm(index)), color=my_cmap(my_norm(index)), zorder=10)
            
        index += 1
    
    yfit = fit_model.coef_[0] * x + fit_model.intercept_
    plt.plot(x, yfit, color=my_cmap(my_norm(0)))
    
    intercept_y = [i for i in range(int(y_max_value+1))]
    intercept_x = [fit_model.intercept_ for i in range(int(y_max_value+1))]
    
    plt.plot(intercept_x, intercept_y, '--', color=txt_color, alpha=0.2)
    
    # Statistical information
    stats_info_caption = "Slope: \nIntercept:\n"
    stats_info_value = str(round(fit_model.coef_[0],2)) + "\n" + str(round(fit_model.intercept_,2)) +  "\n"
    plt.text(x_max_value - 7, y_max_value - 2, stats_info_caption, fontsize=10, color=txt_color, verticalalignment='top', horizontalalignment='right')
    plt.text(x_max_value - 7, y_max_value - 2, stats_info_value, fontsize=10, color=txt_color, verticalalignment='top', horizontalalignment='left')


    fig.set_size_inches(10.5, 6.8)
    plt.tight_layout()
    
    fig.patch.set_facecolor(bg_color)
    if ( caption ):
        fig.text(0.05, 1, "{}\n".format(caption), fontsize=16, fontweight="bold", color=txt_color)
    if ( subcaption ):
        fig.text(0.05, 1, "{}".format(subcaption), fontsize=10, fontweight="normal", color=txt_color)
                
    plt.tight_layout()
    if ( save_pic == 1 ):
        file_name = "Compare_distance_and_angle.jpg"
        fig.savefig(file_name, dpi=200)
        plt.show()

def plotPoison(df):
    # THIS IS UNDER CONSTRUCTION
    goal_model_data = pd.DataFrame(columns=['team','opponent','goals', 'home'])
    counter = 0
    offset = len(df['match_id'].unique())

    for i, match in enumerate(reversed(shots_model)):
        counter += 1

        team1, team2 = match['teamsData'].keys()
    
        # Add the game from team1's perspective.
        index = i*2
        goal_model_data.at[index, 'team'] = match['Team']
        goal_model_data.at[index, 'opponent'] = match['Opponent']
        goal_model_data.at[index, 'goals'] = int(match['teamsData'][team1]['score'])
        goal_model_data.at[index, 'home'] = home_as_integer(match['teamsData'][team1]['side'])

        # Now add the same game from team2's perspective.
        index = index + 1
        goal_model_data.at[index, 'team'] = team_name_from_id(team2)
        goal_model_data.at[index, 'opponent'] = team_name_from_id(team1)
        goal_model_data.at[index, 'goals'] = int(match['teamsData'][team2]['score'])
        goal_model_data.at[index, 'home'] = home_as_integer(match['teamsData'][team2]['side'])


    # I haven't found out why this is needed, but if not forced to float, it is type object, and then
    # the model fitting doesn't work
    goal_model_data['goals'] = goal_model_data['goals'].astype(int)
    goal_model_data['home'] = goal_model_data['home'].astype(int)
    
    poisson_model = smf.glm(formula="Goal ~ Home + Team + Opponent", data=shots_model[['Goal','Home','Team','Opponent']], family=sm.families.Poisson()).fit()

    print(poisson_model.summary())        
    coeff = poisson_model.params      
    pritn(coeff)

    return()