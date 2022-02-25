#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 07:34:08 2021

Creator: Mauri Heinonen
Version: 0.1

This file includes all the extra functions that you need in the data library. You can make new functions in this file if it is necessary.
If you add some functions, you have to also comment on those functions like older functions are commented. Basically, on comments should be a short description about function, also you should make a description of all parameters and variables which one function return to the main program.
"""

import matplotlib.pyplot as plt
import base64
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.patches import Arc
from io import BytesIO
#from soccerplots.radar_chart import Radar
import statistics
import streamlit as st
import mysql.connector
from mysql.connector import errorcode
from pathlib import Path
import xG as xg
from datetime import datetime
import time
import matplotlib.patheffects as path_effects
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.cm as cm
from matplotlib.colors import Normalize


# Global variables for all functions
BALL_MISSED = 5
SCORING_CHANCE = 20
BALL_POSSESSION = 15
BALL_MISSED_COLOR = 'white'
SCORING_CHANCE_COLOR = 'blue'
BALL_POSSESSION_COLOR = 'black'
OTHER_COLOR = 'yellow'

def local_css(file_name):
    """
    This function will add css file to streamlit app.
    Write html-code to streamlit app.

    Parameters
    ----------
    file_name : string
        Path to file.

    Returns
    -------
    None.
    """
    with open(file_name) as file:
        st.markdown(
            f'<style type="text/css">{file.read()}</style>',
            unsafe_allow_html=True
        )



@st.cache(allow_output_mutation=True)
def get_db_connection():
    """
    This function makes a connection to the MySQL database. Basically, this starts MySQL connect.

    Returns
    -------
    Connection variable, what you can use to different database queries.
    """
    connect = ""
    try:
        connect = mysql.connector.connect( host='127.0.0.1', user='root', passwd='password', db='data_library', port=3306)
    except mysql.connector.Error as err:
        connect = ""
        
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
   
    return(connect)


def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def get_table_download_link(df, filename='', caption='Download csv file'):
    """
    Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">{caption}</a>'


def filedownload_csv(df, filename):
    """
    Make dataframe information to downloadable CSV file

    Parameters
    ----------
    df : DataFrame
        Dataframe which one you like to make downloadable CSV file.
    filename : String
        Downloadables file name. When the user download this file, this is the default filename.

    Returns
    -------
    href : String
        This function returns HTML link tag, which you can add to the page.
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}"><i class="fas fa-download"></i> Download table\'s information as a CSV file</a>'
    return(href)


def filedownload_image(img, filename):
    """
    Generates a link that allows the PIL image to be downloaded.

    Parameters
    ----------
    img : PIL image
        Image file which one you like to make downloadable Image file.
    filename : String
        Downloadables file name. When the user download this file, this is the default filename.

    Returns
    -------
    href : String
        This function returns HTML link tag, which you can add to the page.
    """
    imgdata = BytesIO()
    img.savefig(imgdata, format='png', dpi=200)
    imgdata.seek(0)  # rewind the data
    img_str = base64.b64encode(imgdata.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}"><i class="fas fa-download"></i> Download visualisation as a PNG file to your computer</a>'

    return href


def download_image(fig, filename, caption=''):
    """
    Generates a link that allows the PIL image to be downloaded.

    Parameters
    ----------
    img : PIL image
        Image file which one you like to make downloadable Image file.
    filename : String
        Downloadables file name. When the user download this file, this is the default filename.

    Returns
    -------
    href : String
        This function returns HTML link tag, which you can add to the page.
    """
    caption = "<i class='fas fa-download'></i> Download visualisation as a PNG file to your computer" if caption == '' else caption
    img_bytes = fig.to_image(format="png", scale=2)
    img_str = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}" class="download_link">{caption}</a>'

    return href


def show_image(fig):
    """
    Generates a link that allows the PIL image to be downloaded.

    Parameters
    ----------
    img : PIL image
        Image file which one you like to make downloadable Image file.
    filename : String
        Downloadables file name. When the user download this file, this is the default filename.

    Returns
    -------
    href : String
        This function returns HTML link tag, which you can add to the page.
    """
    #caption = "<i class='fas fa-download'></i> Download visualisation as a PNG file to your computer" if caption == '' else caption
    img_bytes = fig.to_image(format="png", scale=2)
    img_str = base64.b64encode(img_bytes).decode()
    #href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}" class="download_link">{caption}</a>'

    return f"data:file/jpg;base64,{img_str}"


def image_to_bytes(img_path, html=1):
    """
    This function makes PNG image to bytes, which we can show on HTML page

    Parameters
    ----------
    img_path : String
        Path to the image file which you want to show.

    Returns
    -------
    image_html : String
        HTML code which includes the image in base64 format.
    """
    
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    if html == 1:
        image_html = "<img src='data:image/png;base64,{}' style='display: block; text-align: center; width: 2em; margin: 5em auto;' />".format(encoded)
    else:
        image_html = f"data:image/png;base64,{encoded}"

    return image_html


def makeBasicQuery(sql, connect):
    """
    Basic function which makes sql querys

    Parameters
    ----------
    sql : String
        SQL query line.
    connect : database connection
        Database connection where you want to make the query.

    Returns
    -------
    rows : dataframe
        Returns queries rows.
    """
    rows = pd.read_sql( sql, connect )
    return(rows)


def checkCookie(name, db):
    user_logged_in = 0
    un = name[0]
    cookies = makeBasicQuery(f"SELECT * FROM session_cookie WHERE unicid = '{un}'", db)
    if len(cookies) > 0:
        logintime = int(cookies['logintime']) + 600
        current_time = int(time.time())

        if current_time < logintime:
            username = cookies['name'].values[0]
            user_rights = makeBasicQuery(f"SELECT * FROM users WHERE user_username = '{username}'", db)
            print(f"USER RIGHTS: {int(user_rights['user_role'].values[0])}")
            if ( int(user_rights['user_role'].values[0]) in [1, 2, 6] ):
                print(f"UN: {un}\nUSERNAME: {username}")
                makeBasicQuery("UPDATE session_cookie SET logintime = " + str(current_time) + " WHERE unicid = '" + un +"'", db)
                user_logged_in = 1

    return 0 #(user_logged_in)


@st.cache(allow_output_mutation=True)
def parse_dataframe(df, new_columns):
    """
    This function will parse data frame to a better format for the show on the web page.

    Parameters
    ----------
    df : dataframe
        Dataframe which function will parse to a new format.
    new_columns : dictionary
        Dictionary where is the old column name and new column name. All those columns which are not included in the dictionary will be deleted.

    Returns
    -------
    new_df : dataframe
        A parsed data frame, where columns are named by new names and useless columns are deleted.
    """
    new_df = pd.DataFrame()
    final_table_columns = new_columns.keys()
    new_df = df[df.columns.intersection(final_table_columns)]
    new_df = new_df.rename(columns=new_columns)
    return new_df


@st.cache(allow_output_mutation=True)
def get_match_info_to_df(df, events):
    """
    Add some information to dataframe about match stats

    Parameters
    ----------
    df : DataFrame
        Dataframe where add some new columns.
    events : DataFrame
        All matches events.

    Returns
    -------
    new_df : DataFrame
        DataFrame where is added few new columns.
    """
    new_df = pd.DataFrame()
    for i, row in df.iterrows():
        match_events = events.loc[(events['IdMatch'] == row['IdMatch'])]
        match_events.index = np.arange(0, len(match_events))

        ppda = count_ppda(match_events, [row['home_team'], row['away_team']])
        df_col = list(df.columns)
        df_col.remove('DateAndTime')

        new_df.at[i, 'DateAndTime'] = datetime.strptime(str(row['DateAndTime']), '%m/%d/%Y %H:%M').strftime("%d.%m.%Y")
        new_df.at[i, df_col] = row[df_col]
        new_df.at[i, 'Home Goals'] = int(len(match_events.loc[(((match_events.event == 'Goal') & (match_events.team == row['home_team'])) | ((match_events.event == 'Own Goal') & (match_events.team == row['away_id'])))]))
        new_df.at[i, 'Home xGsum'] = round(match_events[match_events.team == row['home_team']].sum()['xG'], 2)
        new_df.at[i, 'Home Pass'] = int(len(match_events.loc[((match_events.event == 'Pass') & (match_events.team == row['home_team']))]))
        new_df.at[i, 'Home Chance'] = int(len(match_events.loc[((match_events.event == 'Chance') & (match_events.team == row['home_team']))]))
        new_df.at[i, 'Home Shots'] = int(len(match_events.loc[((match_events.event.str.contains('Shot on target') | match_events.event.str.contains('Shot not on target')) & (match_events.team == row['home_team']))]))
        new_df.at[i, 'Home Shots-%'] = round(divZero(len(match_events.loc[((match_events.event == 'Shot on target') & (match_events.team == row['home_team']))]), new_df.at[i, 'Home Shots']) * 100  , 2)
        new_df.at[i, 'Home PPDA high'] = round(ppda[row['home_team']]['ppda_high'], 2)
        new_df.at[i, 'Home PPDA normal'] = round(ppda[row['home_team']]['ppda_norm'], 2)

        new_df.at[i, 'Away Goals'] = int(len(match_events.loc[(((match_events.event == 'Goal') & (match_events.team == row['away_team'])) | ((match_events.event == 'Own Goal') & (match_events.team == row['home_id'])))]))
        new_df.at[i, 'Away xGsum'] = round(match_events[match_events.team == row['away_team']].sum()['xG'], 2)
        new_df.at[i, 'Away Pass'] = int(len(match_events.loc[((match_events.event == 'Pass') & (match_events.team == row['away_team']))]))
        new_df.at[i, 'Away Chance'] = int(len(match_events.loc[((match_events.event == 'Chance') & (match_events.team == row['away_team']))]))
        new_df.at[i, 'Away Shots'] = int(len(match_events.loc[((match_events.event.str.contains('Shot on target') | match_events.event.str.contains('Shot not on target')) & (match_events.team == row['away_team']))]))
        new_df.at[i, 'Away Shots-%'] = round(divZero(len(match_events.loc[((match_events.event == 'Shot on target') & (match_events.team == row['away_team']))]), new_df.at[i, 'Away Shots']) * 100  , 2)
        new_df.at[i, 'Away PPDA high'] = round(ppda[row['away_team']]['ppda_high'], 2)
        new_df.at[i, 'Away PPDA normal'] = round(ppda[row['away_team']]['ppda_norm'], 2)

        new_df = new_df.replace(np.nan, 0, regex=True)

    return new_df 


@st.cache(allow_output_mutation=True)
def convert_coordinate(df, length=105, width=68):
    """
    This function will convert Stats coordinate to our format

    Parameters
    ----------
    df : dataframe
        Dataframe where same new columns will be added

    Returns
    -------
    returned_df : dataframe
        A parsed data frame, where is also player information and also team information.
    """
    with st.spinner('Loading data...'):
        half_length = length / 2
        half_width = width / 2

        df = df.replace(np.nan, '', regex=True)
        for column_name in ['x', 'x2', 'y', 'y2']:
            df[column_name] = pd.to_numeric(df[column_name])

        returned_df = pd.DataFrame(columns=df.columns)
        ration_x = 65 / 68

        for match_id in list(df.IdMatch.unique()):
            tme = df.loc[(df.IdMatch == match_id)]
            new_df = tme.copy()

            if match_id.isnumeric():
                # These coordinates are for old data, not from data, what we get through API gateway
                # All actions is set to attack direction is left to right

                new_df['x'] = tme['x'].apply(lambda x: x / 100 + half_length if x != "" else "")
                new_df['y'] = tme['y'].apply(lambda x: x / 100 + half_width if x != "" else "")
                new_df['x2'] = tme['x2'].apply(lambda x: x / 100 + half_length if x != "" else "")
                new_df['y2'] = tme['y2'].apply(lambda x: x / 100 + half_width if x != "" else "")

            # Set delta x and delta y value to dataframe
            new_df['dx'] = new_df['x2'] - new_df['x']
            new_df['dy'] = new_df['y2'] - new_df['y']

            # Convert all actions into half of the field for goalmouth pitch usage like shot map
            new_df['xh'] = new_df['y'].apply(lambda x: x * ration_x if x != "" else "")
            new_df['yh'] = new_df['x'].apply(lambda x: length - x if x != "" else "")
            new_df['xh2'] = new_df['y2'].apply(lambda x: x * ration_x if x != "" else "")
            new_df['yh2'] = new_df['x2'].apply(lambda x: length - x if x != "" else "")


            returned_df = returned_df.append(new_df)

        # Convert coordinates to numeric
        for column_name in ['x', 'x2', 'y', 'y2', 'dx', 'dy', 'xh', 'yh']:
            if column_name in returned_df.columns:
                returned_df[column_name] = pd.to_numeric(returned_df[column_name])

        return returned_df


def add_information_df(df, players):
    """
    This function will parse data frame to a better format for the show on the web page.

    Parameters
    ----------
    df : dataframe
        Dataframe where same new columns will be added
    players : dataframe
        DataFrame where is all players which are played in selected games

    Returns
    -------
    df : dataframe
        A parsed data frame, where is also player information and also team information.
    """
    with st.spinner('Loading data...'):
        # Add player information to dataframe
        for i, player in players.iterrows():
            name = f"{player['NickName']} {player['UsualFirstName']}" if player['UsualFirstName'] != '' else f"{player['NickName']}"
            df.loc[(df['id'] == player['IdActor']), 'PlayerName'] = name
            df.loc[(df['id'] == player['IdActor']), 'Number'] = player['JerseyNumber']
            df.loc[(df['id'] == player['IdActor']), 'Position'] = player['Position']
            df.loc[(df['id'] == player['IdActor']), 'Start11'] = player['IsStarter']

        max_df_index = df.index.max()
        df['outcome'] = 0

        df['prev_team'] = df['team'].shift(1)
        df['next_team'] = df['team'].shift(-1)
        df.loc[((df.prev_team == df.team) & (df.event.str.contains('Pass')) | (df.event.str.contains('Cross')) | (df.event.str.contains('Throw-in'))), 'outcome'] = 1
        # Add result of passes
        #for pass_index in df.loc[(df.event.str.contains('Pass')) | (df.event.str.contains('Cross')) | (df.event.str.contains('Throw-in'))].index:
            #next_index = pass_index + 1 if pass_index + 1 <= max_df_index else max_df_index
            #if df.at[pass_index, 'team'] == df.at[next_index, 'team']:
            #    df.at[pass_index, 'outcome'] = 1

        # Add result of shots
        for shot_index in df.loc[(df.event.str.contains('Shot on target'))].index:
            next_index = shot_index + 1 if shot_index + 1 <= max_df_index else max_df_index
            if df.at[next_index, 'event'] == 'Goal':
                df.at[shot_index, 'outcome'] = 1

        # Add xG value to shots in dataframe
        df = xg.xG(df, 0)

        return df


def divZero(a, b):
    """
    Function for division calculations. With this function, we can tackle the error message, if the divider is zero.

    Parameters
    ----------
    a : NUMBER (int or float)
        Number which we want divide.
    b : NUMBER (int or float)
        Number which are divider.

    Returns
    -------
    Result of division calculations in float numeric. If the divider is zero function will return a value 0.0.
    """
    if b == 0:
        return(0.0)
    else:
        return(a/b)


def resultOfShot(shot_index, events):
    """
    This function will return a boolean value if the shot goes to the goal.

    Parameters
    ----------
    shot_index : int
        Index of DataFrame row for the shot.
    events : DataFrame
        DataFrame where is all events.

    Returns
    -------
    result : Integer
        If a shot was goal returned value is 1
        If a shot was blocked returned values is 2
        If a shot was saved returned values is 3
        If a shot was own goal returned value is 4
        In other results returned value will be 0.

    """
    result = 0
    max_index = max(events.index.values.tolist())
    add_to_index =  max_index - shot_index if max_index <= shot_index + 2 else shot_index + 2
    for i in range(shot_index, add_to_index):
        row = events.iloc[i]
        if ( (row['event'] == 'Goal') | (row['event'] == 'Own Goal') ):
            result = 1
            break
        elif ( row['event'] == 'Block' ):
            result = 2
            break
        elif ( 'save' in row['event'] ):
            result = 3
            break
        elif ( row['event'] == 'Own Goal' ):
            result = 4
            break
        
    return result


def resultOfPass(pass_index, events, players):
    """
    This function will return a boolean value if the pass.

    Parameters
    ----------
    pass_index : int
        Index of DataFrame row for the pass.
    events : DataFrame
        DataFrame where is all events.
    players : DataFrame
        All players who are played in this match

    Returns
    -------
    result : Integer
        If a pass was passed to own player function will return True, in other cases it will return False
    """
    result = False
    max_index = max(events.index.values.tolist())
    next_index = max_index - pass_index if max_index <= pass_index + 1 else pass_index + 1
    pass_row = events.iloc[pass_index]
    passer_team = players[(players.IdActor == pass_row.id)]
    row = events.iloc[next_index]
    if row.id:
        reseaver_team = players[(players.IdActor == row.id)]

        if ( len(reseaver_team) > 0 ):
            if ( passer_team['idTeam'].isin(reseaver_team['idTeam']).values[0] ):
                result = True

    return result


def how_many_seconds_to_next_opp_action(df, row_index, myva=0):
    """
    This function will check how many seconds is between our team and the last opponent team action

    Parameters
    ----------
    df : DataFrame
        DataFrame where is all match events.
    row_index : int
        Integer variable, which tells, what is our team action row. We need this information for looking last opponent team action before our team actions.

    Returns
    -------
    time : int
        How many seconds is between our team and opponent actions.
    last_own_action : string
        What is our team last action before the opponent team action?
    """
    ms = 1000
    row = df.iloc[row_index]
    team = row['team']
    start_time = row['time']
    end_time = 0
    time = 0
    last_own_action = ""

    if myva == 0:
        if 'Shot' in row['event']:
            last_own_action = 'Shot'
            time = 10
        elif (row['event'] == 'Goal') or (row['event'] == 'Own Goal'):
            last_own_action = 'Goal'
            time = 10
        else:
            actions = df.loc[(df['time'] > start_time)]
            for i, action in actions.iterrows():
                if action['team'] == team:
                    end_time = action['time']
                    time = (end_time - start_time) / ms
                    last_own_action = action['event']
                else:
                    break
    else:
        if ('Shot' in row['Penetration']) or ('Shoot' in row['Penetration']):
            last_own_action = 'Shot'
            time = 10
        elif 'Goal' in row['Result']:
            last_own_action = 'Goal'
            time = 10
        else:
            actions = df.loc[(df['time'] > start_time)]
            for i, action in actions.iterrows():
                if action['team'] == team:
                    end_time = action['time']
                    time = (end_time - start_time) / ms
                    last_own_action = action['action']
                else:
                    break

    return {'time': time, 'action': last_own_action}


def what_color_action(value={'time': 0, 'action': ''}, ball_missed=5, scoring_chance=20, ball_possession=15):
    """
    This function will tell the right colour of plotline

    Parameters
    ----------
    value : dict
        Dictionary where is key time, what tells how many seconds is between actions. Second key is actions which will tell what is the last action which is involved

    Returns
    -------
    color : string
        Tells the right colour of plotline
    """
    color = OTHER_COLOR
    if value['time'] >= ball_possession:
        color = BALL_POSSESSION_COLOR
    elif (value['time'] <= scoring_chance) and (('Shot' in value['action']) or ('Goal' in value['action'])):
        color = SCORING_CHANCE_COLOR
    elif (value['time'] <= ball_missed) and (('Shot' not in value['action']) or ('Goal' not in value['action'])):
        color = BALL_MISSED_COLOR

    return color


def plotAttackFreeKicks(df, team, opp_team, caption='', subcaption='', height=50, width=65, linecolor='white', pitch='#80B860', BALL_MISSED=5, SCORING_CHANCE=20, BALL_POSSESSION=15, info_text=0):
    """
    plot in the half field all team free kick spots.

    Parameters
    ----------
    df : DataFrame
        DataFrame where is all actions of the match
    team : string
        Team name which free kicks will be plotted
    opp_team : string
        Opponent team name
    caption : string
        Plot caption
    subcaption : string
        Plot subcaption
    height : int
        Field height
    width : int
        Field width
    linecolor : string
        Field line color
    pitch : string
        Field background color
    BALL_MISSED : int
        How many seconds team is missed the ball
    SCORING_CHANCE : int
        How many seconds team is maked the scoring chance
    BALL_POSSESSION : int
        How many seconds team is keept ball possession
    info_text : int
        Show info text in image (1), don't show info text in image (0)

    Returns
    -------
    fig : figure
        Matplotlib figure.
    ax : axes
        Matplotlib axes.
    """

    (fig, ax) = createGoalMouth(height, width, linecolor, pitch)
    if ( team == 'Opponent' ):
        set_pieces = df.index[(df['event'].str.contains('free-kick')) & (df['team'] == opp_team)].tolist()
    else:
        set_pieces = df.index[(df['event'].str.contains('free-kick')) & (df['team'] != team)].tolist()
    
    for pbl_index in set_pieces:
        pbl_index += 1
        row = df.iloc[pbl_index]
        if ( not row['X_half'] ):
            continue
        
        if ( len(df) -1 == pbl_index ):
            action_color='white'
        else:
            result = resultOfShot(pbl_index + 1, df)
            if ( result != '' ):
                if ( result > 0 ):
                    action_color = 'blue'
                else:
                    action_color = 'white'
            else:
                (sec, action) = howManySecondsToNextOppAction(df, pbl_index + 1)
                action_color = whatColorAction(sec, action, BALL_MISSED, SCORING_CHANCE, BALL_POSSESSION)
        
        if ( row['X_half'] and row['Y_half'] ):
            plt.scatter(row['X_half'],row['Y_half'], s=50, edgecolors='white', color=action_color, alpha=0.5, zorder=5)  
            #if ( row['x_tb_dest'] and row['y_tb_dest'] ):
            #    dx = row['x_tb_dest'] - row['x_tb']
            #    dy = row['y_tb_dest'] - row['y_tb']
            #    plt.arrow(row['x_tb'], row['y_tb'], dx, dy, head_width=1, head_length=1, color=action_color, alpha=0.5, length_includes_head=True, zorder=5)
    
    if ( caption ):
        plt.text(0.05, height + 5, "{}".format(caption), fontsize=13, fontweight="bold", ha="left", va="center", color=linecolor)
    if ( subcaption ):
        plt.text(0.05, height + 2, "{}".format(subcaption), fontsize=7, fontweight="normal", ha="left", va="center", color=linecolor)
        
    # Draw info box
    if ( info_text == 1 ):
        plt.text(0,-1, 'Scoring chance did not created', color='white', fontsize=7, va='center', ha='left', zorder=8)
        plt.text(width,-1, 'Scoring chance', color=SCORING_CHANCE_COLOR, fontsize=7, va='center', ha='right', zorder=8)

    fig.set_size_inches(width/10, height/10)
    fig.patch.set_facecolor(pitch)
    plt.tight_layout()
    return(fig, ax)


#@st.cache(allow_output_mutation=True, hash_funcs={mysql.connector.connection_cext.CMySQLConnection: id})
def getxT(coordinate, xt_df):
    """
    Calculate xT in specific x,y coordinate

    Parameters
    ----------
    coordinate : list
        List where first value is x -coordinate and second is y -coordinate.
    db : Database connection
        Database connection for queries.

    Returns
    -------
    xT : int
        Specific coordinate xT value
    """
    
    # Check if this is possible make with out for loop or row by row
    
    
    if ( coordinate[0] == ''):
        return(0)
    
    #xT_df = makeBasicQuery("SELECT * FROM xT", db)
    
    row = round(coordinate[1] / (68 / 13), 0)
    col = round(coordinate[0] / (105 / 21), 0)
    if ( row == 0 ):
        row += 1
        
    if ( col == 0 ):
        col += 1

    xT = xt_df.loc[((xt_df.row == row) & (xt_df.col == col))]
    if ( len(xT)  > 0 ):
        xT = xT['value'].values[0]
    else:
        xT = 0
    return(xT)


#@st.cache(allow_output_mutation=True, hash_funcs={mysql.connector.connection_cext.CMySQLConnection: id})
def getxA(from_coordinate, to_coordinate, db):
    """
    Calculate xA in specific x,y coordinate

    Parameters
    ----------
    from_coordinate : list
        Where pass is given.
        List where first value is x -coordinate and second is y -coordinate.
    to_coordinate : list
        Where pass is received.
        List where first value is x -coordinate and second is y -coordinate.
    db : Database connection
        Database connection for queries.

    Returns
    -------
    xA : int
        Specific coordinate xA value
    """
    
    # Check if this is possible make with out for loop or row by row
        
    from_xT = getxT(from_coordinate, db)
    to_xT = getxT(to_coordinate, db)
    
    xA = 0
    if ( to_xT > 0 ):
        delta_xT = to_xT - from_xT
        if ( delta_xT > 0 ):
            xA = delta_xT
            
    return(xA)


def count_ball_possession_info(df, teams=[]):
    start_time = 0
    bp_section = {}
    for team in teams:
        bp_section[team] = 1

    step = 0
    ball_possession = []

    for ind, row in df.iterrows():
        if ind == 0 or row.prev_team == '':
            continue

        # This action is for same team than previous action
        if row.team == row.prev_team:
            # Add one step for telling how many actions have been able to be done before the opponent gets the ball
            step += 1
        else:
            # This action is for another team than the previous action, so add a ball possession section for the
            # previous team. And start new ball possession section
            bp_time = round((row.time - start_time) / 1000, 2)
            if bp_time < 0:
                bp_time = round((row.time - 0) / 1000, 2)

            if step <= 1:
                continue


            add_to_team = row.prev_team
            if 'Opponents' in teams:
                if teams[0] != row.prev_team:
                    add_to_team = 'Opponents'

            ball_possession.append({'section': bp_section[add_to_team], 'steps': step, 'time': bp_time, 'team': add_to_team})
            bp_section[add_to_team] += 1
            start_time = row.time
            step = 0

    return pd.DataFrame(ball_possession)


def count_ppda(df, teams=[]):
    # PPDA = Number of Passes made by Attacking Team (opponent) / Number of Defensive Actions
    # Both values (passes made and defensive actions) will be calculated in opponent’s final 60% of the pitch.
    # The most common myth about the PPDA is that it displays the quality of the pressing. However, this assumption is untrue, the PPDA does not make any statement about the quality of the pressing.
    # Where Defensive Actions are:
    # possession-winning duels, tackles, interceptions, fouls

    # Count starting point to 40%
    x = 105 * 0.4

    ppda = {}
    # Count how many passes team is made lower than 40% of field and higher that
    count = 0
    for team in teams:
        if team == 'Opponent':
            own_team = teams[0] if count == 1 else teams[1]
            pass_count_high = len(df.loc[(((df.x <= x) & (df.x2 <= x)) & (df.event == 'Pass') & (df.team == own_team))])
            changes_high = len(df.loc[(((df.team != own_team) & (df.next_team != df.team)) & ((df.x <= x) & (df.x2 <= x)))])
            pass_count_norm = len(df.loc[(((df.x > x) & (df.x2 > x)) & (df.event == 'Pass') & (df.team == own_team))])
            changes_norm = len(df.loc[(((df.team != own_team) & (df.next_team != df.team)) & ((df.x > x) & (df.x2 > x)))])
        else:
            pass_count_high = len(df.loc[(((df.x <= x) & (df.x2 <= x)) & (df.event == 'Pass') & (df.team != team))])
            changes_high = len(df.loc[(((df.team == team) & (df.next_team != df.team)) & ((df.x <= x) & (df.x2 <= x)))])
            pass_count_norm = len(df.loc[(((df.x > x) & (df.x2 > x)) & (df.event == 'Pass') & (df.team != team))])
            changes_norm = len(df.loc[(((df.team == team) & (df.next_team != df.team)) & ((df.x > x) & (df.x2 > x)))])

        ppda[team] = {
            'pass_count_high': pass_count_high, 'changes_high': changes_high, 'ppda_high': pass_count_high / changes_high,
            'pass_count_norm': pass_count_norm, 'changes_norm': changes_norm, 'ppda_norm': pass_count_norm / changes_norm
        }
        count += 1

    return ppda


def make_info_boxes(parsed_df, events, collection=''):
    my_cmap = cm.get_cmap('jet')
    if collection == '':
        unique_teams = sorted(list(filter(None, events.team.unique())))
        my_norm = Normalize(vmin=0, vmax=len(unique_teams))
        number_of_columns = len(unique_teams)
    else:
        my_norm = Normalize(vmin=0, vmax=2)
        number_of_columns = 2
        unique_teams = [collection, 'Opponents']

    if number_of_columns > 4:
        number_of_columns = 4

    ball_possession = count_ball_possession_info(events, unique_teams)

    stats_info = st.beta_columns(number_of_columns)
    info = {}
    datas = []
    for stat_col in np.arange(0, len(unique_teams)):
        div_team = unique_teams[stat_col]

        div_col = stat_col if stat_col < number_of_columns else stat_col % number_of_columns

        color = my_cmap(my_norm(stat_col))
        dark_background_color = f"{color[0]*255}, {color[1]*255}, {color[2]*255}, {color[3]}"
        light_background_color = f"{color[0]*255}, {color[1]*255}, {color[2]*255}, 0.3"

        if unique_teams[stat_col] != 'Opponents':
            number_of_team_matches = len(parsed_df.loc[(parsed_df['Home team'] == unique_teams[stat_col])]) + len(parsed_df.loc[(parsed_df['Away team'] == unique_teams[stat_col])])
            info[div_team] = {'match': len(parsed_df.loc[(parsed_df['Home team'] == unique_teams[stat_col])]) + len(parsed_df.loc[(parsed_df['Away team'] == unique_teams[stat_col])]),
                              'goals': parsed_df.loc[(parsed_df['Home team'] == unique_teams[stat_col])]['Home Goals'].sum() + parsed_df.loc[(parsed_df['Away team'] == unique_teams[stat_col])]['Away Goals'].sum(),
                              'passes': (parsed_df.loc[(parsed_df['Home team'] == unique_teams[stat_col])]['Home Pass'].sum() + parsed_df.loc[(parsed_df['Away team'] == unique_teams[stat_col])]['Away Pass'].sum()) / number_of_team_matches,
                              'shots': (parsed_df.loc[(parsed_df['Home team'] == unique_teams[stat_col])]['Home Shots'].sum() + parsed_df.loc[(parsed_df['Away team'] == unique_teams[stat_col])]['Away Shots'].sum()) / number_of_team_matches,
                              'avg_xg': (parsed_df.loc[(parsed_df['Home team'] == unique_teams[stat_col])]['Home xGsum'].sum() + parsed_df.loc[(parsed_df['Away team'] == unique_teams[stat_col])]['Away xGsum'].sum()) / number_of_team_matches,
                              'won': len(parsed_df.loc[(parsed_df['Home team'] == unique_teams[stat_col]) & (parsed_df['Home Goals'] > parsed_df['Away Goals'])]) + len(parsed_df.loc[(parsed_df['Away team'] == unique_teams[stat_col]) & (parsed_df['Home Goals'] < parsed_df['Away Goals'])]),
                              'lost': len(parsed_df.loc[(parsed_df['Home team'] == unique_teams[stat_col]) & (parsed_df['Home Goals'] < parsed_df['Away Goals'])]) + len(parsed_df.loc[(parsed_df['Away team'] == unique_teams[stat_col]) & (parsed_df['Home Goals'] > parsed_df['Away Goals'])]),
                              'PPDA high': parsed_df.loc[(parsed_df['Home team'] == unique_teams[stat_col])]['Home PPDA high'].sum() + parsed_df.loc[(parsed_df['Away team'] == unique_teams[stat_col])]['Away PPDA high'].sum(),
                              #'PPDA mid/low': parsed_df.loc[(parsed_df['Home team'] == unique_teams[0]) & (parsed_df['Home PPDA mid/low'])],
                              }
        else:
            number_of_team_matches = len(parsed_df.loc[(parsed_df['Home team'] != unique_teams[0])]) + len(parsed_df.loc[(parsed_df['Away team'] != unique_teams[0])])
            info[div_team] = {'match': len(parsed_df.loc[(parsed_df['Home team'] != unique_teams[0])]) + len(parsed_df.loc[(parsed_df['Away team'] != unique_teams[0])]),
                              'goals': parsed_df.loc[(parsed_df['Home team'] != unique_teams[0])]['Home Goals'].sum() + parsed_df.loc[(parsed_df['Away team'] != unique_teams[0])]['Away Goals'].sum(),
                              'passes': (parsed_df.loc[(parsed_df['Home team'] != unique_teams[0])]['Home Pass'].sum() + parsed_df.loc[(parsed_df['Away team'] != unique_teams[0])]['Away Pass'].sum()) / number_of_team_matches,
                              'shots': (parsed_df.loc[(parsed_df['Home team'] != unique_teams[0])]['Home Shots'].sum() + parsed_df.loc[(parsed_df['Away team'] != unique_teams[0])]['Away Shots'].sum()) / number_of_team_matches,
                              'avg_xg': (parsed_df.loc[(parsed_df['Home team'] != unique_teams[0])]['Home xGsum'].sum() + parsed_df.loc[(parsed_df['Away team'] != unique_teams[0])]['Away xGsum'].sum()) / number_of_team_matches,
                              'won': len(parsed_df.loc[(parsed_df['Home team'] != unique_teams[0]) & (parsed_df['Home Goals'] > parsed_df['Away Goals'])]) + len(parsed_df.loc[(parsed_df['Away team'] != unique_teams[0]) & (parsed_df['Home Goals'] < parsed_df['Away Goals'])]),
                              'lost': len(parsed_df.loc[(parsed_df['Home team'] != unique_teams[0]) & (parsed_df['Home Goals'] < parsed_df['Away Goals'])]) + len(parsed_df.loc[(parsed_df['Away team'] != unique_teams[0]) & (parsed_df['Home Goals'] > parsed_df['Away Goals'])]),
                              'PPDA high': parsed_df.loc[(parsed_df['Home team'] != unique_teams[0])]['Home PPDA high'].sum() + parsed_df.loc[(parsed_df['Away team'] != unique_teams[0])]['Away PPDA high'].sum(),
                              #'PPDA mid/low': parsed_df.loc[(parsed_df['Home team'] != unique_teams[0]) & (parsed_df['Home PPDA mid/low'])],
                              }

        tie = info[div_team]['match'] - info[div_team]['won'] - info[div_team]['lost']

        grid_template_columns = 'grid-template-columns:'
        won_bar = f"<div id='won'>{info[div_team]['won']}</div>" if info[div_team]['won'] > 0 else ""
        grid_template_columns += f" {round(info[div_team]['won'] / info[div_team]['match'] * 100, 2)}%" if info[div_team]['won'] > 0 else ""

        tie_bar = f"<div id='tie'>{tie}</div>" if tie > 0 else ""
        grid_template_columns += f" {round(tie / info[div_team]['match'] * 100, 2)}%" if tie > 0 else ""

        lost_bar = f"<div id='lost'>{info[div_team]['lost']}</div>" if info[div_team]['lost'] > 0 else ""
        grid_template_columns += f" {round(info[div_team]['lost'] / info[div_team]['match'] * 100, 2)}%" if info[div_team]['lost'] > 0 else ""

        ball_possession_avg = ball_possession[(ball_possession.team == div_team)]['time'].mean()
        ball_possession_sum = ball_possession[(ball_possession.team == div_team)]['time'].sum()

        stats_info[div_col].markdown(
            f"<div style='border: 1px solid rgba({dark_background_color}); background-color: rgba({light_background_color});' class='stats_div'>"
            f"<div style='background-color: rgba({dark_background_color});' class='stats_goals'>{int(info[div_team]['goals'])}</div>"
            f"<div class='stats_header'>{unique_teams[stat_col]}</div>"
            f"<div id='stats_text'>"
                f"Stats from {info[div_team]['match']} games<br />"
                f"{round(info[div_team]['goals'] / info[div_team]['match'], 2)} goals / game<br />"
                f"{round(int(info[div_team]['goals']) / info[div_team]['shots'], 2)} goals / shots<br />"
                f"{round(info[div_team]['avg_xg'], 2)} xG / game<br />"
                f"{round(info[div_team]['passes'], 2)} passes / game<br />"
                f"{round(info[div_team]['shots'], 2)} shots / game<br />"
                f"{round(info[div_team]['passes'] / (ball_possession_sum / 60), 2)} passes per ball possession minutes<br />"
                f"{round(info[div_team]['passes'] / ball_possession.loc[(ball_possession.team == div_team)]['section'].max(), 2)} passes per ball possession section<br />"
                f"PPDA per match: {round(info[div_team]['PPDA high'] / info[div_team]['match'], 2)}<br />"
                f"Average ball possession time is {round(ball_possession_avg / info[div_team]['match'], 2)} seconds.<br />"
                f"<p>&nbsp;</p>"
            f"</div>"
            f"<div style='margin-top: 2rem;'><b>Number of won/drawn/lost from selected games</div>"
            f"<div id='bar' style='{grid_template_columns};'>{won_bar}{tie_bar}{lost_bar}</div>"
            f"</div>", unsafe_allow_html=True)

        theta_values = ['goals / game',
                        'goals / shots',
                        'xG / game',
                        'shots / game',
                        'passes / minutes',
                        'passes / possession section',
                        'avg possession time',
                        'goals / game']

        r_values = [round(info[div_team]['goals'] / info[div_team]['match'], 2),
                    round(info[div_team]['goals'] / info[div_team]['shots'], 2),
                    round(info[div_team]['avg_xg'], 2),
                    round(info[div_team]['shots'], 2),
                    round(info[div_team]['passes'] / (ball_possession_sum / 60), 2),
                    round(info[div_team]['passes'] / ball_possession.loc[(ball_possession.team == div_team)]['section'].max(), 2),
                    round(ball_possession_avg, 2),
                    round(info[div_team]['goals'] / info[div_team]['match'], 2)]

        datas.append(go.Scatterpolar(
            name=div_team,
            fill='toself',
            r=r_values,
            theta=theta_values,
            hovertemplate='%{text}',
            text=['<b>Basic information</b><br /><i>{}</i> {}'.format(theta_values[i], r_values[i]) for i in np.arange(0, len(r_values))],
        ))


    fig = go.Figure(data=datas)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        template=None,
        polar=dict(
            radialaxis=dict(
                visible=True
            ),
            angularaxis=dict(
                direction="clockwise"
            )
        ),
        showlegend=True
    )

    st.plotly_chart(fig, config={'displayModeBar': False})

    # < div style = 'background-image: url({radar_image}); background-repeat: no-repeat; background-position: left top; background-size: 100% 100%;' > < / div >

@st.cache(allow_output_mutation=True)
def make_player_based_stats(df, xt_df):
    """
    This function will make a data frame where is basic information per player who is played in the selected match.

    Parameters
    ----------
    df : DataFrame
        Includes those match ids, where we want to get player-based stats. In this data frame should be a column named IdMatch.

    Returns
    -------
    df_stats : DataFrame
        Returns dataframe where is player based basic statistics like how many shots this player is made during the game. If there are multiple games selected, all values are the averages of those games..
    """

    df_stats = pd.DataFrame(columns=['PlayerId', 'Player', 'Number', 'Team', 'Position', 'Games', 'Passes', 'Pass-%', 'Shots', 'Shot-%', 'Goals', 'xGsum', 'xTsum', 'xAsum', 'Duels', 'Yellows', 'Reds'])
    df_stats.set_index('PlayerId', inplace=True)

    match_players = list(df.id.unique())

    # go throw players and set extra information for players
    for player in filter(None, match_players):
        player_rows = df.loc[(df.id == player)]

        # Count how many matches are the selected player played in the data set
        number_of_matches = len(df.loc[(df.id == player)]['IdMatch'].unique())

        # Get some basic information from the player
        player_all_passes = player_rows[((player_rows.event.str.contains('Pass')) | (player_rows.event.str.contains('Cross')))]
        player_shots = player_rows[(player_rows['event'].str.contains('Shot'))]
        shots_on_target = len(player_rows[(player_rows['event'].str.contains('Shot on target'))])
        player_goals = player_rows[(player_rows.event == 'Goal')]
        duels = player_rows[(player_rows.duel != "None")]

        xt = 0
        xa = 0
        for ind, shot in player_shots.iterrows():
            xt += getxT([shot['x'], shot['y']], xt_df)
            #from_coordinate = df.iloc[ind-1]
            #xa += getxA([from_coordinate['x'], from_coordinate['y']], [coordinate['x'], coordinate['y']], db)

        sum_of_xg = 0
        number_of_shots = len(player_shots)
        if number_of_shots > 0:
            sum_of_xg = sum(player_shots.xG)
            
        success_passes = 0
        number_of_passes = len(player_all_passes)
        number_of_goals = len(player_goals)
        if number_of_passes > 0:
            success_passes = len(player_all_passes.loc[(player_all_passes.outcome == 1)])

        for ind, passes in player_all_passes.iterrows():
            if passes.outcome == 1:
                xt += getxT([passes['x'], passes['y']], xt_df)
                xa += getxA([passes['x'], passes['y']], [passes['x2'], passes['y2']], xt_df)

        df_stats.at[player, 'Player'] = player_rows.PlayerName.unique()[0]
        df_stats.at[player, 'Team'] = player_rows.team.unique()[0]
        df_stats.at[player, 'Position'] = player_rows.Position.unique()[0]
        df_stats.at[player, 'Number'] = int(player_rows.Number.unique()[0])
        df_stats.at[player, 'Games'] = number_of_matches
        df_stats.at[player, 'Passes'] = round(number_of_passes / number_of_matches, 2)
        df_stats.at[player, 'Pass-%'] = round(divZero(success_passes, number_of_passes) * 100, 2)
        df_stats.at[player, 'Shots'] = round(number_of_shots / number_of_matches, 2)
        df_stats.at[player, 'Shot-%'] = round(divZero(shots_on_target, number_of_shots) * 100, 2)
        df_stats.at[player, 'Goals'] = number_of_goals
        df_stats.at[player, 'xGsum'] = round(sum_of_xg / number_of_matches, 2)
        df_stats.at[player, 'xTsum'] = round(xt, 2)
        df_stats.at[player, 'xAsum'] = round(xa, 2)
        df_stats.at[player, 'Duels'] = round(len(duels) / number_of_matches, 2)
        df_stats.at[player, 'Yellows'] = round(len(player_rows[(player_rows.event == 'Yellow card')]) / number_of_matches, 2)
        df_stats.at[player, 'Reds'] = round(len(player_rows[(player_rows.event == 'Red card')]) / number_of_matches, 2)

    df_stats = df_stats.sort_values(by=['Team', 'Number'])
    return df_stats


def show_basic_stats(df, events, team='', filename='download.xlsx', caption='Download XLSX file to your computer'):
    if df.empty:
        st.error("You have to select basic stats!<br />Function needs next variables: df, events, team, filename, caption.")
        return

    make_info_boxes(df, events, team)
    st.write(' ')
    st.dataframe(df)
    st.markdown("<div style='width: 100%; text-align: center; margin-top: 0;'>" + get_table_download_link(df, filename, f"<i class='fas fa-download'></i> {caption}") + "</div>", unsafe_allow_html=True)


def show_player_based_stats(events, filename, db):
    if events.empty:
        st.error("You have to select matches!<br />Function needs next variables: events, filename.")
        return

    xt_df = makeBasicQuery("SELECT * FROM xT", db)
    player_stats = make_player_based_stats(events, xt_df)
    st.markdown(f"<p class='paragraph'>This table is presented information for {len(player_stats)} players from selected matches.</p>", unsafe_allow_html=True)
    st.dataframe(player_stats)
    st.markdown(filedownload_csv(player_stats, filename),unsafe_allow_html=True)


def show_set_piece_form(events, team, orig, column=st, goal_kick_index=[], corner_index=[]):
    if events.empty:
        st.error("You have to select matches!<br />Function needs next variables: events, filename.")
        return pd.DataFrame(), '', '', ''

    free_kick = column.radio('Select set pieces type', ['Free kicks', 'Corners', 'Goal kicks', 'Throw ins'], 0, key=f"Free_kick_{team}")
    arrow = column.radio('Show destination', ['No', 'Yes'], 0, key=f"Arrow_{team}")

    if free_kick == 'Free kicks':
        set_pieces = events.loc[(events.event == 'Pass (DB)')]
        res = filter(lambda i: i in list(set_pieces.index), goal_kick_index)
        set_pieces = set_pieces.drop(res, inplace=False)
        res = filter(lambda i: i in list(set_pieces.index), corner_index)
        set_pieces = set_pieces.drop(res, inplace=False)

        set_pieces = set_pieces.loc[(set_pieces.team == team)] if team != 'Opponent' else set_pieces.loc[(set_pieces.team != orig)]
    elif free_kick == 'Corners':
        side = column.radio('Select side of corner', ['Both', 'Left', 'Right'], 0, key=f"{column}_side_of_corner")
        if side == 'Left':
            set_pieces = events.iloc[corner_index]
            set_pieces = set_pieces.loc[((set_pieces.team == team) & (events.y > 40))] if team != 'Opponent' else set_pieces.loc[((set_pieces.team != orig) & (events.y > 40))]
        elif side == 'Right':
            set_pieces = events.iloc[corner_index]
            set_pieces = set_pieces.loc[((set_pieces.team == team) & (events.y < 10))] if team != 'Opponent' else set_pieces.loc[((set_pieces.team != orig) & (events.y < 10))]
        else:
            set_pieces = events.iloc[corner_index]
            set_pieces = set_pieces.loc[(set_pieces.team == team)] if team != 'Opponent' else set_pieces.loc[(set_pieces.team != orig)]
    elif free_kick == 'Goal kicks':
        set_pieces = events.iloc[goal_kick_index]
        set_pieces = set_pieces.loc[(set_pieces.team == team)] if team != 'Opponent' else set_pieces.loc[(set_pieces.team != orig)]
    elif free_kick == 'Throw ins':
        set_pieces = events.loc[((events.event == 'Throw-in') & (events.team == team))] if team != 'Opponent' else events.loc[((events.event == 'Throw-in') & (events.team != team))]

    set_pieces = set_pieces.drop(set_pieces.loc[(set_pieces.event == 'End of Half')].index, inplace=False)

    sorted = set_pieces.drop_duplicates(subset=['PlayerName', 'Number', 'Position'])
    sorted = sorted.sort_values(by=['Number'], ascending=True)
    set_piece_player = sorted.to_dict('records')
    set_piece_player.insert(0, {'Number': '', 'PlayerName': '', 'Position': ''})

    player_named = column.selectbox('Show free kicks what is given by the player', set_piece_player, format_func=lambda opt: f"#{int(opt['Number'])} {opt['PlayerName']} ({opt['Position']})" if opt['PlayerName'] != '' else '', key=f"given_by_player_{team}")
    if player_named['PlayerName'] != '':
        set_pieces = set_pieces.loc[(set_pieces.PlayerName == player_named['PlayerName'])]

    return set_pieces, free_kick, arrow, player_named


def make_set_piece_information(df, team=''):
    # First filtering only set pieces
    goal_kick_index = list(df.loc[(df.event == 'Out for goal kick')].index)
    goal_kick_index = list(np.asarray(goal_kick_index) + 1)

    corner_index = list(df.loc[(df.event == 'Out for corner')].index)
    corner_index = list(np.asarray(corner_index) + 1)

    set_pieces = df.loc[(df.event_category.str.contains('Set Play'))]

    set_pieces_info = pd.DataFrame(columns=['Player', 'Team', 'Set Piece', 'Count', 'Lane'])
    index = 0
    for spi, row in set_pieces.iterrows():
        if spi in goal_kick_index:
            set_piece_type = 'Goal Kick'
        elif spi in corner_index:
            set_piece_type = 'Corner'
        elif row.event == 'Throw-in':
            set_piece_type = 'Throw in'
        elif 'Penalty' in row.event:
            set_piece_type = 'Penalty'
        else:
            set_piece_type = 'Free kick'

        lane_width = 68  / 3
        if row.y <= lane_width:
            lane = 'Right'
        elif (row.y > lane_width) and (row.y <= lane_width * 2):
            lane = 'Center'
        else:
            lane = 'Left'

        set_pieces_info.at[index, 'Player'] = row['PlayerName']
        if team == '':
            set_pieces_info.at[index, 'Team'] = row['team']
        else:
            set_pieces_info.at[index, 'Team'] = row['team'] if row['team'] == team else 'Opponent'

        set_pieces_info.at[index, 'Set Piece'] = set_piece_type
        set_pieces_info.at[index, 'Lane'] = lane

        index += 1

    grouped = set_pieces_info.groupby(['Player', 'Team', 'Set Piece', 'Lane'])['Count'].size().reset_index()

    return grouped