#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 4.5.2021

@author: Mauri Heinonen
Version: 1.0
"""
import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import statistics
from ExtraFunctions import *
import pandas as pd
import plotly.express as px


@st.cache(allow_output_mutation=True)
def ellipse_arc(x_center=0, y_center=0, a=1, b=1, start_angle=0, end_angle=2*np.pi, n=100, closed=False):
    t = np.linspace(start_angle, end_angle, n)
    x = x_center + a*np.cos(t)
    y = y_center + b*np.sin(t)
    path = f'M {x[0]}, {y[0]}'
    for k in range(1, len(t)):
        path += f'L{x[k]}, {y[k]}'
    if closed:
        path += ' Z'
    return path


@st.cache(allow_output_mutation=True)
def draw_full_pitch(length=105, width=68, linecolor='white', pitch='#80B860'):
    """
    Creates a plot of the pitch, which size is given with length and width.

    Parameters
    ----------
    length : int
        The 'length' is the length of the pitch (goal to goal).
        Default value is 105. 68
    width : int
        The 'width' is the width of the pitch (sideline to sideline).
        Default value is 68.
    linecolor : string
        Colors of field lines, you have to give it in hex code.
        Default values is white (#ffffff).
    pitch : string
        Field color, you have to give it in hex code.
        Default values is green (#).

    Returns
    -------
    field : list
        Array where is all field notations.
    """
    penalty_area = 16.5
    gk_box = 5.5
    goal = 7.32
    circle_radius = 9.15
    penalty_spot = 11
    spot = 0.5
    field = []

    if length <= 95:
        return(str("Lengt has to be atleast 95?"))
    elif ( (length >= 131) or (width >= 101) ):
        return(str("Field dimensions are too big. Maximum length is 130, maximum width is 100"))
    else:
        # Pitch Outline & Centre Line
        field.append({'type': 'rect', 'xref': 'x','yref': 'y','x0': 0,'y0': 0,'x1': length,'y1': width,'line_color': linecolor,'line_width': 1})
        # Center line
        field.append({'type': 'line', 'xref': 'x','yref': 'y','x0': length/2, 'y0': 0, 'x1': length/2,'y1': width,'line_color': linecolor,'line_width': 1})
        # center spot
        field.append({'type': 'circle', 'xref': 'x', 'yref': 'y', 'x0': length / 2 - circle_radius, 'y0': width / 2 - circle_radius, 'x1': length / 2 + circle_radius, 'y1': width / 2 + circle_radius, 'line_color': linecolor, 'line_width': 1})
        # Center circle
        field.append({'type': 'circle','xref': 'x','yref': 'y','x0':length/2-spot, 'y0':width/2-spot,'x1':length/2+spot,'y1':width/2+spot,'line_color': linecolor,'line_width': 1,'fillcolor': linecolor})
        # Left Penalty Area
        field.append({'type': 'rect', 'xref': 'x','yref': 'y','x0': 0, 'y0': width/2 - penalty_area,'x1': penalty_area,'y1': width/2 + penalty_area,'line_color': linecolor,'line_width': 1})
        # Right Penalty Area
        field.append({'type': 'rect', 'xref': 'x', 'yref': 'y', 'x0': length, 'y0': width / 2 - penalty_area, 'x1': length - penalty_area,'y1': width / 2 + penalty_area, 'line_color': linecolor, 'line_width': 1})
        # Left GK Area
        field.append({'type': 'rect', 'xref': 'x','yref': 'y','x0': 0, 'y0': width/2 - goal / 2 - gk_box,'x1': gk_box,'y1': width/2 + goal / 2 + gk_box,'line_color': linecolor,'line_width': 1})
        # Right GK Area
        field.append({'type': 'rect', 'xref': 'x', 'yref': 'y', 'x0': length, 'y0': width / 2 - goal / 2 - gk_box, 'x1': length - gk_box,'y1': width / 2 + goal / 2 + gk_box, 'line_color': linecolor, 'line_width': 1})
        # Left Penalty spot
        field.append({'type': 'circle', 'xref': 'x', 'yref': 'y', 'x0': penalty_spot - spot, 'y0': width / 2 - spot, 'x1': penalty_spot + spot, 'y1': width / 2 + spot, 'line_color': linecolor, 'line_width': 1, 'fillcolor': linecolor})
        # Right Penalty spot
        field.append({'type': 'circle', 'xref': 'x', 'yref': 'y', 'x0': length - penalty_spot - spot, 'y0': width / 2 - spot, 'x1': length - penalty_spot + spot, 'y1': width / 2 + spot, 'line_color': linecolor, 'line_width': 1, 'fillcolor': linecolor})
        # Left goal
        field.append({'type': 'rect', 'xref': 'x','yref': 'y','x0': -2, 'y0': width/2 - goal / 2,'x1': 0,'y1': width/2 + goal / 2,'line_color': linecolor,'line_width': 1})
        # Right goal
        field.append({'type': 'rect', 'xref': 'x', 'yref': 'y', 'x0': length + 2, 'y0': width / 2 - goal / 2, 'x1': length,'y1': width / 2 + goal / 2, 'line_color': linecolor, 'line_width': 1})
        # Prepare Arcs
        field.append({'type': 'path', 'path': ellipse_arc(penalty_spot, width / 2, circle_radius, circle_radius, math.radians(52.5)*(-1), math.radians(52.5)), 'line_color': linecolor, 'line_width': 1})
        field.append({'type': 'path', 'path': ellipse_arc(length - penalty_spot, width / 2, circle_radius, circle_radius, math.radians(127.5), math.radians(232.5)), 'line_color': linecolor, 'line_width': 1})

        field.append({'type': 'path', 'path': ellipse_arc(0, 0, 1, 1, math.radians(0),math.radians(90)), 'line_color': linecolor, 'line_width': 1})
        field.append({'type': 'path', 'path': ellipse_arc(0, width, 1, 1, math.radians(0), (-1)*math.radians(90)), 'line_color': linecolor, 'line_width': 1})
        field.append({'type': 'path', 'path': ellipse_arc(length, 0, 1, 1, math.radians(90),math.radians(180)), 'line_color': linecolor, 'line_width': 1})
        field.append({'type': 'path', 'path': ellipse_arc(length, width, 1, 1, math.radians(180), math.radians(270)), 'line_color': linecolor, 'line_width': 1})

    return field


@st.cache(allow_output_mutation=True)
def draw_half_pitch(height=50, width=65, linecolor='white'):
    """
    Creates a plot of the pitch, which size is given with length and width.

    Parameters
    ----------
    height : int
        The 'height' is the height of the pitch (goal to center line).
        Default value is 50
    width : int
        The 'width' is the width of the pitch (sideline to sideline).
        Default value is 65.
    linecolor : string
        Colors of field lines, you have to give it in hex code.
        Default values is white (#ffffff).

    Returns
    -------
    field : list
        Array where is all field notations.
    """
    normal_field = 105
    center_spot = normal_field / 2

    penalty_area = 16.5
    gk_box = 5.5
    goal = 7.32
    circle_radius = 9.15
    penalty_spot = 11
    spot = 0.5
    half_field = []

    if height <= penalty_area + 20:
        st.error(f"Height of the field has to be at least {penalty_area + 20}?")
        return
    elif (height >= 52.5) or (width > 65):
        st.error("Field dimensions are too big. Maximum height is 52.5, maximum width is 65")
        return
    else:
        # Pitch Outline
        half_field.append({'type': 'rect', 'xref': 'x','yref': 'y','x0': 0,'y0': 0,'x1': width,'y1': height,'line_color': linecolor,'line_width': 1})
        # Penalty Area
        half_field.append({'type': 'rect', 'xref': 'x', 'yref': 'y', 'x0': width / 2 - goal / 2 - penalty_area, 'y0': 0, 'x1': width / 2 + goal / 2 + penalty_area, 'y1': penalty_area, 'line_color': linecolor, 'line_width': 1})
        # GK Area
        half_field.append({'type': 'rect', 'xref': 'x', 'yref': 'y', 'x0': width / 2 - goal / 2 - gk_box, 'y0': 0, 'x1': width / 2 + goal / 2 + gk_box, 'y1': gk_box, 'line_color': linecolor, 'line_width': 1})
        # Penalty spot
        half_field.append({'type': 'circle', 'xref': 'x', 'yref': 'y', 'x0': width / 2 - spot , 'y0': penalty_spot - spot, 'x1': width / 2 + spot, 'y1': penalty_spot + spot, 'line_color': linecolor, 'line_width': 1, 'fillcolor': linecolor})
        # Right goal
        half_field.append({'type': 'rect', 'xref': 'x', 'yref': 'y', 'x0': width / 2 - goal / 2, 'y0': -2, 'x1': width / 2 + goal / 2,'y1': 0, 'line_color': linecolor, 'line_width': 1})
        # Prepare Arcs
        half_field.append({'type': 'path', 'path': ellipse_arc(width / 2, penalty_spot, circle_radius, circle_radius, math.radians(37),math.radians(143)), 'line_color': linecolor, 'line_width': 1})

        half_field.append({'type': 'path', 'path': ellipse_arc(0, 0, 1, 1, math.radians(0),math.radians(90)), 'line_color': linecolor, 'line_width': 1})
        half_field.append({'type': 'path', 'path': ellipse_arc(width, 0, 1, 1, math.radians(90),math.radians(180)), 'line_color': linecolor, 'line_width': 1})

        # make center circle
        if height < center_spot:
            over_height = (center_spot - height) + 0.2
            tan = np.degrees(np.arctan(over_height / circle_radius))
            starting_degree = tan * (-1)
            ending_degree = (180 - abs(starting_degree)) * (-1)
            #print(f"{over_height} / {circle_radius} > ATAN: {tan} : {starting_degree} - {ending_degree}")
            half_field.append({'type': 'path', 'path': ellipse_arc(width / 2, height + over_height, circle_radius, circle_radius, math.radians(starting_degree), math.radians(ending_degree)), 'line_color': linecolor, 'line_width': 1})

    return half_field


@st.cache(allow_output_mutation=True)
def draw_interception(df, team='', orig='', field='', caption='', subcaption='', length=105, width=68, type='', ball_missed=5, scoring_chance=20, ball_possession=15, info_text=0,  linecolor='white', fillcolor='#80B860', intercept_index=[]):
    """
    Plot interception positions in full field

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
    field_length : int
        How long field is
    field_width : int
        What is field height
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
    df_copy = df.copy()
    df_copy.index = np.arange(0, len(df_copy))

    if len(intercept_index) > 0:
        interceptions = df.iloc[intercept_index]
        if team != 'Opponent':
            interceptions = interceptions.loc[((interceptions.team != '') & (interceptions.team != interceptions.prev_team) & (interceptions.team == team))]
        else:
            interceptions = interceptions.loc[((interceptions.team != '') & (interceptions.team != interceptions.prev_team) & (interceptions.team != orig))]
    else:
        # Take only those events where previous event was made by different team than selected team
        if team != 'Opponent':
            interceptions = df.loc[((df.team != '') & (df.team != df.prev_team) & (df.team == team))]
        else:
            interceptions = df.loc[((df.team != '') & (df.team != df.prev_team) & (df.team != orig))]

    x_mean = interceptions.x.mean()
    count_of_interceptions = len(interceptions)
    # Set background and other information for background of field
    layout = go.Layout(
        font_color=linecolor,
        showlegend=False,
        xaxis=dict(showgrid=False, range=[-3, length+3], zeroline=False, visible=False),
        yaxis=dict(range=[-3, width+10], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1, automargin=True),
        autosize=True, plot_bgcolor=fillcolor, paper_bgcolor=fillcolor, margin={'l': 10, 'r': 10, 't': 20, 'b': 20}, shapes=field
    )

    datas = []
    annotations = []
    color = ['red', 'blue']

    # Add starting point with hover effect
    datas.append(go.Scatter(x=interceptions['x'], y=interceptions['y'], opacity=0.7, name="", hovertemplate='%{text}',
                            text=[
                                '<b>{}</b><br /><i>Player:</i> {}<br /><i>Position:</i> {}<br /><i>Event:</i> {}<br /><i>Team:</i> {}<br /><i>Coordinate:</i> ({},{})'.format(
                                    type, r.PlayerName, r.Position, r.event, r.team, round(r['x'], 2), round(r['y'], 2))
                                for i, r in interceptions.iterrows()], showlegend=False, mode='markers', marker_size=10,
                            marker_color=[
                                what_color_action(how_many_seconds_to_next_opp_action(df, i), ball_missed,
                                                     scoring_chance, ball_possession) for i, r in
                                interceptions.iterrows()], marker_line_width=1, marker_line_color=linecolor))

    # Average line where attacks are started
    datas.append(go.Scatter(x=[x_mean, x_mean], y=[0, width],
                               mode='lines', line={'dash': 'dash', 'width': 2, 'color': 'red'}, showlegend=False, opacity=0.8, text=f"Average distance from own goal where attacks are started is {round(x_mean, 2)} meters.",
                               hovertemplate=f"Average distance from own goal where attacks are started is {round(x_mean, 2)} meters.", xaxis='x', yaxis='y'))

    # Add captions
    if caption != '':
        annotations.append({'font_color': linecolor, 'font_size': 16, 'x': 0, 'y': width + 6, 'showarrow': False,
                            'text': "{} (count of starts {})".format(caption, count_of_interceptions), 'textangle': 0,
                            'xanchor': 'left', 'yanchor': 'bottom'})
    if subcaption != '':
        annotations.append({'font_color': linecolor, 'font_size': 12, 'x': 0, 'y': width + 6, 'showarrow': False,
                            'text': "{}".format(subcaption), 'textangle': 0, 'xanchor': 'left', 'yanchor': 'top'})

    # Add info box
    if info_text == 1:
        annotations.append({'font_color': BALL_MISSED_COLOR, 'font_size': 8, 'x': -2, 'y': 0, 'showarrow': False,
                            'text': f"Ball possession lost in {ball_missed} seconds", 'textangle': 270, 'xanchor': 'center', 'yanchor': 'bottom'})
        annotations.append({'font_color': OTHER_COLOR, 'font_size': 8, 'x': -2, 'y': width, 'showarrow': False,
                            'text': f"Ball possession continued for {ball_missed} to {ball_possession} seconds", 'textangle': 270, 'xanchor': 'center', 'yanchor': 'top'})
        annotations.append({'font_color': BALL_POSSESSION_COLOR, 'font_size': 8, 'x': length + 2, 'y': width, 'showarrow': False, 'yanchor': 'top',
                            'text': f"Ball possession continued for at least {ball_possession} seconds", 'textangle': 90, 'xanchor': 'center'})
        annotations.append({'font_color': SCORING_CHANCE_COLOR, 'font_size': 8, 'x': length + 2, 'y': 0, 'showarrow': False, 'yanchor': 'bottom',
                            'text': f"Scoring chance created in {scoring_chance} seconds", 'textangle': 90, 'xanchor': 'center'})

        # Add attacking direction arrow
        annotations.append(
                {'opacity': 0.5, 'x': length/2 + 10, 'y': -1, 'ax': length/2 - 10, 'ay': -1, 'xref': "x", 'yref': "y", 'axref': "x",
                 'ayref': "y", 'text': "attacking direction", 'font_color': linecolor, 'xanchor': 'left', 'yanchor': 'top', 'font_size': 9, 'showarrow': True, 'arrowhead': 1, 'arrowwidth': 1, 'arrowcolor': linecolor})

    fig = go.Figure(data=datas, layout=layout)
    fig.update_layout(annotations=annotations)
    fig.update_shapes(layer='below')

    return fig, annotations, count_of_interceptions


def add_destination(df):
    # Add arrow, if user wants this
    color = ['gray', 'gray']
    annotations = []

    for ai, ar in df.iterrows():
        annotations.append(
            {'opacity': 0.5, 'x': ar.x2, 'y': ar.y2, 'ax': ar.x, 'ay': ar.y, 'xref': "x", 'yref': "y", 'axref': "x",
             'ayref': "y", 'text': "", 'showarrow': True, 'arrowhead': 3, 'arrowwidth': 1.5, 'arrowcolor': color[ar['outcome']]})

    return annotations


@st.cache(allow_output_mutation=True)
def draw_events(df, field, caption='', subcaption='', length=105, width=68, type='', half=0, linecolor='white', fillcolor='#80B860', all_df=pd.DataFrame(), ball_missed=5, scoring_chance=20, ball_possession=15, info_text=1):
    # Set background and other information for background of field
    layout = go.Layout(
        font_color=linecolor,
        showlegend=False,
        xaxis=dict(showgrid=False, range=[-3, length+3], zeroline=False, visible=False),
        yaxis=dict(range=[-3, width+10], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1, automargin=True),
        autosize=True, plot_bgcolor=fillcolor, paper_bgcolor=fillcolor, margin={'l': 10, 'r': 10, 't': 20, 'b': 20}, shapes=field
    )

    datas = []
    annotations = []
    color = ['red', 'blue']

    if half == 0:
        x = 'x'
        y = 'y'
    else:
        x = 'xh'
        y = 'yh'

    # Add starting point with hover effect
    datas.append(go.Scatter(x=df[x], y=df[y], opacity=0.7, name="", hovertemplate='%{text}', text=[
        '<b>{}</b><br /><i>Player:</i> {}<br /><i>Position:</i> {}<br /><i>Event:</i> {}<br /><i>Outcome:</i> {}<br /><i>Coordinate:</i> ({},{})'.format(
            type, r.PlayerName, r.Position, r.event, r.outcome, round(r[x], 2), round(r[y], 2)) for i, r in
        df.iterrows()], showlegend=False, mode='markers', marker_size=15,
                            marker_color=[what_color_action(how_many_seconds_to_next_opp_action(all_df, i), ball_missed,scoring_chance, ball_possession) for i, r in df.iterrows()], marker_line_width=1,
                            marker_line_color=linecolor))

    # [what_color_action(how_many_seconds_to_next_opp_action(df, i), ball_missed,scoring_chance, ball_possession) for i, r in interceptions.iterrows()]
    # [color[r['outcome']] for i, r in df.iterrows()]



    # Add attacking direction arrow
    annotations.append(
            {'opacity': 0.5, 'x': length/2 + 10, 'y': -2, 'ax': length/2 - 10, 'ay': -2, 'xref': "x", 'yref': "y", 'axref': "x",
             'ayref': "y", 'text': "attacking direction", 'font_color': linecolor, 'xanchor': 'left', 'yanchor': 'top', 'font_size': 9, 'showarrow': True, 'arrowhead': 1, 'arrowwidth': 1, 'arrowcolor': linecolor})

    # Add info box
    if info_text == 1:
        annotations.append({'font_color': BALL_MISSED_COLOR, 'font_size': 8, 'x': -2, 'y': 0, 'showarrow': False,
                            'text': f"Ball possession lost in {ball_missed} seconds", 'textangle': 270, 'xanchor': 'center', 'yanchor': 'bottom'})
        annotations.append({'font_color': OTHER_COLOR, 'font_size': 8, 'x': -2, 'y': width, 'showarrow': False,
                            'text': f"Ball possession continued for {ball_missed} to {ball_possession} seconds", 'textangle': 270, 'xanchor': 'center', 'yanchor': 'top'})
        annotations.append({'font_color': BALL_POSSESSION_COLOR, 'font_size': 8, 'x': length + 2, 'y': width, 'showarrow': False, 'yanchor': 'top',
                            'text': f"Ball possession continued for at least {ball_possession} seconds", 'textangle': 90, 'xanchor': 'center'})
        annotations.append({'font_color': SCORING_CHANCE_COLOR, 'font_size': 8, 'x': length + 2, 'y': 0, 'showarrow': False, 'yanchor': 'bottom',
                            'text': f"Scoring chance created in {scoring_chance} seconds", 'textangle': 90, 'xanchor': 'center'})

    # Add text
    if caption != '':
        annotations.append({'font_color': linecolor, 'font_size': 16, 'x': 0, 'y': width + 6, 'showarrow': False,
                            'text': "{}".format(caption), 'textangle': 0,
                            'xanchor': 'left', 'yanchor': 'bottom'})
    if subcaption != '':
        annotations.append({'font_color': linecolor, 'font_size': 12, 'x': 0, 'y': width + 6, 'showarrow': False,
                            'text': "{}".format(subcaption), 'textangle': 0, 'xanchor': 'left', 'yanchor': 'top'})

    fig = go.Figure(data=datas, layout=layout)
    fig.update_layout(annotations=annotations)
    fig.update_shapes(layer='below')

    return fig, annotations


def change_color(data, fillcolor, linecolor):
    keys = []
    if isinstance(data, dict):
        for field_size in data.keys():
            for row in data[field_size]:
                if 'line_color' in row.keys():
                    row['line_color'] = linecolor

                if 'fillcolor' in row.keys():
                    row['fillcolor'] = linecolor
    elif isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                if 'font_color' in row.keys():
                    row['font_color'] = linecolor

                if 'text' in row.keys():
                    if row['text'] == 'attacking direction':
                        row['arrowcolor'] = linecolor

    return data


def _change_range(value, old_range, new_range):
    '''
    Convert a value from one range to another one, maintaining ratio.
    '''
    return ((value-old_range[0]) / (old_range[1]-old_range[0])) * (new_range[1]-new_range[0]) + new_range[0]


def draw_passing_network(df, min_pass, field, selected_position='', game_phase='All', threshold=5, length=105, width=68, only_starting_11=1, caption="", subcaption='', linecolor='white', fillcolor='#80B860', starting_eleven=[]):
    df['receiver'] = df['id'].shift(-1)
    df['receiver_team'] = df['team'].shift(-1)

    if only_starting_11 == 1:
        unique_players = df.loc[(df.Start11 == 1)]
        unique_players = unique_players['id'].unique()
    else:
        unique_players = df['id'].unique()

    player_location = {}
    max_pair_count = 0
    max_pass_count = 0

    players_are_selected = False
    if len(starting_eleven) > 0:
        players_are_selected = True

    for action in unique_players:
        next_player = False
        if action == '':
            continue

        if players_are_selected:
            for row in starting_eleven:
                if row['id'] == action:
                    next_player = True
                    break

            if not next_player:
                continue

        orig = df.loc[(df['id'] == action)]
        player_id = str(orig.id.unique()[0])
        name = str(orig.PlayerName.unique()[0])
        number = int(orig.Number.unique()[0])
        position = str(orig.Position.unique()[0])
        team = str(orig.team.unique()[0])

        if game_phase == 'Opening + build up':
            orig = orig.loc[(orig['x'] < length / 2)]
        elif game_phase == 'Penetration + finishing':
            orig = orig.loc[(orig['x'] > length / 2)]

        x = orig['x'].mean()
        y = orig['y'].mean()

        if x <= 0 and y <= 0:
            continue

        pass_row = orig.loc[((orig.event == 'Pass') | (orig.event == 'Cross'))]
        num_of_passes = len(pass_row)

        if max_pair_count < num_of_passes:
            max_pair_count = num_of_passes

        passes_to = {}
        for i, one_pass in pass_row.iterrows():
            if one_pass.receiver_team != one_pass.team:
                continue

            if one_pass.receiver in passes_to.keys():
                passes_to[one_pass.receiver] += 1
            else:
                passes_to[one_pass.receiver] = 1

        if len(passes_to) > 0:
            if max_pass_count < max(passes_to.values()):
                max_pass_count = max(passes_to.values())

        player_location[action] = {'id': player_id, 'name': name, 'number': number, 'team': team, 'position': position, 'x': x, 'y': y, 'count': num_of_passes, 'passes_to': passes_to}

    # Set background and other information for background of field
    layout = go.Layout(
        hovermode='closest',
        font_color=linecolor,
        showlegend=False,
        xaxis=dict(showgrid=False, range=[-3, length+3], zeroline=False, visible=False),
        yaxis=dict(range=[-3, width+10], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1, automargin=True),
        autosize=True, plot_bgcolor=fillcolor, paper_bgcolor=fillcolor, margin={'l': 10, 'r': 10, 't': 20, 'b': 20}, shapes=field
    )

    formation = []
    passes = []
    annotations = []
    index = 0

    norm = Normalize(vmin=0, vmax=len(player_location))
    edge_cmap = cm.get_cmap('Blues')

    for player in player_location.keys():
        player_passes = player_location[player]

        if len(starting_eleven) > 0:
            marker_size = 50
        else:
            marker_size = _change_range(player_passes['count'], (0, max_pair_count), (25, 75)) / threshold

        edge_color = edge_cmap(norm(index))
        colors = list(edge_color)
        marker_fill = f"rgba({colors[0]*255}, {colors[1]*255}, {colors[2]*255}, 0.7)"
        line = f"rgba({colors[0]*255}, {colors[1]*255}, {colors[2]*255}, 1.0)"

        # Draw player position
        formation.append(
            go.Scatter(x=[player_passes['x']], y=[player_passes['y']], hovertemplate=f"<b>#{player_passes['number']} {player_passes['name']}</b><br />{player_passes['position']}", name='',
                           opacity=1, text=f"<b>{player_passes['number']}</b>", showlegend=False, mode='markers+text',
                       textposition='middle center', marker_size=marker_size, marker_color=fillcolor,
                       textfont=dict(color=line, size=16), marker_line_width=3, marker_line_color=line))

        # Draw passing lines
        drawed_passer = []
        for receiver in player_passes['passes_to']:
            if receiver not in player_location.keys():
                continue

            if selected_position['id'] != '':
                if selected_position['id'] != player:
                    continue

            if player_passes['passes_to'][receiver] >= min_pass:
                line_size = _change_range(player_passes['passes_to'][receiver], (0, max_pass_count), (2.5, 10))

                if f"{player_location[receiver]['id']}-{player_passes['id']}" in drawed_passer:
                    if player_passes['x'] <= player_location[receiver]['x']:
                        pass_location = -0.5
                    else:
                        pass_location = 0.5
                else:
                    if player_passes['x'] <= player_location[receiver]['x']:
                        pass_location = 0.5
                    else:
                        pass_location = -0.5

                passes.append(
                    go.Scatter(x=[player_passes['x'], player_location[receiver]['x'] + pass_location], y=[player_passes['y'], player_location[receiver]['y'] + pass_location],
                               mode='lines', line={'width': line_size, 'color': line}, showlegend=False, opacity=0.8, text='{} number of Passes: {}'.format(player_passes['name'], player_passes['passes_to'][receiver]),
                               hovertemplate=f"{player_passes['name']} number of Passes {player_passes['passes_to'][receiver]}", xaxis='x', yaxis='y'))

            drawed_passer.append(f"{player_passes['id']}-{player_location[receiver]['id']}")

        index += 1

    # Add attacking direction arrow
    annotations.append(
            {'opacity': 0.5, 'x': length/2 + 10, 'y': -2, 'ax': length/2 - 10, 'ay': -2, 'xref': "x", 'yref': "y", 'axref': "x",
             'ayref': "y", 'text': "attacking direction", 'font_color': linecolor, 'xanchor': 'left', 'yanchor': 'top', 'font_size': 9, 'showarrow': True, 'arrowhead': 1, 'arrowwidth': 1, 'arrowcolor': linecolor})

    # Add text
    if caption != '':
        annotations.append({'font_color': linecolor, 'font_size': 16, 'x': 0, 'y': width + 6, 'showarrow': False,
                            'text': "{}".format(caption), 'textangle': 0,
                            'xanchor': 'left', 'yanchor': 'bottom'})
    if subcaption != '':
        annotations.append({'font_color': linecolor, 'font_size': 12, 'x': 0, 'y': width + 6, 'showarrow': False,
                            'text': "{}".format(subcaption), 'textangle': 0, 'xanchor': 'left', 'yanchor': 'top'})

    datas = passes + formation
    fig = go.Figure(data=datas, layout=layout)
    fig.update_layout(annotations=annotations)
    fig.update_shapes(layer='below')
    return fig, annotations


def make_areas(x, y, cols, rows, length=105, width=68):
    """

    """
    area = 0
    bin_length = length / cols
    bin_width = width / rows
    if np.isnan(x) or np.isnan(y):
        area = np.nan
    else:
        x_col = x // bin_length + 1 if x % bin_length > 0 or x == 0 else x // bin_length
        y_row = y // bin_width + 1 if y % bin_width > 0 or y == 0 else y // bin_width

        area = (x_col * 4 - y_row) + 1

    return area, x_col, y_row


def draw_target_heatmap(df, field, field_size='half', length=65, width=50, cols=20, rows=10, caption='', subcaption='', linecolor='white', fillcolor='#80B860'):
    annotations = []
    datas = []

    end_x = 'xh2' if field_size == 'half' else 'x2'
    end_y = 'yh2' if field_size == 'half' else 'y2'

    # Set background and other information for background of field
    layout = go.Layout(
        hovermode='closest',
        font_color=linecolor,
        showlegend=False,
        xaxis=dict(showgrid=False, range=[-3, length+3], zeroline=False, visible=False),
        yaxis=dict(range=[-3, width+10], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1, automargin=True),
        autosize=True, plot_bgcolor=fillcolor, paper_bgcolor=fillcolor, margin={'l': 10, 'r': 10, 't': 20, 'b': 20}, shapes=field
    )

    # Draw heatmap
    norm = Normalize(vmin=0, vmax=len(df))
    edge_cmap = cm.get_cmap('Blues')
    colors = []

    for color in np.arange(0, len(df)):
        if color == 0:
            colors.append('rgba(255,255,255,0.1)')
        else:
            colors.append(f"rgba{edge_cmap(norm(color))}")

    datas.append(go.Histogram2d(y=df[end_y],
                                x=df[end_x],
                                histnorm='probability',
                                autobinx=False,
                                xbins=dict(start=0, end=length, size=length/cols),
                                autobiny=False,
                                ybins=dict(start=0, end=width, size=width/rows),
                                xgap=1,
                                ygap=1,
                                colorscale=colors,
                                #zmin=0,
                                opacity=0.9))

    # Add attacking direction arrow
    if field_size == 'full':
        annotations.append(
            {'opacity': 0.5, 'x': length/2 + 10, 'y': -2, 'ax': length/2 - 10, 'ay': -2, 'xref': "x", 'yref': "y", 'axref': "x",
            'ayref': "y", 'text': "attacking direction", 'font_color': linecolor, 'xanchor': 'left', 'yanchor': 'top', 'font_size': 9, 'showarrow': True, 'arrowhead': 1, 'arrowwidth': 1, 'arrowcolor': linecolor})

    # Add text
    if caption != '':
        annotations.append({'font_color': linecolor, 'font_size': 16, 'x': 0, 'y': width + 6, 'showarrow': False,
                            'text': "{}".format(caption), 'textangle': 0,
                            'xanchor': 'left', 'yanchor': 'bottom'})
    if subcaption != '':
        annotations.append({'font_color': linecolor, 'font_size': 12, 'x': 0, 'y': width + 6, 'showarrow': False,
                            'text': "{}".format(subcaption), 'textangle': 0, 'xanchor': 'left', 'yanchor': 'top'})

    fig = go.Figure(data=datas, layout=layout)
    fig.update_layout(annotations=annotations)
    fig.update_shapes(layer='below')
    return fig, annotations


def draw_passing_heatmap(df, field, player={}, length=105, width=68, cols=30, rows=10, caption='', subcaption='', linecolor='white', fillcolor='#80B860', action='pass'):
    annotations = []
    datas = []
    if action == 'pass':
        player_passes = df.loc[((df['id'] == player['id']) & (df.event.str.contains('Pass') | df.event.str.contains('Cross')))]
    elif action == 'action':
        player_passes = df.loc[(df['id'] == player['id'])]
    else:
        player_passes = df

    # Set background and other information for background of field
    layout = go.Layout(
        hovermode='closest',
        font_color=linecolor,
        showlegend=False,
        xaxis=dict(showgrid=False, range=[-3, length+3], zeroline=False, visible=False),
        yaxis=dict(range=[-3, width+10], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1, automargin=True),
        autosize=True, plot_bgcolor=fillcolor, paper_bgcolor=fillcolor, margin={'l': 10, 'r': 10, 't': 20, 'b': 20}, shapes=field
    )

    # Draw heatmap
    norm = Normalize(vmin=0, vmax=len(player_passes))
    edge_cmap = cm.get_cmap('Blues')
    colors = []
    for color in np.arange(0, len(player_passes)):
        if color == 0:
            colors.append('rgba(255,255,255,0.1)')
        else:
            colors.append(f"rgba{edge_cmap(norm(color))}")

    datas.append(go.Histogram2d(x=player_passes.x,
                                y=player_passes.y,
                                #histnorm='probability',
                                autobinx=False,
                                xbins=dict(start=0, end=length, size=length/cols),
                                autobiny=False,
                                ybins=dict(start=0, end=width, size=width/rows),
                                xgap=1,
                                ygap=1,
                                colorscale=colors,
                                zmin=0,
                                opacity=0.9))

    # Add attacking direction arrow
    annotations.append(
        {'opacity': 0.5, 'x': length/2 + 10, 'y': -2, 'ax': length/2 - 10, 'ay': -2, 'xref': "x", 'yref': "y", 'axref': "x",
         'ayref': "y", 'text': "attacking direction", 'font_color': linecolor, 'xanchor': 'left', 'yanchor': 'top', 'font_size': 9, 'showarrow': True, 'arrowhead': 1, 'arrowwidth': 1, 'arrowcolor': linecolor})

    # Add text
    if caption != '':
        annotations.append({'font_color': linecolor, 'font_size': 16, 'x': 0, 'y': width + 6, 'showarrow': False,
                            'text': "{}".format(caption), 'textangle': 0,
                            'xanchor': 'left', 'yanchor': 'bottom'})
    if subcaption != '':
        annotations.append({'font_color': linecolor, 'font_size': 12, 'x': 0, 'y': width + 6, 'showarrow': False,
                            'text': "{}".format(subcaption), 'textangle': 0, 'xanchor': 'left', 'yanchor': 'top'})

    fig = go.Figure(data=datas, layout=layout)
    fig.update_layout(annotations=annotations)
    fig.update_shapes(layer='below')
    return fig, annotations


def draw_bar_chart(df, home_team='', away_team='', caption='', subcaption=''):
    datas = []
    annotations = []

    y_axes = list(df['Set Piece'].unique())
    max_value = df['Count'].max()
    min_value = max_value * (-1)

    for team in [home_team, away_team]:
        x_bar = []
        y_bar = []
        for sp in y_axes:
            pcs = df.loc[((df['Set Piece'] == sp) & (df['Team'] == team))]
            count_of_sp = pcs['Count'].sum()
            x_bar.append(count_of_sp * (-1)) if team == home_team else x_bar.append(count_of_sp)
            y_bar.append(sp)

        # Make home team bar
        color = 'rgb(55, 83, 109)' if team == home_team else 'rgb(26, 118, 255)'
        datas.append(go.Bar(x=x_bar, y=y_bar, name=f"{team}", marker_color=color,  orientation='h', text=[f"<b style='font-size: 1.2rem;'>{abs(i)}</b>" for i in x_bar], textposition="inside", textangle=0, textfont_color="white",))


    layout = go.Layout(
        font_color='black',
        showlegend=True,
        height=400,
        xaxis=dict(gridwidth=0.2, gridcolor='rgba(0, 0, 0, 0.1)', showline=False, showticklabels=False, showgrid=False, zeroline=False, visible=False),
        yaxis=dict(gridwidth=0.5, gridcolor='rgba(0, 0, 0, 0.1)', showline=True, showticklabels=True, showgrid=True, zeroline=True, visible=True, automargin=True),
        autosize=True, plot_bgcolor='white', paper_bgcolor='white', margin={'l': 10, 'r': 10, 't': 20, 'b': 20}
    )

    if caption != '':
        annotations.append({'font_color': 'black', 'font_size': 16, 'x': '>40', 'y': max_passes + 10, 'showarrow': False, 'yanchor': 'bottom',
                            'text': "<b>{}</b>".format(caption), 'textangle': 0, 'xanchor': 'right'})
    if subcaption != '':
        annotations.append({'font_color': 'black', 'font_size': 12, 'x': '>40', 'y': max_passes + 8, 'showarrow': False, 'yanchor': 'top',
                            'text': "{}".format(subcaption), 'textangle': 0, 'xanchor': 'right'})

    fig = go.Figure(data=datas, layout=layout)
    fig.update_layout(annotations=annotations, barmode='relative')
    fig.update_shapes(layer='below')

    return fig


@st.cache(allow_output_mutation=True)
def passes_to_box(df, team, orig, event, field, field_size='half', caption='', subcaption='', length=68, width=50, type='', ball_missed=5, scoring_chance=20, ball_possession=15, info_text=0, linecolor='white', fillcolor='#80B860', game_type_index=[]):
    # Set background and other information for background of field
    layout = go.Layout(
        font_color=linecolor,
        showlegend=False,
        xaxis=dict(showgrid=False, range=[-3, length+3], zeroline=False, visible=False),
        yaxis=dict(range=[-3, width+10], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1, automargin=True),
        autosize=True, plot_bgcolor=fillcolor, paper_bgcolor=fillcolor, margin={'l': 10, 'r': 10, 't': 20, 'b': 20}, shapes=field
    )

    datas = []
    annotations = []
    size = 15

    start_x = 'xh' if field_size == 'half' else 'x'
    start_y = 'yh' if field_size == 'half' else 'y'
    end_x = 'xh2' if field_size == 'half' else 'x2'
    end_y = 'yh2' if field_size == 'half' else 'y2'

    to_box_line = 7.32/2 + 16.5
    box_start_x = length / 2 - to_box_line if field_size == 'half' else length - 16.5
    box_end_x = length / 2 + to_box_line if field_size == 'half' else length
    box_start_y = 0 if field_size == 'half' else width / 2 - to_box_line
    box_end_y = 16.5 if field_size == 'half' else width / 2 + to_box_line

    if len(game_type_index) > 0:
        parsed_events = df.iloc[game_type_index]
        events = parsed_events.loc[((parsed_events.team == team) & (parsed_events.event.str.contains(event)))] if team != 'Opponent' else parsed_events.loc[((parsed_events.team != orig) & (parsed_events.event.str.contains(event)))]
    else:
        events = df.loc[((df.team == team) & (df.event.str.contains(event)))] if team != 'Opponent' else df.loc[((df.team != orig) & (df.event.str.contains(event)))]

    #events = df.loc[((df.team == team) & (df.event.str.contains(event)))] if team != 'Opponent' else df.loc[((df.team != orig) & (df.event.str.contains(event)))]
    events = events.loc[(events[end_x].between(box_start_x, box_end_x) & events[end_y].between(box_start_y, box_end_y))]
    count_of_events = len(events)

    # Add starting point with hover effect
    datas.append(go.Scatter(x=events[start_x], y=events[start_y], opacity=0.7, name="", hovertemplate='%{text}', text=[
        '<b>{}</b><br><i>Player:</i> {}<br><i>Event:</i> {}'.format(type, r.PlayerName, r.event) for i, r in
        events.iterrows()], showlegend=False, mode='markers', marker_size=size,
                            marker_color=[
                                what_color_action(how_many_seconds_to_next_opp_action(df, i), ball_missed,
                                                     scoring_chance, ball_possession) for i, r in
                                events.iterrows()], marker_line_width=1,
                            marker_line_color=linecolor))

    # Add attacking direction arrow
    if field_size == 'full':
        annotations.append(
                {'opacity': 0.5, 'x': length/2 + 10, 'y': -2, 'ax': length/2 - 10, 'ay': -2, 'xref': "x", 'yref': "y", 'axref': "x",
                 'ayref': "y", 'text': "attacking direction", 'font_color': linecolor, 'xanchor': 'left', 'yanchor': 'top', 'font_size': 9, 'showarrow': True, 'arrowhead': 1, 'arrowwidth': 1, 'arrowcolor': linecolor})

    for ai, ar in events.iterrows():
        annotations.append(
            {'opacity': 0.5, 'x': ar[end_x], 'y': ar[end_y], 'ax': ar[start_x], 'ay': ar[start_y], 'xref': "x", 'yref': "y", 'axref': "x",
             'ayref': "y", 'text': "", 'showarrow': True, 'arrowhead': 3, 'arrowwidth': 1.5, 'arrowcolor': what_color_action(how_many_seconds_to_next_opp_action(df, ai), ball_missed, scoring_chance, ball_possession)})

    # Add text
    if caption != '':
        annotations.append({'font_color': linecolor, 'font_size': 16, 'x': 0, 'y': width + 6, 'showarrow': False,
                            'text': "{} (count of events {})".format(caption, count_of_events), 'textangle': 0,
                            'xanchor': 'left', 'yanchor': 'bottom'})
    if subcaption != '':
        annotations.append({'font_color': linecolor, 'font_size': 12, 'x': 0, 'y': width + 6, 'showarrow': False,
                            'text': "{}".format(subcaption), 'textangle': 0, 'xanchor': 'left', 'yanchor': 'top'})

    fig = go.Figure(data=datas, layout=layout)
    fig.update_layout(annotations=annotations)
    fig.update_shapes(layer='below')

    return fig, annotations


@st.cache(allow_output_mutation=True)
def passes_lead_to_shot(df, team, orig, event, field, field_size='half', caption='', subcaption='', length=68, width=50, type='', ball_missed=5, scoring_chance=20, ball_possession=15, info_text=0, linecolor='white', fillcolor='#80B860', game_type_index=[]):
    # Set background and other information for background of field
    layout = go.Layout(
        font_color=linecolor,
        showlegend=False,
        xaxis=dict(showgrid=False, range=[-3, length+3], zeroline=False, visible=False),
        yaxis=dict(range=[-3, width+10], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1, automargin=True),
        autosize=True, plot_bgcolor=fillcolor, paper_bgcolor=fillcolor, margin={'l': 10, 'r': 10, 't': 20, 'b': 20}, shapes=field
    )

    datas = []
    annotations = []
    size = 15

    start_x = 'xh' if field_size == 'half' else 'x'
    start_y = 'yh' if field_size == 'half' else 'y'
    end_x = 'xh2' if field_size == 'half' else 'x2'
    end_y = 'yh2' if field_size == 'half' else 'y2'

    if len(game_type_index) > 0:
        parsed_events = df.iloc[game_type_index]
        events = parsed_events.loc[((parsed_events.team == team) & ((parsed_events.event.str.contains('Pass') | parsed_events.event.str.contains('Cross')) & parsed_events.event.str.contains('Shot').shift(-1)))] if team != 'Opponent' else parsed_events.loc[((parsed_events.team != orig) & ((parsed_events.event.str.contains('Pass') | parsed_events.event.str.contains('Cross')) & parsed_events.event.str.contains('Shot').shift(-1)))]
    else:
        events = df.loc[((df.team == team) & ((df.event.str.contains('Pass') | df.event.str.contains('Cross')) & df.event.str.contains('Shot').shift(-1)))] if team != 'Opponent' else df.loc[((df.team != orig) & ((df.event.str.contains('Pass') | df.event.str.contains('Cross')) & df.event.str.contains('Shot').shift(-1)))]


    # Add starting point with hover effect
    datas.append(go.Scatter(x=events[start_x], y=events[start_y], opacity=0.7, name="", hovertemplate='%{text}', text=[
        '<b>{}</b><br><i>Player:</i> {}<br><i>Event:</i> {}'.format(type, r.PlayerName, r.event) for i, r in
        events.iterrows()], showlegend=False, mode='markers', marker_size=size,
                            marker_color='blue', marker_line_width=1,
                            marker_line_color=linecolor))

    for ai, ar in events.iterrows():
        annotations.append(
            {'opacity': 0.5, 'x': ar[end_x], 'y': ar[end_y], 'ax': ar[start_x], 'ay': ar[start_y], 'xref': "x", 'yref': "y", 'axref': "x",
             'ayref': "y", 'text': "", 'showarrow': True, 'arrowhead': 3, 'arrowwidth': 1.5, 'arrowcolor': 'blue'})

    # Add attacking direction arrow
    annotations.append(
            {'opacity': 0.5, 'x': length/2 + 10, 'y': -2, 'ax': length/2 - 10, 'ay': -2, 'xref': "x", 'yref': "y", 'axref': "x",
             'ayref': "y", 'text': "attacking direction", 'font_color': linecolor, 'xanchor': 'left', 'yanchor': 'top', 'font_size': 9, 'showarrow': True, 'arrowhead': 1, 'arrowwidth': 1, 'arrowcolor': linecolor})

    # Add text
    if caption != '':
        annotations.append({'font_color': linecolor, 'font_size': 16, 'x': 0, 'y': width + 6, 'showarrow': False,
                            'text': "{}".format(caption), 'textangle': 0,
                            'xanchor': 'left', 'yanchor': 'bottom'})
    if subcaption != '':
        annotations.append({'font_color': linecolor, 'font_size': 12, 'x': 0, 'y': width + 6, 'showarrow': False,
                            'text': "{}".format(subcaption), 'textangle': 0, 'xanchor': 'left', 'yanchor': 'top'})

    fig = go.Figure(data=datas, layout=layout)
    fig.update_layout(annotations=annotations)
    fig.update_shapes(layer='below')

    return fig, annotations


@st.cache(allow_output_mutation=True)
def length_of_passes(data, caption='', subcaption='', compare_team=''):
    if compare_team == '':
        teams = list(data['team'].unique())
        if '' in teams:
            teams.remove('')
    else:
        teams = [compare_team, 'Opponent']

    accurate_passes_length = {
        teams[0]: {'0-10': 0, '10-20': 0, '20-30': 0, '30-40': 0, '>40': 0},
        teams[1]: {'0-10': 0, '10-20': 0, '20-30': 0, '30-40': 0, '>40': 0}
    }

    nonaccurate_passes_length = {
        teams[0]: {'0-10': 0, '10-20': 0, '20-30': 0, '30-40': 0, '>40': 0},
        teams[1]: {'0-10': 0, '10-20': 0, '20-30': 0, '30-40': 0, '>40': 0}
    }

    average = {teams[0]: 0, teams[1]: 0}
    count = {teams[0]: 0, teams[1]: 0}

    for index, passes in data.loc[((data.event.str.contains('Pass')) | (data.event.str.contains('Cross')))].iterrows():
        pass_team = teams[0] if passes['team'] == teams[0] else teams[1]

        count[pass_team] += 1
        distance = np.sqrt((passes['x2'] - passes['x']) ** 2 + (passes['y2'] - passes['y']) ** 2)
        average[pass_team] += distance if distance >= 0 else 0

        if passes.outcome == 0:
            if distance <= 10:
                nonaccurate_passes_length[pass_team]['0-10'] += 1
            elif distance <= 20:
                nonaccurate_passes_length[pass_team]['10-20'] += 1
            elif distance <= 30:
                nonaccurate_passes_length[pass_team]['20-30'] += 1
            elif distance <= 40:
                nonaccurate_passes_length[pass_team]['30-40'] += 1
            else:
                nonaccurate_passes_length[pass_team]['>40'] += 1
        else:
            if distance <= 10:
                accurate_passes_length[pass_team]['0-10'] += 1
            elif distance <= 20:
                accurate_passes_length[pass_team]['10-20'] += 1
            elif distance <= 30:
                accurate_passes_length[pass_team]['20-30'] += 1
            elif distance <= 40:
                accurate_passes_length[pass_team]['30-40'] += 1
            else:
                accurate_passes_length[pass_team]['>40'] += 1

    norm = Normalize(vmin=0, vmax=4)
    edge_cmap = cm.get_cmap('Blues')

    datas = []
    annotations = []
    max_passes = 0
    # Make team 0 line chart with scatter
    team_ind = 0
    for team in teams:
        x_values = list(nonaccurate_passes_length[team].keys())
        y_values = list(nonaccurate_passes_length[team].values())
        if max(y_values) > max_passes:
            max_passes = max(y_values)
        x_accurate = list(accurate_passes_length[team].keys())
        y_accurate = list(accurate_passes_length[team].values())

        if max(y_accurate) > max_passes:
            max_passes = max(y_accurate)

        color = edge_cmap(norm(team_ind))
        line_color = f"rgba({color[0]}, {color[1]}, {color[2]}, 1)"
        fill_color = f"rgba({color[0]}, {color[1]}, {color[2]}, 0.5)"

        datas.append(go.Scatter(x=x_accurate, y=y_accurate, opacity=1, name=f"{team} accurate passes", hovertemplate=y_accurate, text=y_accurate, showlegend=True, mode='lines', marker_size=10, marker_color=line_color, marker_line_width=1, marker_line_color=line_color))
        datas.append(go.Scatter(fill='tozeroy', x=x_values, y=y_values, opacity=1, name=f"{team} non accurate passes", hovertemplate=y_values, text=y_values, showlegend=True, mode='lines', marker_size=10, marker_color=fill_color, marker_line_width=1, marker_line_color=fill_color, fillcolor=fill_color))
        team_ind += 1

    layout = go.Layout(
        font_color='black',
        showlegend=True,
        xaxis=dict(gridwidth=0.2, gridcolor='rgba(0, 0, 0, 0.1)', showline=True, showticklabels=True, showgrid=True, zeroline=True, visible=True),
        yaxis=dict(range=[0, max_passes+20], gridwidth=0.5, gridcolor='rgba(0, 0, 0, 0.1)', showline=True, showticklabels=True, showgrid=True, zeroline=True, visible=True, automargin=True),
        autosize=True, plot_bgcolor='white', paper_bgcolor='white', margin={'l': 10, 'r': 10, 't': 20, 'b': 20}
    )

    # Add text
    annotations.append({'font_color': 'black', 'font_size': 12, 'x': '>40', 'y': max_passes - 10, 'showarrow': False, 'yanchor': 'top',
                        'text': f"Team {teams[0]} made {count[teams[0]]} passes and average lengths of those was {round(average[teams[0]]/count[teams[0]], 2)} meters.<br />"
                                f"Team {teams[1]} made {count[teams[1]]} passes and average lengths of passes was {round(average[teams[1]]/count[teams[1]], 2)} meters.<br />",
                        'textangle': 0, 'xanchor': 'right'})

    if caption != '':
        annotations.append({'font_color': 'black', 'font_size': 16, 'x': '>40', 'y': max_passes + 10, 'showarrow': False, 'yanchor': 'bottom',
                            'text': "<b>{}</b>".format(caption), 'textangle': 0, 'xanchor': 'right'})
    if subcaption != '':
        annotations.append({'font_color': 'black', 'font_size': 12, 'x': '>40', 'y': max_passes + 8, 'showarrow': False, 'yanchor': 'top',
                            'text': "{}".format(subcaption), 'textangle': 0, 'xanchor': 'right'})

    fig = go.Figure(data=datas, layout=layout)
    fig.update_layout(annotations=annotations)
    fig.update_shapes(layer='below')

    return fig


@st.cache(allow_output_mutation=True)
def draw_scoring_steps(match_events, team, steps=1, scoring_chance=[], field={}, caption='', subcaption='', length=105, width=68, linecolor='white', fillcolor='#80B860'):
    # Set background and other information for background of field
    layout = go.Layout(
        font_color=linecolor,
        showlegend=False,
        xaxis=dict(showgrid=False, range=[-3, length+3], zeroline=False, visible=False),
        yaxis=dict(range=[-3, width+10], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1, automargin=True),
        autosize=True, plot_bgcolor=fillcolor, paper_bgcolor=fillcolor, margin={'l': 10, 'r': 10, 't': 20, 'b': 20}, shapes=field
    )

    annotations = []
    datas = []

    row = match_events.loc[(match_events['index'] == scoring_chance['index'])]
    row_index = list(match_events.index[(match_events['index'] == scoring_chance['index'])])

    if team == 'Opponent':
        team = row['team'].values[0]

    # Draw all other steps
    steps_information = pd.DataFrame(columns=['Number', 'Action', 'Name', 'Team', 'Result'])
    for steps_back in np.arange(0, steps+1):
        drawing_step = steps - steps_back if steps - steps_back > 0 else ''
        this_index = row_index[0]-steps_back
        back = match_events.iloc[this_index]

        steps_information.at[drawing_step, 'Number'] = int(back.Number)
        steps_information.at[drawing_step, 'Action'] = back.event
        steps_information.at[drawing_step, 'Name'] = back.PlayerName
        steps_information.at[drawing_step, 'Team'] = back.team

        if back.team == team:
            datas.append(go.Scatter(x=[back['x']], y=[back['y']], opacity=1, name="",
                                    hovertemplate=f"<b>Path to scoring chance</b><br /><i>Player:</i> {str(back.PlayerName)}<br /><i>Position:</i> {back.team}<br /><i>Event:</i> {back.event}",
                                    text=f"{drawing_step}", showlegend=False, mode='markers+text', textposition='top left',
                                    marker_size=10, marker_color=fillcolor, marker_line_width=1,
                                    marker_line_color=linecolor))

            if (back['x'] == back['x2']) and (back['y'] == back['y2']):
                next = match_events.iloc[this_index + 1]
                if next['team'] != team:
                    next['x'] = 105 - next['x']
                    next['y'] = 68 - next['y']

                steps_information.at[drawing_step, 'Result'] = next.event
                color = 'blue' if 'Goal' in next['event'] else linecolor
                annotations.append({'opacity': 1, 'x': next['x'], 'y': next['y'], 'ax': back['x'], 'ay': back['y'], 'xref': "x", 'yref': "y", 'axref': "x", 'ayref': "y", 'text': "", 'showarrow': True, 'arrowhead': 3, 'arrowwidth': 1.5, 'arrowcolor': color})
            elif np.isnan(back['x2']) and np.isnan(back['y2']):
                next = match_events.iloc[this_index + 1]
                if next['team'] != team:
                    next['x'] = 105 - next['x']
                    next['y'] =68 - next['y']

                steps_information.at[drawing_step, 'Result'] = next.event
                color = 'blue' if 'Goal' in next['event'] else linecolor
                annotations.append({'opacity': 1, 'x': next['x'], 'y': next['y'], 'ax': back['x'], 'ay': back['y'], 'xref': "x", 'yref': "y", 'axref': "x", 'ayref': "y", 'text': "", 'showarrow': True, 'arrowhead': 3, 'arrowwidth': 1.5, 'arrowcolor': color})
            else:
                steps_information.at[drawing_step, 'Result'] = 'Success'
                annotations.append({'opacity': 1, 'x': back['x2'], 'y': back['y2'], 'ax': back['x'], 'ay': back['y'], 'xref': "x", 'yref': "y", 'axref': "x", 'ayref': "y", 'text': "", 'showarrow': True, 'arrowhead': 3, 'arrowwidth': 1.5, 'arrowcolor': linecolor})
        else:
            break


    # Add text and attacking direction arrow
    annotations.append(
            {'opacity': 0.5, 'x': length/2 + 10, 'y': -2, 'ax': length/2 - 10, 'ay': -2, 'xref': "x", 'yref': "y", 'axref': "x",
             'ayref': "y", 'text': "attacking direction", 'font_color': linecolor, 'xanchor': 'left', 'yanchor': 'top', 'font_size': 9, 'showarrow': True, 'arrowhead': 1, 'arrowwidth': 1, 'arrowcolor': linecolor})

    # Add text
    if caption != '':
        annotations.append({'font_color': linecolor, 'font_size': 16, 'x': 0, 'y': width + 6, 'showarrow': False,
                            'text': "{}".format(caption), 'textangle': 0,
                            'xanchor': 'left', 'yanchor': 'bottom'})
    if subcaption != '':
        annotations.append({'font_color': linecolor, 'font_size': 12, 'x': 0, 'y': width + 6, 'showarrow': False,
                            'text': "{}".format(subcaption), 'textangle': 0, 'xanchor': 'left', 'yanchor': 'top'})

    fig = go.Figure(data=datas, layout=layout)
    fig.update_layout(annotations=annotations)
    fig.update_shapes(layer='below')

    return fig, annotations, steps_information


@st.cache(allow_output_mutation=True)
def draw_shots(df, field, caption='', subcaption='', length=105, width=68, type='', half=0, info_text=0, linecolor='white', fillcolor='#80B860'):
    # Set background and other information for background of field
    layout = go.Layout(
        font_color=linecolor,
        showlegend=False,
        xaxis=dict(showgrid=False, range=[-3, length+3], zeroline=False, visible=False),
        yaxis=dict(range=[-3, width+10], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1, automargin=True),
        autosize=True, plot_bgcolor=fillcolor, paper_bgcolor=fillcolor, margin={'l': 10, 'r': 10, 't': 20, 'b': 20}, shapes=field
    )

    datas = []
    annotations = []
    color = ['red', 'blue']
    multiply_size = 30
    size = 30

    if half == 0:
        x = 'x'
        y = 'y'
    else:
        x = 'xh'
        y = 'yh'

    # Add starting point with hover effect
    datas.append(go.Scatter(x=df[x], y=df[y], opacity=0.7, name="", hovertemplate='%{text}', text=[
        '<b>{}</b><br><i>Player:</i> {}<br><i>Event:</i> {}<br><i>xG:</i> {}'.format(type, r.PlayerName, r.event,
                                                                                     round(r.xG, 2)) for i, r in
        df.iterrows()], showlegend=False, mode='markers', marker_size=size * df.xG,
                            marker_color=[color[r['Goal']] for i, r in df.iterrows()], marker_line_width=1,
                            marker_line_color=linecolor))


    if info_text == 1:
        # information box about xG size
        info_y = width + 5
        info_x = length - 10
        xGs = np.arange(0.1, 0.6, 0.1) * multiply_size
        xGs_text = list(map(lambda v : round(v / multiply_size, 2), xGs))

        yi = []
        xi = []

        for i in np.arange(0, 5):
            yi.append(info_y)
            xi.append((info_x - 11) + 4.5 * i)

        datas.append(go.Scatter(x=xi, y=yi, opacity=0.7, name="", showlegend=False, mode='markers+text',
                                text=xGs_text, textposition='middle right', textfont=dict(color=linecolor, size=8),
                                marker_size=xGs, marker_color='rgba(255, 255, 255, 0)', marker_line_width=1,
                                marker_line_color=linecolor))

        # information box about goal or not
        xi = [info_x + 1, info_x + 6]
        yi = [info_y - 3, info_y - 3]

        datas.append(go.Scatter(x=xi, y=yi, opacity=0.7, name='', showlegend=False, mode='markers+text',
                                text=['TRUE', 'FALSE'], textposition='middle right', textfont=dict(color=linecolor, size=8),
                                marker_size=xGs[1], marker_color=['#4925e8', '#e82525'], marker_line_width=1,
                                marker_line_color=linecolor))


        # Statistical information
        stats_info_caption = "Shots<br />Goals<br />xG Sum<br />xG per shot<br />Shots per goal"
        stats_info_value = str(len(df)) + "<br />" + str(len(df.loc[(df.Goal == 1)])) +  "<br />" +  str(round(sum(df.xG), 2)) + "<br />" +  str(round(statistics.mean(df.xG), 2)) + "<br />" + str(round(divZero(len(df), len(df.loc[(df.Goal == 1)])), 2))
        annotations.append({'font_color': linecolor, 'font_size': 8, 'x': length - 4, 'y': 5, 'showarrow': False, 'text': "{}".format(stats_info_caption), 'textangle': 0, 'xanchor': 'right', 'align': 'right'})
        annotations.append({'font_color': linecolor, 'font_size': 8, 'x': length - 3, 'y': 5, 'showarrow': False, 'text': "{}".format(stats_info_value), 'textangle': 0, 'xanchor': 'left', 'align': 'left'})

    # Add text
    if caption != '':
        annotations.append({'font_color': linecolor, 'font_size': 16, 'x': 0, 'y': width + 6, 'showarrow': False,
                            'text': "{}".format(caption), 'textangle': 0,
                            'xanchor': 'left', 'yanchor': 'bottom'})
    if subcaption != '':
        annotations.append({'font_color': linecolor, 'font_size': 12, 'x': 0, 'y': width + 6, 'showarrow': False,
                            'text': "{}".format(subcaption), 'textangle': 0, 'xanchor': 'left', 'yanchor': 'top'})

    fig = go.Figure(data=datas, layout=layout)
    fig.update_layout(annotations=annotations)
    fig.update_shapes(layer='below')

    return fig, annotations, datas