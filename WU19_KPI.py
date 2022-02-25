import matplotlib.pyplot as plt

from ExtraFunctions import *
import DrawVisualisations as dv
import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib import font_manager
import math

import instat_functions as ifunc

def parseJSONFile(df):
    parsedFile = pd.DataFrame()

    i = 0
    for key, value in df.iterrows():
        for row in value['rows']:
            for highlights in row['highlights']:
                parsedFile.at[i, 'Name'] = row['name']
                parsedFile.at[i, 'Position'] = highlights['start'] * 1000
                parsedFile.at[i, 'Duration'] = (highlights['end'] - highlights['start']) * 1000
                for event in highlights['events']:
                    pref = event['name'].split(':')
                    if (len(pref) > 1):
                        pref_key = pref[0]
                        pref_value = pref[1]
                        # if ( pref_value.isdigit() ):
                        if (re.match('^[0-9\.]*$', pref_value) and (('x' in pref_key) or ('y' in pref_key))):
                            pref_value = float(pref_value)

                        if (pref_key in ['Open', 'Penetration', 'Result', 'SP'] and pref_key in parsedFile.columns):
                            if (not pd.isnull(parsedFile[pref_key].iloc[i])):
                                pref_value = pref_value + ":" + parsedFile.at[i, pref_key]

                        parsedFile.at[i, pref_key] = pref_value
                    else:
                        parsedFile.at[i, pref[0]] = True
                i += 1

    parsedFile = parsedFile.replace(np.nan, '', regex=True)
    parsedFile = parsedFile.rename(columns={'\u2028': 'NL'})

    home_team = np.unique(parsedFile['Home'])
    home_team = np.delete(home_team, np.where(home_team == ''))[0]
    away_team = np.unique(parsedFile['Away'])
    away_team = np.delete(away_team, np.where(away_team == ''))[0]

    conditions = [
        (parsedFile['Name'].str.contains(home_team)),
        (parsedFile['Name'].str.contains(away_team))
    ]

    values = [home_team, away_team]
    parsedFile['Team'] = np.select(conditions, values)
    parsedFile['action'] = parsedFile['Name'].str.replace('{} - '.format(home_team), '')
    parsedFile['action'] = parsedFile['action'].str.replace('{} - '.format(away_team), '')
    parsedFile = parsedFile.sort_values(by=['Position'])
    parsedFile.index = np.arange(0, len(parsedFile))

    return parsedFile


@st.cache(allow_output_mutation=True)
def passes_to_box(df, team, field, field_size='half', caption='', subcaption='', length=65, width=50, ball_missed=5, scoring_chance=20, ball_possession=15, linecolor='white', fillcolor='#80B860'):
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

    start_x = 'x_tb' if field_size == 'half' else 'x'
    start_y = 'y_tb' if field_size == 'half' else 'y'
    end_x = 'x_tb_dest' if field_size == 'half' else 'x2'
    end_y = 'y_tb_dest' if field_size == 'half' else 'y2'

    to_box_line = 7.32/2 + 16.5
    box_start_x = length / 2 - to_box_line if field_size == 'half' else length - 16.5
    box_end_x = length / 2 + to_box_line if field_size == 'half' else length
    box_start_y = 0 if field_size == 'half' else width / 2 - to_box_line
    box_end_y = 16.5 if field_size == 'half' else width / 2 + to_box_line

    events = df.loc[(df.Team == team)]

    events = events.loc[(events[end_x].between(box_start_x, box_end_x) & events[end_y].between(box_start_y, box_end_y))]
    count_of_events = len(events)

    # Add starting point with hover effect
    datas.append(go.Scatter(x=events[start_x], y=events[start_y], opacity=0.7, name="", hovertemplate='', text='', showlegend=False, mode='markers', marker_size=size,
                            marker_color=[
                                what_color_action(how_many_seconds_to_next_opp_action(df, i, 1), ball_missed,
                                                     scoring_chance, ball_possession) for i, r in
                                events.iterrows()], marker_line_width=1, hoverinfo='none',
                            marker_line_color=linecolor))

    # Add attacking direction arrow
    if field_size == 'full':
        annotations.append(
                {'opacity': 0.5, 'x': length/2 + 10, 'y': -2, 'ax': length/2 - 10, 'ay': -2, 'xref': "x", 'yref': "y", 'axref': "x",
                 'ayref': "y", 'text': "attacking direction", 'font_color': linecolor, 'xanchor': 'left', 'yanchor': 'top', 'font_size': 9, 'showarrow': True, 'arrowhead': 1, 'arrowwidth': 1, 'arrowcolor': linecolor})

    for ai, ar in events.iterrows():
        annotations.append(
            {'opacity': 0.5, 'x': ar[end_x], 'y': ar[end_y], 'ax': ar[start_x], 'ay': ar[start_y], 'xref': "x", 'yref': "y", 'axref': "x",
             'ayref': "y", 'text': "", 'showarrow': True, 'arrowhead': 3, 'arrowwidth': 1.5, 'arrowcolor': what_color_action(how_many_seconds_to_next_opp_action(df, ai, 1), ball_missed, scoring_chance, ball_possession)})

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
    fig.update_traces(hovertemplate=None)
    fig.update_shapes(layer='below')

    return fig, annotations


@st.cache(allow_output_mutation=True)
def passes_between_lines(df, team, field, version='dline', press='mid', caption='', subcaption='', length=65, width=50, ball_missed=5, scoring_chance=20, ball_possession=15, linecolor='white', fillcolor='#80B860'):
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
    fcolor = 'grey'
    lcolor = 'grey'
    start_x = 'x'
    start_y = 'y'
    end_x = 'x_dest'
    end_y = 'y_dest'

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

    formation_x = [97 + goalie_press, 81.5 + dline_press, 81.5 + dline_press, 81.5 + dline_press, 81.5 + dline_press, 52 + mline_press, 52 + mline_press, 52 + mline_press, 52 + mline_press, 29.5 + sline_press, 29.5 + sline_press]
    formation_y = [34, 54.4, 40.8, 27.2, 13.6, 54.4, 40.8, 27.2, 13.6, 40.8, 27.2]
    events = df.loc[(df.Team == team)]

    if version == 'dline':
        events = events.loc[((events.Open.str.contains('Higher Pockets') | events.Open.str.contains('10 area')) & (events.Press.str.lower() == press))]
    else:
        events = events.loc[((events.Open.str.contains('Lower Pockets') | events.Open.str.contains('6 area')) & (events.Press.str.lower() == press))]

    count_of_events = len(events)

    # Add starting point with hover effect
    datas.append(go.Scatter(x=events[start_x], y=events[start_y], opacity=0.7, name="", hovertemplate='%{text}', text=[
        '<b>{}</b><br><i>Team:</i> {}'.format(type, r.Team) for i, r in
        events.iterrows()], showlegend=False, mode='markers', marker_size=size,
                            marker_color=[
                                what_color_action(how_many_seconds_to_next_opp_action(df, i, 1), ball_missed,
                                                     scoring_chance, ball_possession) for i, r in
                                events.iterrows()], marker_line_width=1, hoverinfo='none',
                            marker_line_color=linecolor))

    datas.append(go.Scatter(x=formation_x, y=formation_y, opacity=0.9, name="Player", text='', showlegend=False, mode='markers', marker_size=30, marker_color=fcolor, marker_line_width=1, hoverinfo='none', marker_line_color=fcolor))

    # Add attacking direction arrow
    annotations.append(
            {'opacity': 0.5, 'x': length/2 + 10, 'y': -2, 'ax': length/2 - 10, 'ay': -2, 'xref': "x", 'yref': "y", 'axref': "x",
             'ayref': "y", 'text': "attacking direction", 'font_color': linecolor, 'xanchor': 'left', 'yanchor': 'top', 'font_size': 9, 'showarrow': True, 'arrowhead': 1, 'arrowwidth': 1, 'arrowcolor': linecolor})

    for ai, ar in events.iterrows():
        annotations.append(
            {'opacity': 0.5, 'x': ar[end_x], 'y': ar[end_y], 'ax': ar[start_x], 'ay': ar[start_y], 'xref': "x", 'yref': "y", 'axref': "x",
             'ayref': "y", 'text': "", 'showarrow': True, 'arrowhead': 3, 'arrowwidth': 1.5, 'arrowcolor': what_color_action(how_many_seconds_to_next_opp_action(df, ai, 1), ball_missed, scoring_chance, ball_possession)})

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
    fig.update_traces(hovertemplate=None)
    fig.update_shapes(layer='below')

    return fig, annotations


@st.cache(allow_output_mutation=True)
def draw_interception(df, team='', field='', caption='', subcaption='', length=105, width=68, type='', ball_missed=5, scoring_chance=20, ball_possession=15, info_text=0,  linecolor='white', fillcolor='#80B860'):
    df_copy = df.copy()
    df_copy.index = np.arange(0, len(df_copy))

    interceptions = df.loc[((df.action == 'Interception') & (df.team == team))]

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
                                what_color_action(how_many_seconds_to_next_opp_action(df, i, 1), ball_missed,
                                                     scoring_chance, ball_possession) for i, r in
                                interceptions.iterrows()], marker_line_width=1, hoverinfo='none', marker_line_color=linecolor))

    # Average line where attacks are started
    datas.append(go.Scatter(x=[x_mean, x_mean], y=[0, width],
                               mode='lines', line={'dash': 'dash', 'width': 2, 'color': 'red'}, showlegend=False, opacity=0.8, text=f"Average distance from own goal where attacks are started is {round(x_mean, 2)} meters.",
                               hovertemplate=f"Average distance from own goal where attacks are started is {round(x_mean, 2)} meters.", hoverinfo='none', xaxis='x', yaxis='y'))

    # Add captions
    if caption != '':
        annotations.append({'font_color': linecolor, 'font_size': 16, 'x': 0, 'y': width + 6, 'showarrow': False,
                            'text': "{} (count of interceptions {})".format(caption, count_of_interceptions), 'textangle': 0,
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
    fig.update_traces(hovertemplate=None)
    fig.update_shapes(layer='below')

    return fig, annotations, count_of_interceptions


@st.cache(allow_output_mutation=True)
def xG(df, all=0):
    with st.spinner('Calculating xG values...'):
        data = df.copy(deep=True)
        data = data.loc[(data.Penetration.str.contains('Shoot') | data.Penetration.str.contains('Shot'))]
        # drop all another game phases off from dataframe

        if all == 0:
            delete_set_pieces = data[data.SP != ''].index
            data.drop(delete_set_pieces, inplace=True)

        shots_model = pd.DataFrame(
            columns=['Goal', 'X', 'Y', 'Distance', 'Angle', 'Degree', 'Result', 'SP', 'Layer', 'gender'])

        # return empty dataframe
        if len(data) == 0:
            return shots_model

        a = [(65 / 2) - (7.32 / 2), 0]  # Goal left post
        c = [(65 / 2) + (7.32 / 2), 0]  # Goal right post

        for i, row in data.iterrows():
            shots_model.at[i, 'X'] = row['xh']
            shots_model.at[i, 'Y'] = row['yh']

            # Distance of the center spot
            shots_model.at[i, 'C'] = abs(np.sqrt((row['x'] - 105 / 2) ** 2 + (row['y'] - 68 / 2) ** 2))

            # Distance in metres and shot angle in radians.
            shots_model.at[i, 'Distance'] = np.sqrt(
                (shots_model.at[i, 'X'] - 65 / 2) ** 2 + (shots_model.at[i, 'Y']) ** 2)
            b = [shots_model.at[i, 'X'], shots_model.at[i, 'Y']]  # Shot location
            ang = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
            if ang < 0:
                ang = np.pi + ang

            shots_model.at[i, 'Angle'] = ang
            shots_model.at[i, 'Degree'] = math.degrees(ang)
            shots_model.at[i, 'Result'] = row['Result']
            shots_model.at[i, 'SP'] = row['SP']

            # if ( 'Team' in row.columns ):
            #    shots_model.at[i,'Gendre'] = 'Women' if 'Women' in row['Team'] else 'Men'

            # Was this shot a goal
            is_goal = 1 if 'Goal' in row['Result'] else 0
            shots_model.at[i, 'Goal'] = is_goal

        # Original model is made with Angle and Distance
        # model_variables = ['Angle','Distance','BodyPart','Gendre','SP'] #,'X','C']
        model_variables = ['Angle', 'Distance']  # ,'X','C']

        # Fit the model
        # test_model = smf.glm(formula="Goal ~ " + ' + '.join(model_variables), data=shots_model,family=sm.families.Binomial()).fit()
        # print(test_model.summary())
        # print(test_model.params)
        # b = test_model.params

        b = [3, -3, 0]

        # Add an xG to my dataframe
        xG = shots_model.apply(calculate_xG, axis=1, args=(model_variables, b))

        shots_model = shots_model.assign(xG=xG)
        df = df.assign(xG=xG)
        df = df.assign(Goal=shots_model.Goal)

        return df


@st.cache
def calculate_xG(sh, model_variables, b):
    bsum = b[0]
    for i,v in enumerate(model_variables):

        bsum = bsum + b[i + 1] * sh[v]

    if ( 'Penalty Kick' in sh['SP'] ):
        xG = 0.75
    else:
        xG = 1 / (1 + np.exp(bsum))

    return xG


def cumulative_xg(df, home_team, away_team):
    cumulative = pd.DataFrame()

    # Install Finlandica FONT if it is not installed to computer
    fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

    r = re.compile(".*Finlandica.*")
    finlandica = list(filter(r.match, fonts))

    # Add Finlandica font to use
    for font_file in finlandica:
        font_manager.fontManager.addfont(font_file)

    if len(list(filter(r.match, fonts))) == 0:
        print("EI LÖYDY!!!")
    else:
        # Set font family globally
        plt.rcParams['font.family'] = 'Finlandica'

        # Set font color to FAF blue
        plt.rcParams['text.color'] = '#002858'
        plt.rcParams['axes.labelcolor'] = '#002858'
        plt.rcParams['xtick.color'] = '#002858'
        plt.rcParams['ytick.color'] = '#002858'

    index = 0
    last_index = []
    for team_name in [home_team, away_team]:
        cumulative.at[index, 'team'] = team_name
        cumulative.at[index, 'time'] = 0
        cumulative.at[index, 'cumulative xG'] = 0
        data = df.loc[((df.team == team_name) & (df.Penetration.str.contains('Shot') | df.Penetration.str.contains('Shoot')))]
        cumulative_xg = 0
        index += 1
        for i, row in data.iterrows():
            cumulative.at[index, 'team'] = row.team
            cumulative.at[index, 'time'] = row.time / 1000 / 60
            cumulative_xg += row.xG
            cumulative.at[index, 'cumulative xG'] = cumulative_xg
            index += 1

        last_index.append(index - 1)

    max_time = cumulative.time.max()
    max_time = (5 - (max_time % 5)) + max_time
    new_index = 0
    for team_name in [home_team, away_team]:
        cumulative.at[index + new_index, 'team'] = team_name
        cumulative.at[index + new_index, 'time'] = max_time
        cumulative.at[index + new_index, 'cumulative xG'] = cumulative.at[last_index[new_index], 'cumulative xG']
        new_index += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#002858')
    ax.spines['left'].set_color('#002858')
    ax.tick_params(axis='both', length=0, labelsize=12)

    ax.step(x=cumulative.loc[cumulative.team == home_team]['time'], y=cumulative.loc[cumulative.team == home_team]['cumulative xG'], where='post', color='#002858', label=home_team)
    ax.step(x=cumulative.loc[cumulative.team == away_team]['time'], y=cumulative.loc[cumulative.team == away_team]['cumulative xG'], where='post', color='#A99B70', label=away_team)
    x_pos = np.arange(0, max_time, 5)
    plt.xlim(0, max_time)
    plt.ylim(float(0.0), float(cumulative['cumulative xG'].max()))
    ax.legend(frameon=False)

    plt.suptitle(f"CUMULATIVE XG", fontsize=16, fontweight='bold', wrap=True, ha='center', va='bottom')

    return fig, ax


# Start streamlit software  -  Pages head info, like title which is showed in head
st.set_page_config(
    page_title='Stats API gateway to download data to local computer',
    page_icon='favicon.ico',
    layout='wide',
    initial_sidebar_state='collapsed'
)

ifunc.local_css('style_wu19.css')
st.markdown(
    '<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.11.2/css/all.css">',
    unsafe_allow_html=True
)



event_files = st.sidebar.file_uploader("SELECT MATCH EVENT FILE:",  type="json", accept_multiple_files=True)


if len(event_files) > 0:
    all_matches = pd.DataFrame()
    config = {'displayModeBar': False}
    FIELD_LINE_COLOR = 'white'
    FIELD_BG_COLOR = '#171716'
    ball_missed = 5
    scoring_chance = 20
    ball_possession = 15

    for json_file in event_files:
        df = parseJSONFile(pd.json_normalize(json.load(json_file)))
        df['team'] = df.Team
        df['time'] = df.Position
        df['event'] = df.action
        df['PlayerName'] = ''

        for column in ['action', 'Team', 'team']:
            df[f"prev_{column}"] = df[column].shift(1)
            df[f"next_{column}"] = df[column].shift(-1)


        home_team = df.Home.unique()
        away_team = df.Away.unique()
        tournament = df.Tournament.unique()
        nt = df.NT.unique()
        date = df.Date.unique()
        place = df.Place.unique()

        home_team = np.delete(home_team, np.where(home_team == ''))[0]
        away_team = np.delete(away_team, np.where(away_team == ''))[0]
        tournament = np.delete(tournament, np.where(tournament == ''))[0]
        nt = np.delete(nt, np.where(nt == ''))[0]
        date = np.delete(date, np.where(date == ''))[0]
        place = np.delete(place, np.where(place == ''))[0]

        for column in ['x', 'y', 'x_tb', 'y_tb', 'x_tb_dest', 'y_tb_dest', 'y_dest', 'x_dest', 'Half']:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column])

        df = df.replace(np.nan, 0, regex=True)

        home_goal_tot = len(df.loc[(df.Result.str.contains('Goal') & (df.Team == home_team))])
        away_goal_tot = len(df.loc[(df.Result.str.contains('Goal') & (df.Team == away_team))])

        period_goals = ''
        if 'Half' in df.columns:
            home_goal_1st = len(df.loc[((df.Result.str.contains('Goal') & (df.Team == home_team)) & (df.Half == 1))])
            away_goal_1st = len(df.loc[((df.Result.str.contains('Goal') & (df.Team == away_team)) & (df.Half == 1))])
            period_goals = f"[{home_goal_1st} - {away_goal_1st}]"


        # show statistics
        st.markdown(
            f"<div style='margin: -8rem -5rem; position: fixed; opacity: 0.3; width: 100%; height: 100%; z-index: 0; background-image: url({image_to_bytes('background.png', 0)});  background-repeat: no-repeat; background-position: right top; background-size: cover;'>&nbsp;</div>"
            #f"<div style='margin: -14rem -4rem; position: fixed; opacity: 1; transform: rotate(5deg); width: 100vw; height: 200px; z-index: 15; background-image: url({image_to_bytes('FA_logo.png', 0)});  background-repeat: no-repeat; background-position: right top; background-size: 155px;'>&nbsp;</div>"
            "<div id='caption'>"
            f"<div id='goals'><div id='total_goals'>{home_goal_tot} - {away_goal_tot}</div>"
            f"<div id='period_goals'>{period_goals}</div></div>"
            "<h1 style='color: white;'>"
            f"<div style='color: white; font-size: 3.5rem; padding-bottom: 0px;'>{home_team} - {away_team}</div><br />"
            f"{nt} • {tournament} • {place}<br />"
            f"<div style='color: white; font-weight: 100;'>{date}</div>"
            "</h1></div>",
            unsafe_allow_html=True
        )

        max_col = 4
        col = st.columns(max_col)
        info = {}
        first_third = 105 / 3
        goal_area_x_left = 68 / 2 - 16.5
        goal_area_x_right = 68 / 2 + 16.5

        ration_x = 65 / 68
        df['xh'] = df['y'].apply(lambda x: x * ration_x if x != "" else "")
        df['yh'] = df['x'].apply(lambda x: 105 - x if x != "" else "")

        df = xG(df, 1)
        df = df.replace(np.nan, 0, regex=True)

        # Drop rows off what we don't need (Drop all unnecessary rows from parsed_json dataframe)
        for delete_row in [f"Offence {home_team}", f"Offence {away_team}", f"Defence {home_team}", f"Defence {away_team}"]:
            df = df.drop(df.loc[(df.action == delete_row)].index, inplace=False)

        # Regenerate dataframe indexes
        df.index = np.arange(0, len(df))

        for team in [home_team, away_team]:

            xG = df.loc[(df.Team == team)]['xG'].sum() if 'xG' in df.columns else 0
            box_penetration = len(df.loc[((df.x_tb_dest.between(goal_area_x_left, goal_area_x_right) & (df.y <= 16.5)) & (df.Team == team))]) if 'x_tb_dest' in df.columns else 0
            success_key_passes = 0
            passes = len(df.loc[(df.action.str.contains('Pass') & (df.Team == team))])
            pass_pros = round(len(df.loc[(df.AR.str.contains('Success') & (df.Team == team))]) / passes * 100, 2) if 'AR' in df.columns and passes > 0 else 0.0
            scoring_chances = len(df.loc[(df.Info.str.contains('Highlight') & (df.Team == team))]) if 'Info' in df.columns else 0
            lost_ball_in_own_third = len(df.loc[((df.x <= first_third) & (df.action == 'Interception') & (df.Team == team))]) if 'x' in df.columns else 0
            shots = len(df.loc[(df.Penetration.str.contains('Shot') & (df.Team == team))])
            xG_shots = xG / shots if shots > 0 else 0
            goal_tot = len(df.loc[(df.Result.str.contains('Goal') & (df.Team == team))])
            shot_conversion = round(shots / goal_tot,2) if goal_tot > 0 else 0.0

            info[team] = {'xG': round(xG, 2),
                          'xG/Laukaus': round(xG_shots, 2),
                          'Laukaukset': shots,
                          'Laukaukset maalia kohti': len(df.loc[(df.Penetration.str.contains('Shot') & ~df.Result.str.contains('Shot over') & (df.Team == team))]),
                          'Laukausta / Maali': shot_conversion,
                          'Avainsyötöt': len(df.loc[(df.Penetration.str.contains('Key pass') & (df.Team == team))]),
                          #'Onnistuneet avainsyötöt': success_key_passes,
                          'Maalipaikkoja': scoring_chances + len(df.loc[(df.Penetration.str.contains('Shot') & (df.Team == team))]),
                          'Syötöt': passes,
                          'Syöttöprosentti': 0,
                          'Pallon menetykset omalla kolmanneksella': lost_ball_in_own_third,
                          'Eteneminen puolustuslinjan eteen': len(df.loc[((df.Open.str.contains('Higher Pockets') | df.Open.str.contains('10 area')) & (df.Team == team))]),
                          'Boksiin murtautumiset': box_penetration,
                          'Sivurajaheitot': len(df.loc[(df.SP.str.contains('Throw In') & (df.Team == team))]),
                          'Kulmat': len(df.loc[(df.SP.str.contains('Corner') & (df.Team == team))]),
                          'Vapaapotkut': len(df.loc[(df.SP.str.contains('FK') & (df.Team == team))]),
                          'Rangaistuspotkut': len(df.loc[(df.SP.str.contains('Penalty Kick') & (df.Team == team))]),
                          }

        i = 0
        # ['xG', 'xG/Laukaus', 'Laukaukset', 'Laukaukset maalia kohti', 'Avainsyötöt', 'Onnistuneet avainsyötöt', 'Maalipaikkoja', 'Syötöt', 'Syöttöprosentti', 'Pallon menetykset omalla kolmanneksella', 'Eteneminen puolustuslinjan eteen', 'Boksiin murtautumiset']
        for value in ['xG', 'xG/Laukaus', 'Laukaukset', 'Laukaukset maalia kohti', 'Laukausta / Maali', 'Avainsyötöt', 'Maalipaikkoja', 'Pallon menetykset omalla kolmanneksella', 'Eteneminen puolustuslinjan eteen', 'Boksiin murtautumiset', 'Syötöt', 'Syöttöprosentti', 'Sivurajaheitot', 'Kulmat', 'Vapaapotkut', 'Rangaistuspotkut']:
            col[i].markdown(
                f"<div id='info_box'>"
                f"<span class='heading'>{value}</span>"
                f"<span class='info'>{info[home_team][value]} - {info[away_team][value]}"
                f"</div>",
                unsafe_allow_html=True
            )

            i = i + 1 if i + 1 < max_col else 0

        st.markdown("<p>&nbsp;</p>", unsafe_allow_html=True)

        st.markdown(f"<center><span style='color: gray;'>Ball possession lost in {ball_missed} seconds</span> • "
                    f"<span style='color: {OTHER_COLOR};'>Ball possession continued for {ball_missed} to {ball_possession} seconds</span> • "
                    f"<span style='color: {BALL_POSSESSION_COLOR};'>Ball possession continued for at least {ball_possession} seconds</span> • "
                    f"<span style='color: {SCORING_CHANCE_COLOR};'>Scoring chance created in {scoring_chance} seconds</span></center>", unsafe_allow_html=True)
        USED_COLUMN = 0
        c = st.columns(2)
        half_field = dv.draw_half_pitch(50, 65)
        full_field = dv.draw_full_pitch(105, 68)

        for team in [home_team, away_team]:
            if 'x_tb_dest' in df.columns:
                (fig, annotations) = passes_to_box(df, team, half_field, 'half', f"Penetrations to the box by {team}", f"From match {home_team} - {away_team}", 65, 50, ball_missed, scoring_chance, ball_possession, FIELD_LINE_COLOR, FIELD_BG_COLOR)
                c[USED_COLUMN].plotly_chart(fig, use_container_width=True, sharing='streamlit', config=config, key=f"{team}_{USED_COLUMN}")
                c[USED_COLUMN].markdown("<div style='width: 100%; text-align: center; margin-top: -2.7rem;'>" + download_image(fig, f'to_box_{team}.png', f"<i class='fas fa-download'></i> Download image as a PNG file to your computer") + "</div>", unsafe_allow_html=True)

            if 'x' in df.columns:
                (fig, annotations, count_of_interceptions) = draw_interception(df, team, full_field, f"Interceptions by {team}", f"From match {home_team} - {away_team}", 105, 68, 'Interception', ball_missed, scoring_chance, ball_possession, 0, FIELD_LINE_COLOR, FIELD_BG_COLOR)
                fig.update_layout(annotations=annotations, shapes=full_field, plot_bgcolor=FIELD_BG_COLOR, paper_bgcolor=FIELD_BG_COLOR)
                c[USED_COLUMN].plotly_chart(fig, use_container_width=True, sharing='streamlit', config=config)
                c[USED_COLUMN].markdown("<div style='width: 100%; text-align: center; margin-top: -2.7rem;'>" + download_image(fig, f'interceptions_{team}.png', "<i class='fas fa-download'></i> Download interceptions as a PNG file to your computer") + "</div>", unsafe_allow_html=True)

            if 'x' in df.columns:
                team_shots = df[((df.Team == team) & (df.Penetration.str.contains('Shot') | df.Penetration.str.contains('Shoot')))]
                (fig, annotations, datas) = dv.draw_shots(team_shots, half_field, f"Shots of {team}", f"From match {home_team} - {away_team}", 65, 50, 'Shots', 1, 1)
                fig.update_layout(annotations=annotations, shapes=half_field, plot_bgcolor=FIELD_BG_COLOR, paper_bgcolor=FIELD_BG_COLOR)
                c[USED_COLUMN].plotly_chart(fig, use_container_width=True, sharing='streamlit', config=config)
                c[USED_COLUMN].markdown("<div style='width: 100%; text-align: center; margin-top: -2.7rem;'>" + download_image(fig, f'shots_{team}.png', "<i class='fas fa-download'></i> Download shot map as a PNG file to your computer") + "</div>", unsafe_allow_html=True)

            if 'x_dest' in df.columns:
                for press in ['low', 'mid', 'high']:
                    (fig, annotations) = passes_between_lines(df, team, full_field, 'dline', press, f"Passes between the lines {team} agains {press} block", f"From match {home_team} - {away_team}", 105, 68, ball_missed, scoring_chance, ball_possession, FIELD_LINE_COLOR, FIELD_BG_COLOR)
                    c[USED_COLUMN].plotly_chart(fig, use_container_width=True, sharing='streamlit', config=config, key=f"{team}_{press}_{USED_COLUMN}")
                    c[USED_COLUMN].markdown("<div style='width: 100%; text-align: center; margin-top: -2.7rem;'>" + download_image(fig, f'to_box_{team}.png', f"<i class='fas fa-download'></i> Download image as a PNG file to your computer") + "</div>", unsafe_allow_html=True)

                #for press in ['low', 'mid', 'high']:
                #    (fig, annotations) = passes_between_lines(df, team, full_field, 'mline', press, f"Pass to front of the midfield {team} agains {press} block", f"From match {home_team} - {away_team}", 105, 68, ball_missed, scoring_chance, ball_possession, FIELD_LINE_COLOR, FIELD_BG_COLOR)
                #    c[USED_COLUMN].plotly_chart(fig, use_container_width=True, sharing='streamlit', config=config, key=f"{team}_{press}_{USED_COLUMN}")
                #    c[USED_COLUMN].markdown("<div style='width: 100%; text-align: center; margin-top: -2.7rem;'>" + download_image(fig, f'to_box_{team}.png', f"<i class='fas fa-download'></i> Download image as a PNG file to your computer") + "</div>", unsafe_allow_html=True)

            USED_COLUMN += 1

        (fig, ax) = cumulative_xg(df, home_team, away_team)
        st.pyplot(fig, dpi=200, clear_figure=True)
        plt.close(fig)


else:
    st.markdown(
        f"<div style='margin: -7rem -5rem; position: fixed; opacity: 0.3; width: 100%; height: 100%; z-index: 0; background-image: url({image_to_bytes('background_image.png', 0)});  background-repeat: no-repeat; background-position: right top; background-size: cover;'>&nbsp;</div>"
        "<div id='caption'>"
        "<h1 style='color: white;'>"
        "SINUN TULEE VALITA YKSITTÄISEN OTTELUN EVENT DATA TAI USEAMMAN OTTELUN EVENT DATA, JOTTA VOIMME LUODA VISUALISOINNIT JA TILASTOT<br />"
        "Avaa sivuvalikko ja valitse halutut event data tiedostot (JSON-muodossa)."
        "</h1></div>",
        unsafe_allow_html=True
    )