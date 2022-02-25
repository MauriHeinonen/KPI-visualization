#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:28:26 2021

@author: mheinone
"""

import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import base64
import json
from io import BytesIO

def getDatabaseConnection():
    """
    This function makes a connection to the MySQL database. Basically, this starts MySQL connect.

    Returns
    -------
    Connection variable, what you can use to different database queries.
    """
    engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user="root", pw="password", db="data_library"))
    connect = engine.connect()
    return connect

def readExcelFileInfo(filename, sheet):
    excel_df = pd.read_excel (filename, sheet_name=sheet)
    excel_df = excel_df.replace("-", 0)
    excel_df = excel_df.replace(np.nan, "")
    excel_df = excel_df.replace("%", "")
    
    return(excel_df)

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

def parseJSON(data, columns=[]):
    parsed_json = pd.DataFrame()
    df_columns = list(data[0][0].keys())

    if ( len(columns) > 0 ):
        if ( len(columns) > len(df_columns) ):
            res_columns = columns
        else:
            res_columns = list(filter(lambda i: i in columns, df_columns))
    else:
        res_columns = df_columns
    
    i = 0
    numeric_columns = []
    for value in data:
        for row in value:
            for df_c in res_columns:
                if ( df_c in list(row.keys()) ):
                    if ( row[df_c].isnumeric() & (df_c not in numeric_columns )):   
                        numeric_columns.append(df_c)

                    parsed_json.at[i, df_c] = row[df_c]
            i += 1

    parsed_json = parsed_json.replace(np.nan, '', regex=True)
    parsed_json['id'] = parsed_json['id'].astype(int)
    return(parsed_json)

def deleteDuplicates(table, results_final, con) -> object:
    new_results = results_final
    tables = pd.read_sql(f"SHOW TABLES LIKE '{table}'", con)
    if ( len(tables) > 0 ):
        old_results = pd.read_sql(f"SELECT * FROM {table}", con)
        if ( len(old_results) > 0 ):
            df = pd.merge(old_results, results_final, how='outer', indicator=True)
            new_results = df[df['_merge']=='right_only'][results_final.columns]

    return(new_results)

def deleteOldRows(table, match_id, con) -> object:
    deleted_rows = []
    tables = pd.read_sql(f"SHOW TABLES LIKE '{table}'", con)
    if ( len(tables) > 0 ):
        for match in match_id:
            sql_del = f"DELETE FROM {table} WHERE IdMatch = {match}"
            rs = con.execute(sql_del)
            deleted_rows.append(rs)

    return(deleted_rows)

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df, filename='', hreftext='Download file'):
    """
    Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">{hreftext}</a>'


def saveJSON(filename, df):
    with open(filename, 'w') as json_file:
        json.dump(df, json_file)


def make_24_areas(x, y, length=105, width=68):
    """

    """
    area = 0
    bin_length = length / 6
    bin_width = width / 4
    if np.isnan(x) or np.isnan(y):
        area = np.nan
    else:
        x_col = x // bin_length + 1 if x % bin_length > 0 or x == 0 else x // bin_length
        y_row = y // bin_width + 1 if y % bin_width > 0 or y == 0 else y // bin_width

        area = (x_col * 4 - y_row) + 1

    return(area)