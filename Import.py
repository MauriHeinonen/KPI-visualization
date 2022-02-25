#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created: 11.11.2020

@author: Your name
Version: 1.0

Short description of your Python application.

Changes:
            11.11.2020  Short description of made changes to the script
"""


# Import packages that you need in your Python application.
# Order of imported packages should be first integrated packages,
# after that alphabetic order other packages and for last your made packages.

# Build-in modules
import base64                   # Provides encoding binary data to printable ASCII characters and decoding such encodings back to binary data
from datetime import datetime   # If you like to print or use some time information
from io import BytesIO
import json                     # Package for handling JSON -files (https://docs.python.org/3/library/json.html)
import math                     # Provides access to the mathematical functions (https://docs.python.org/3/library/math.html).
from pathlib import Path        # Use if you like to figure out some system paths (https://docs.python.org/3/library/pathlib.html).
import re                       # Package for regular expressions (https://docs.python.org/3/library/re.html)
import statistics               # Provides functions for calculating mathematical statistics of numeric (Real-valued) data (https://docs.python.org/3/library/statistics.html).
import time

# Other modules
"""
Package for visualization of data. If you like to make just images, this package is most useful and versatile for this.
(https://plotly.com/python/)
"""
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from matplotlib.patches import Arc
import matplotlib.cm as cm

"""
If you like to save your data for long-term follow, you should use databases. This package is for MySQL database connections.
(https://dev.mysql.com/doc/connector-python/en/)
"""
import mysql.connector
from mysql.connector import errorcode

import numpy as np                      # https://numpy.org/learn/
import pandas as pd                     # Easy tool for large data handling in Python (https://pandas.pydata.org/docs/)

"""
Package for visualization of data. If you use the Streamlit package and like to make interactive images, you should use these packages.
(https://plotly.com/python/)
"""
import plotly.graph_objs as go
import plotly.express as px

import statsmodels.api as sm            # Python module that provides classes and functions for the estimation of many different statistical models. Basic module
import statsmodels.formula.api as smf   # Python module that provides classes and functions for the estimation of many different statistical models. Using different formulas.
import streamlit as st                  # Framework for interactive web application, which is made with Python

# Self made modules



