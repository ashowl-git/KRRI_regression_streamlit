# 분석전에 필요한 라이브러리들을 불러오기

# plotly라이브러리가 없다면 아래 설치
# conda install -c plotly plotly=4.12.0
# conda install -c conda-forge cufflinks-py
# conda install seaborn
# download_modify_metro_regression_streamlit.py

import glob 
import os
import sys, subprocess
from subprocess import Popen, PIPE
import numpy as np

import pandas as pd

import math

import streamlit as st
import sklearn
import seaborn as sns
# sns.set(font="D2Coding") 
# sns.set(font="Malgun Gothic") 
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats("retina")
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go 
# import chart_studio.plotly as py
# import cufflinks as cf
# # get_ipython().run_line_magic('matplotlib', 'inline')
# 사이킷런 라이브러리 불러오기 _ 통계, 학습 테스트세트 분리, 선형회귀등
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error

from metro_regression_streamlit import DF4

st.dataframe(DF4)




