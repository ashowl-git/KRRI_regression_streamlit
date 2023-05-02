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

#필요한 데이터 불러오기
df1 = pd.read_excel('data/일사량DB.xlsx')
df2 = pd.read_excel('data/경사일사량DB.xlsx')
df3 = pd.read_excel('data/맑은날DB.xlsx')

#사이드메뉴바 만들기
st.sidebar.header('Specify Input Parameters')
#집광면적
LENGTH = st.sidebar.number_input('LENGTH (mm)', 0, 5000, 1300)
WIDTH = st.sidebar.number_input('WIDTH (mm)', 0, 5000, 1300)
EA = st.sidebar.number_input('EA', 0, 100000, 100)
집광면적 = LENGTH*WIDTH*EA/1000000
#설비용량
설치용량 = st.sidebar.number_input('설치용량 [W]', 0, 1000, 450)
설비용량 = 설치용량*EA/1000

집광효율 = st.sidebar.number_input('집광효율 (%)', 0.00, 100.00, 15.02)
시스템효율 = st.sidebar.number_input('시스템 효율 (%)', 0.00, 100.00, 7.00)
인버터효율 = st.sidebar.number_input('인버터효율 (%)', 0.00, 100.00, 96.70)


#지역=a
st.subheader('a')
지역명 = ['강릉', '광주', '대관령', '대구', '대전', '목포','부산', '서산', '서울', '원주', '인천', '전주', '청주', '추풍령', '춘천', '포항', '흑산도']
지역 = st.sidebar.selectbox('지역', 지역명)
a = df1[지역]
st.dataframe(a)

#.맑은날 일수  = f
st.subheader('f')
st.dataframe(df3)
f = df3['일수']

#지역별 수평일사량 = bb
st.subheader('b')
b= [a[0] / f[0], a[1] / f[1], a[2] / f[2], a[3] / f[3], a[4] / f[4], a[5] / f[5], a[6] / f[6], a[7] / f[7], a[8] / f[8], a[9] / f[9], a[10] / f[10], a[11] / f[11]]
bb = pd.DataFrame(b, index=['01월', '02월', '03월', '04월', '05월', '06월', '07월', '08월', '09월', '10월', '11월', '12월'], columns=['수평일사량'])
round(bb['수평일사량'],3)
st.dataframe(bb)

#방위별 경사일사량 = cc
st.subheader('c')
방위별경사각 = ['South_15', 'South_30', 'South_45', 'South_60', 'South_75', 'South_90', 'East_90', 'West_90', 'North_90']
경사각도 = st.sidebar.selectbox('방위_경사', 방위별경사각)
c = df2[경사각도]
st.dataframe(c)

#경사일사량 = dd
st.subheader('d')
d = c[0] * b[0], c[0] * b[1], c[0] * b[2], c[0] * b[3], c[0] * b[4], c[0] * b[5], c[0] * b[6], c[0] * b[7], c[0] * b[8], c[0] * b[9], c[0] * b[10], c[0] * b[11]
dd = pd.DataFrame(d, index=['01월', '02월', '03월', '04월', '05월', '06월', '07월', '08월', '09월', '10월', '11월', '12월'], columns=['경사일사량'])
st.dataframe(dd)

#일일발전량 = ee
st.subheader('e')
e = [d[0] * 집광효율 * 집광면적 * 인버터효율 * 시스템효율/1000000, 
d[1] * 집광효율 * 집광면적 * 인버터효율 * 시스템효율/1000000, 
d[2] * 집광효율 * 집광면적 * 인버터효율 * 시스템효율/1000000, 
d[3] * 집광효율 * 집광면적 * 인버터효율 * 시스템효율/1000000, 
d[4] * 집광효율 * 집광면적 * 인버터효율 * 시스템효율/1000000, 
d[5] * 집광효율 * 집광면적 * 인버터효율 * 시스템효율/1000000, 
d[6] * 집광효율 * 집광면적 * 인버터효율 * 시스템효율/1000000, 
d[7] * 집광효율 * 집광면적 * 인버터효율 * 시스템효율/1000000, 
d[8] * 집광효율 * 집광면적 * 인버터효율 * 시스템효율/1000000, 
d[9] * 집광효율 * 집광면적 * 인버터효율 * 시스템효율/1000000, 
d[10] * 집광효율 * 집광면적 * 인버터효율 * 시스템효율/1000000, 
d[11] * 집광효율 * 집광면적 * 인버터효율 * 시스템효율/1000000,]
ee = pd.DataFrame(e, index=['01월', '02월', '03월', '04월', '05월', '06월', '07월', '08월', '09월', '10월', '11월', '12월'], columns=['일일발전량'])
st.dataframe(ee)


#월간발전량 = g
g = [e[0] * f[0], e[1] * f[1], e[2] * f[2], e[3] * f[3], e[4] * f[4], e[5] * f[5], e[6] * f[6], e[7] * f[7], e[8] * f[8], e[9] * f[9], e[10] * f[10], e[11] * f[11]]
gg = pd.DataFrame(g, index=['01월', '02월', '03월', '04월', '05월', '06월', '07월', '08월', '09월', '10월', '11월', '12월'], columns=['월간발전량'])
st.dataframe(gg)
st.line_chart(gg, use_container_width=True)





