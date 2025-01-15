import streamlit as st
from streamlit_option_menu import option_menu
import base64
from email import header
from html.entities import html5
from importlib.resources import read_binary
from markdown import markdown
from numpy.core.fromnumeric import var
import sys
from streamlit.web import cli as stcli
from PIL import Image
from functions import *
import streamlit.components.v1 as components
import pandas as pd
from st_clickable_images import clickable_images
import numpy as np
import statsmodels.api as sm
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import seaborn as sns
from io import BytesIO
from statsmodels.formula.api import ols
# from streamlit.state.session_state import SessionState
# import tkinter
import matplotlib
# matplotlib.use('TkAgg') testing
# matplotlib.use('Agg')
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.tree import DecisionTreeRegressor, plot_tree
import sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
import time
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import dtale
from dtale.views import startup
from dtale.app import get_instance
import webbrowser
import dtale.global_state as global_state
import dtale.app as dtale_app
from matplotlib.pyplot import axis, hist
from scipy import stats as stats
# from bioinfokit.analys import stat
from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm
from statsmodels.graphics.factorplots import interaction_plot
from sklearn.decomposition import PCA
from st_click_detector import click_detector
from streamlit.components.v1 import html
from click_image_dashboards import st_click_image_dashboards
from click_image_advanalytics import st_click_image_advanalytics
from dtale.views import startup
from streamlit_quill import st_quill
# import pandas_profiling
# from pandas_profiling import ProfileReport
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import dataingestion
from streamlit_card import card
import base64
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from streamlit_extras.metric_cards import style_metric_cards
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate
from scipy.stats import mode
# from tigerpreds import predictions
import streamlit_antd_components as sac
import pygwalker as pyg
from st_tabs import TabBar
from streamlit_extras.add_vertical_space import add_vertical_space
import streamlit_antd_components as sac
from st_click_detector import click_detector
import plotly.express as px
# from geopy.geocoders import Nominatim
# import folium
from time import sleep
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from tenacity import retry, stop_after_attempt, wait_fixed
import re

def indaloeda():

    init=1
    df_consolidated, df_combined, indalovator_data, indalogrow_data = dataingestion.readdata(init)

    value = st_click_image_dashboards()
    if value is None:
        # st.stop()
        pass

    st.success("{} selected".format(value))

    if value=='EDA':
        
        st.subheader("Exploratory data analysis")

        with st.spinner("Analyzing and summarizing dataset and generating dataset profile"):

            my_bar = st.progress(0)

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)

            start_time = time.time()

            edaenv = st.expander("Guidance on EDA", expanded=False)

            with edaenv:

                st.info("User guide")

                def show_pdf(file_path):
                    # Opening tutorial from file path
                    with open(file_path, "rb") as f:
                        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

                    # Embedding PDF in HTML
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="1500" height="800" type="application/pdf"></iframe>'

                    # Displaying File
                    st.markdown(pdf_display, unsafe_allow_html=True)

                col1, col2,col3= st.columns(3)
                with col1:  
                    if st.button('Read PDF tutorial',key='1'):            
                        show_pdf('Automated flow\DtaleInstructions-compressed.pdf')
                with col2:
                    st.button('Close PDF tutorial',key='2')                   
                with col3:
                    with open("Automated flow\DtaleInstructions-compressed.pdf", "rb") as pdf_file:
                        PDFbyte = pdf_file.read()
                    st.download_button(label="Download PDF tutorial", key='3',
                            data=PDFbyte,
                            file_name="EDA Instructions.pdf",
                            mime='application/octet-stream')

            datadescrip = st.expander("Description of data")

            with datadescrip:

                st.write(df_consolidated.describe(include='all'))
                
            edaprofiling = st.expander("Profile of dataset", expanded=False)
            
            with edaprofiling:
            
                # @st.cache(allow_output_mutation=True)
                # def gen_profile_report(df, *report_args, **report_kwargs):
                #     return df.profile_report(*report_args, **report_kwargs)

                # pr = gen_profile_report(df, explorative=True)

                # st_profile_report(pr)

                # df.drop(columns='BRANDNAME', axis=1)

                @st.cache(allow_output_mutation=True)
                def gen_profile_report(df_consolidated, *report_args, **report_kwargs):
                    return ProfileReport(df_consolidated, *report_args, **report_kwargs)

                # Assuming `df` is your DataFrame
                pr = gen_profile_report(
                    df_consolidated,
                    explorative=True,
                    title="Data profile",
                    dataset={
                        "description": "This profiling report shows an overview of the data",
                        "copyright_holder": "Heuristix",
                        "copyright_year": "2024",
                        "url": "https://www.ryanblumenow.com",
                    },
                    vars={"num": {"low_categorical_threshold": 0}},
                )

                st_profile_report(pr)

        startup(data_id="1", data=df_consolidated.sample(15000)) # All records, no OHE

        if get_instance("1") is None:
            startup(data_id="1", data=df_consolidated.sample(15000))

        d=get_instance("1")

        # webbrowser.open_new_tab('http://localhost:8501/dtale/main/1') # New window/tab
        # components.html("<iframe src='/dtale/main/1' />", width=1000, height=300, scrolling=True) # Element
        html = f"""<iframe src="/dtale/main/1" height="1000" width="1400"></iframe>""" # Iframe
        # html = "<a href='/dtale/main/1' target='_blank'>Dataframe 1</a>" # New tab link

        st.markdown(html, unsafe_allow_html=True)

        from mitosheet.streamlit.v1 import spreadsheet
        spreadsheet(df_consolidated)

        # d = dtale.show(pd.DataFrame(df2.sample(1000)))
        st.session_state.corr_img = d.get_corr_matrix()
        st.session_state.corr_df = d.get_corr_matrix(as_df=True)
        st.session_state.pps_img = d.get_pps_matrix()
        st.session_state.pps_df = d.get_pps_matrix(as_df=True)

        print(st.session_state.corr_df)

        checkbtn = st.button("Validate data")

        if checkbtn == True:
            df_amended = get_instance(data_id="1").data # The amended dataframe ready for upsert
            st.write("Sample of amended data:")
            st.write("")
            st.write(df_amended.head(5))

        clearchanges = st.button("Clear changes made to data")
        if clearchanges == True:
            global_state.cleanup()

        st.write("")
        
        st.subheader("Notes on EDA")

        # Spawn a new Quill editor
        st.subheader("Notes on exploratory data analysis")
        edacontent = st_quill(placeholder="Write your notes here", value=st.session_state.edanotes, key="edaquill")

        st.session_state.edanotes = edacontent

        st.write("Exploratory data analysis took ", time.time() - start_time, "seconds to run")