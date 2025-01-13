import streamlit as st
import hydralit as hy
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

#add an import to Hydralit
from hydralit import HydraHeadApp
from hydralit import HydraApp

#create a wrapper class
class indalovaluemap(HydraHeadApp):

#wrap all your code in this method and you should be done

    def run(self):

        from st_on_hover_tabs import on_hover_tabs

        st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)

        with st.sidebar:
                st.image('indalologo.jpg')
                add_vertical_space(1)
                value = on_hover_tabs(tabName=['KPI Overview', 'Dashboard/Analytics'], 
                                    iconName=['contacts', 'dashboard', 'account_tree', 'table', 'report', 'edit', 'update', 'pivot_table_chart', 'menu_book'],
                                    styles = {'navtab': {'background-color':'#6d0606',
                                                        'color': 'white',
                                                        'font-size': '18px',
                                                        'transition': '.3s',
                                                        'white-space': 'nowrap',
                                                        'text-transform': 'uppercase'},
                                            'tabOptionsStyle': {':hover :hover': {'color': 'red',
                                                                            'cursor': 'pointer'}},
                                            'iconStyle':{'position':'fixed',
                                                            'left':'7.5px',
                                                            'text-align': 'left'},
                                            'tabStyle' : {'list-style-type': 'none',
                                                            'margin-bottom': '30px',
                                                            'padding-left': '30px'}},
                                    key="hoversidebar",
                                    default_choice=0)

        css = '''
        <style>
            .stTabs [data-baseweb="tab-highlight"] {
                background-color:blue;
            }
        </style>
        '''

        st.markdown(css, unsafe_allow_html=True)

        # init=1
        # df_consolidated, df_combined, indalovator_data, indalogrow_data = dataingestion.readdata(init)

        import seaborn as sns
        import ppscore as pps
        import statsmodels.api as sm

        # Step 1: Handling NaN and infinite values
        # Replace infinite values with NaN
        df_filtered_by_cohort.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop columns with too many NaN values
        threshold = 0.5  # Drop columns with more than 50% missing values
        # df.dropna(thresh=int(threshold * len(df)), axis=1, inplace=True)

        # Fill remaining NaN values with the median of the column
        df_filtered_by_cohort.fillna(df_filtered_by_cohort.median(), inplace=True)

        # Step 2: Correlation Analysis for Sales and Profit
        correlation_matrix = df_filtered_by_cohort.corr()
        st.header("Correlation matrix")
        st.write(correlation_matrix)
        # # sales_correlation = correlation_matrix['Sales per customer'].drop('Sales per customer').sort_values(ascending=False)
        # # profit_correlation = correlation_matrix['Benefit per order'].drop('Benefit per order').sort_values(ascending=False)

        # # Step 3: PPS Analysis
        # pps_sales = pps.predictors(df, 'Sales per customer').sort_values(by='ppscore', ascending=False)
        # pps_profit = pps.predictors(df, 'Benefit per order').sort_values(by='ppscore', ascending=False)

        # # Step 4: Regression Analysis
        # # Selecting top correlated variables for regression
        # top_sales_predictors = sales_correlation.index[:5].tolist()
        # top_profit_predictors = profit_correlation.index[:5].tolist()

        # # Sales Regression
        # X_sales = df[top_sales_predictors]
        # y_sales = df['Sales per customer']
        # X_sales = sm.add_constant(X_sales)  # adding a constant
        # model_sales = sm.OLS(y_sales, X_sales).fit()

        # # Profit Regression
        # X_profit = df[top_profit_predictors]
        # y_profit = df['Benefit per order']
        # X_profit = sm.add_constant(X_profit)  # adding a constant
        # model_profit = sm.OLS(y_profit, X_profit).fit()

        # # Streamlit Code
        # st.title('Sales and Profit Analysis')

        # # Correlation Analysis
        # st.header('1. Correlation Analysis')
        # st.markdown("### Top Correlations for Sales and Profit")

        # top_sales_corr = sales_correlation.index[0]
        # top_profit_corr = profit_correlation.index[0]

        # # Visualizing Top Correlations
        # fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # sns.barplot(x=sales_correlation.head(5).values, y=sales_correlation.head(5).index, ax=axes[0], palette='Blues_d')
        # axes[0].set_title('Top 5 Correlations with Sales per Customer')
        # axes[0].set_xlabel('Correlation Coefficient')
        # axes[0].set_ylabel('')

        # sns.barplot(x=profit_correlation.head(5).values, y=profit_correlation.head(5).index, ax=axes[1], palette='Reds_d')
        # axes[1].set_title('Top 5 Correlations with Benefit per Order')
        # axes[1].set_xlabel('Correlation Coefficient')
        # axes[1].set_ylabel('')

        # st.pyplot(fig)

        # st.info(f"**Insight on Sales per Customer:** The variable most strongly correlated with Sales per Customer is `{top_sales_corr}` with a correlation coefficient of {sales_correlation[top_sales_corr]:.2f}. This suggests that changes in `{top_sales_corr}` are closely related to changes in sales, indicating a potential area to focus on for sales strategies.")

        # st.info(f"**Insight on Benefit per Order:** The variable most strongly correlated with Benefit per Order is `{top_profit_corr}` with a correlation coefficient of {profit_correlation[top_profit_corr]:.2f}. This indicates that `{top_profit_corr}` plays a significant role in determining profitability, and efforts to optimize this variable could enhance profit margins.")

        # # PPS Analysis
        # st.header('2. Predictive Power Score (PPS) Analysis')
        # st.markdown("### Top PPS Scores for Sales and Profit")

        # top_sales_pps = pps_sales.iloc[0]['x']
        # top_profit_pps = pps_profit.iloc[0]['x']

        # # Visualizing Top PPS Scores
        # fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # sns.barplot(x=pps_sales['ppscore'].head(5), y=pps_sales['x'].head(5), ax=axes[0], palette='Blues_d')
        # axes[0].set_title('Top 5 PPS Scores for Sales per Customer')
        # axes[0].set_xlabel('PPS Score')
        # axes[0].set_ylabel('')

        # sns.barplot(x=pps_profit['ppscore'].head(5), y=pps_profit['x'].head(5), ax=axes[1], palette='Reds_d')
        # axes[1].set_title('Top 5 PPS Scores for Benefit per Order')
        # axes[1].set_xlabel('PPS Score')
        # axes[1].set_ylabel('')

        # st.pyplot(fig)

        # st.info(f"**Predictive Insight for Sales:** The variable with the highest PPS score for predicting Sales per Customer is `{top_sales_pps}` with a score of {pps_sales.iloc[0]['ppscore']:.2f}. This means `{top_sales_pps}` is a strong predictor of sales outcomes, suggesting it should be a key focus in predictive models and strategies.")

        # st.info(f"**Predictive Insight for Profit:** The variable with the highest PPS score for predicting Benefit per Order is `{top_profit_pps}` with a score of {pps_profit.iloc[0]['ppscore']:.2f}. This highlights `{top_profit_pps}` as a crucial variable in predicting profitability and should be targeted for optimization efforts.")

        # # Regression Analysis
        # st.header('3. Regression Analysis')
        # st.markdown("### Significant Predictors from Regression Analysis")

        # sales_pvalues = model_sales.pvalues[1:]  # excluding the constant
        # significant_sales_vars = sales_pvalues[sales_pvalues < 0.05].index.tolist()

        # profit_pvalues = model_profit.pvalues[1:]  # excluding the constant
        # significant_profit_vars = profit_pvalues[profit_pvalues < 0.05].index.tolist()

        # if significant_sales_vars:
        #     st.subheader('Sales per Customer')
        #     st.markdown(f"The significant predictors of Sales per Customer in the regression model are: `{', '.join(significant_sales_vars)}`.")
        #     st.info(f"**Regression Insight for Sales:** The variables `{', '.join(significant_sales_vars)}` are statistically significant predictors of Sales per Customer with p-values less than 0.05. This suggests that these variables have a strong impact on sales and should be considered in sales strategies.")
        # else:
        #     st.subheader('Sales per Customer')
        #     st.markdown('There are no statistically significant predictors of Sales per Customer in the regression model.')
        #     st.info('**Regression Insight for Sales:** No variables were found to be statistically significant predictors of Sales per Customer. This suggests that other factors, not included in this analysis, may influence sales.')

        # if significant_profit_vars:
        #     st.subheader('Benefit per Order')
        #     st.markdown(f"The significant predictors of Benefit per Order in the regression model are: `{', '.join(significant_profit_vars)}`.")
        #     st.info(f"**Regression Insight for Profit:** The variables `{', '.join(significant_profit_vars)}` are statistically significant predictors of Benefit per Order with p-values less than 0.05. This indicates these variables have a strong influence on profitability and should be prioritized for improvement to enhance profits.")
        # else:
        #     st.subheader('Benefit per Order')
        #     st.markdown('There are no statistically significant predictors of Benefit per Order in the regression model.')
        #     st.info('**Regression Insight for Profit:** No variables were found to be statistically significant predictors of Benefit per Order. This implies that other factors, not included in this analysis, may affect profitability.')
