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
import requests
from geopy.geocoders import Nominatim
import folium
from time import sleep
from geopy.exc import GeocoderTimedOut
from tenacity import retry, stop_after_attempt, wait_fixed
import re

def indalodashboards():

    def add_vertical_space(num_lines: int = 1) -> None:
        """
        Add vertical space to your Streamlit app.

        Args:
            num_lines (int, optional): Height of the vertical space (given in number of lines). Defaults to 1.
        """
        for _ in range(num_lines):
            st.write("")  # This is just a way to do a line break!

    # st.set_page_config(layout="wide")

    menucol1, menucol2, menucol3 = st.columns(3)

    # with menucol1:

        # pagesel = option_menu(None, ["Dashboard", "Advanced analytics", "Make a prediction"],
        #                         icons=['None', 'None', 'None'],
        #                         menu_icon="app-indicator", default_index=0, orientation="horizontal",
        #                         styles={
        #         "container": {"padding": "5!important", "background-color": "#f3e5e5", "color": "#8B0103"},
        #         "icon": {"color": "black", "font-size": "25px"}, 
        #         "nav-link": {"font-size": "12px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        #         "nav-link-selected": {"background-color": "#8B0103", "color": "#f3e5e5", "font-weight": "normal"},
        #     }
        #     )

    if 'edanotes' not in st.session_state:
        st.session_state.edanotes = "Write your notes on EDA here"
    if 'dbnotes' not in st.session_state:
        st.session_state.dbnotes = "Write your notes on dashboard analysis here"

    init = 1
    df_consolidated, df_combined, indalovator_data, indalogrow_data = dataingestion.readdata(init)

    vars = list(df_consolidated.columns.values.tolist())

    st.session_state['pagechoice'] = 'dashboards'

    # if pagesel == "Dashboard":

        # value = sac.steps(
        # items=[
        #     sac.StepsItem(title='EDA', description='Explore'),
        #     sac.StepsItem(title='Overall', description='Entire dataset'),
        #     sac.StepsItem(title='Agency/Region', description='Grouped by area'),
        #     sac.StepsItem(title='Manager', description='Manager and below'),
        #     sac.StepsItem(title='Advisor', description='Individual data'),
        # ], format_func='title'
        # )

        # dbrdanalytics, dbrdmetrics, dbrdinteractive, dbrdneeds, dbrdpyg = st.tabs([":red[Analytics]", ":red[Metrics]", ":red[Interactive]", ":red[Needs assessment]", ":red[Pyg]"])

        # dbrdtype = TabBar(tabs=["Analytics","Tab2"],default=0,background = "white", color="red",activeColor="red",fontSize="14px")

        # import extra_streamlit_components as stx

        # chosen_id = stx.tab_bar(data=[
        #     stx.TabBarItemData(id=1, title="ToDo", description="Tasks to take care of"),
        #     stx.TabBarItemData(id=2, title="Done", description="Tasks taken care of"),
        #     stx.TabBarItemData(id=3, title="Overdue", description="Tasks missed out"),
        # ], default=1)

    df=df_consolidated

    colimg1, colimg2, colimg3 = st.columns([3.3,3,1])

    colimg2.image("indalologo.jpg", width=180, caption="")

    col1, col2, col3 = st.columns([3.1,3,1])

    col2.header("Indalo Analytics")
    
    value = sac.buttons([
        sac.ButtonsItem(label='Overall'),
        sac.ButtonsItem(label='Indalovator'),
        sac.ButtonsItem(label='Indalogrow'),
        sac.ButtonsItem(label='Indaloaccel'),
        # sac.ButtonsItem(icon='apple'),
        # sac.ButtonsItem(label='google', icon='google', color='#25C3B0'),
        # # sac.ButtonsItem(label='disabled', disabled=True),
        # sac.ButtonsItem(label='link', icon='share-fill', href='https://ant.design/components/button'),
    ], label='Shortcuts', align='left')

    if value == "Overall":
        df_filtered_by_cohort = df_consolidated
    elif value == "Indalovator":
        # df_filtered_by_cohort = indalovator_data
        df_filtered_by_cohort = df_consolidated[df_consolidated['Cohort'].fillna('') == 'Indalovator']
    elif value == "Indalogrow":
        # df_filtered_by_cohort = indalogrow_data
        df_filtered_by_cohort = df_consolidated[df_consolidated['Cohort'].fillna('') == 'Indalogrow']
    elif value == "Indaloaccel":
        # df_filtered_by_cohort = indalogrow_data
        df_filtered_by_cohort = df_consolidated[df_consolidated['Cohort'].fillna('') == 'Indaloaccel']

    selected_cohort = sac.cascader(items=[
        sac.CasItem('All', icon='house'),
        sac.CasItem('General', icon='house'),
        sac.CasItem('Core indicators', icon='person', children=[
            sac.CasItem('Economic', icon='person-circle'),
            sac.CasItem('Financial', icon='person'),
            sac.CasItem('Environmental', icon='twitter')]),
        sac.CasItem('Non-core indicators', icon='person', children=[
            sac.CasItem('Economic', icon='person-circle'),
            sac.CasItem('Environmental', icon='twitter'),
            sac.CasItem('Social', icon='person-circle'),]),
        sac.CasItem('Project-specific indicators', icon='person', children=[
            sac.CasItem('SiAGIA', icon='person-circle'),
            sac.CasItem('SIAWECCA', icon='twitter')]),
                ], label='Indicator types', index=0, multiple=True, search=True, clear=True)

    dbrddbrd, dbrdmetrics, dbrdanalytics, dbrdpyg, dbrdmap = st.tabs([":red[Dashboard]", ":red[Metrics]", ":red[Analytics]", ":red[Custom visualizations]", ":red[Map of influence]"]) # indalohome, ":red[Home]", 

    def generate_card_with_overlay(image_url, button1_text, button2_text, button3_text, card_text, expln_text, card_title):
        html_code = f'''
            <style>
                .card-container {{
                    display: flex;
                    border: 1px solid #fff; /* Change to white background */
                    width: 375px;
                    overflow: hidden;
                    flex-direction: column;
                    margin-bottom: 20px;
                    background-color: #f0f0f0; /* Grey background */
                    padding: 15px; /* Optional padding */
                    border-radius: 10px; /* Optional rounded corners */
                }}
                .card-left {{
                    padding: 10px;
                    cursor: pointer;
                    font-family: 'Roboto', sans-serif;
                    font-size: 12px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: flex-start;
                }}
                .card-left img {{
                    max-width: 100%;
                    max-height: 100%;
                }}
                .card-right {{
                    padding: 10px;
                    font-family: 'Roboto', sans-serif;
                    font-size: 14px;
                    overflow-y: hidden;
                }}
                .overlay {{
                    display: none;
                    position: fixed;
                    top: 1px;
                    left: 0;
                    width: 90%;
                    height: 90%;
                    background-color: transparent; /* Transparent background */
                    z-index: 1;
                    justify-content: center;
                    align-items: center;
                }}
                .overlay-image {{
                    max-width: 80%; /* Adjust the width of the overlay image */
                    max-height: 80%; /* Adjust the height of the overlay image */
                }}
                .card-title {{
                    text-align: center;
                    margin-top: 10px;
                    font-family: 'Roboto', sans-serif;
                    font-size: 14px;
                }}
                .button-container {{
                    display: flex;
                    justify-content: center;
                }}
                button {{
                    background-color: #207DCE;
                    color: white;
                    padding: 10px 15px;
                    border: none;
                    border-radius: 15px; /* Make buttons rounded */
                    cursor: pointer;
                    margin: 5px;
                }}
            </style>
            <div class="card-container" style="font-family: 'Roboto', sans-serif;">
                <h3 class="card-title">{card_title}</h3>
                <div class="card-left" onclick="openOverlay()">
                    <img src="{image_url}" style="width: 375px; height: 250px; margin-top: 25px;">
                </div>
                <div class="card-right">
                    <div class="button-container">
                        <button id="graph_expln_button">{button1_text}</button>
                        <button id="graph_expln_button2">{button2_text}</button>
                        <button id="reset_button">{button3_text}</button>
                    </div>
                    <p id="dynamic_heading"></p>
                    <p class="card-text" id="dynamic_text">{" "}</p>
                    <p class="card-text"><small class="text-muted">Last updated 3 mins ago</small></p>
                </div>
            </div>
            <div class="overlay" id="imageOverlay" onclick="closeOverlay()">
                <img src="{image_url}" class="overlay-image">
            </div>
            <script>
                function openOverlay() {{
                    document.getElementById("imageOverlay").style.display = "block";
                }}
                function closeOverlay() {{
                    document.getElementById("imageOverlay").style.display = "none";
                }}
                document.getElementById("graph_expln_button").addEventListener("click", function() {{
                    document.getElementById("dynamic_heading").style.fontWeight = "bold";
                    document.getElementById("dynamic_heading").innerText = "Analysis";
                    document.getElementById("dynamic_text").innerText = "{card_text}";
                    adjustCardHeight();
                }});
                document.getElementById("graph_expln_button2").addEventListener("click", function() {{
                    document.getElementById("dynamic_heading").style.fontWeight = "bold";
                    document.getElementById("dynamic_heading").innerText = "Recommendations";
                    document.getElementById("dynamic_text").innerText = "{expln_text}";
                    adjustCardHeight();
                }});
                document.getElementById("reset_button").addEventListener("click", function() {{
                    document.getElementById("dynamic_heading").innerText = "";
                    document.getElementById("dynamic_text").innerText = "";
                    adjustCardHeight();
                }});
                function adjustCardHeight() {{
                    var cardContainer = document.querySelector('.card-container');
                    cardContainer.style.height = 'auto';
                }}
            </script>
        '''
        return html_code
    
    def generate_card_with_overlay_interactive(image_url, button1_text, button2_text, button3_text, card_text, expln_text, card_title):        
        html_code_interactive = f'''
            <style>
                .card-container {{
                    display: flex;
                    border: 1px solid #fff; /* Change to white background */
                    width: 375px;
                    overflow: hidden;
                    flex-direction: column;
                    margin-bottom: 20px;
                    background-color: #f0f0f0; /* Grey background */
                    padding: 15px; /* Optional padding */
                    border-radius: 10px; /* Optional rounded corners */
                }}
                .card-left {{
                    padding: 10px;
                    cursor: pointer;
                    font-family: 'Roboto', sans-serif;
                    font-size: 12px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: flex-start;
                }}
                .card-left img {{
                    max-width: 100%;
                    max-height: 100%;
                }}
                .card-right {{
                    padding: 10px;
                    font-family: 'Roboto', sans-serif;
                    font-size: 14px;
                    overflow-y: hidden;
                }}
                .overlay {{
                    display: none;
                    position: fixed;
                    top: 1px;
                    left: 0;
                    width: 90%;
                    height: 90%;
                    background-color: transparent; /* Transparent background */
                    z-index: 1;
                    justify-content: center;
                    align-items: center;
                }}
                .overlay-image {{
                    max-width: 80%; /* Adjust the width of the overlay image */
                    max-height: 80%; /* Adjust the height of the overlay image */
                }}
                .card-title {{
                    text-align: center;
                    margin-top: 10px;
                    font-family: 'Roboto', sans-serif;
                    font-size: 14px;
                }}
                .button-container {{
                    display: flex;
                    justify-content: center;
                }}
                button {{
                    background-color: #207DCE;
                    color: white;
                    padding: 10px 15px;
                    border: none;
                    border-radius: 15px; /* Make buttons rounded */
                    cursor: pointer;
                    margin: 5px;
                }}
            </style>
            <div class="card-container" style="font-family: 'Roboto', sans-serif;">
                <h3 class="card-title">{card_title}</h3>
                <div class="card-left" onclick="openOverlay()">
                    {image_url}
                </div>
                <div class="card-right">
                    <div class="button-container">
                        <button id="graph_expln_button">{button1_text}</button>
                        <button id="graph_expln_button2">{button2_text}</button>
                        <button id="reset_button">{button3_text}</button>
                    </div>
                    <p id="dynamic_heading"></p>
                    <p class="card-text" id="dynamic_text">{" "}</p>
                    <p class="card-text"><small class="text-muted">Last updated 3 mins ago</small></p>
                </div>
            </div>
            <!--<div class="overlay" id="imageOverlay" onclick="closeOverlay()">
                {image_url}
            </div>-->
            <script>
                function openOverlay() {{
                    document.getElementById("imageOverlay").style.display = "block";
                }}
                function closeOverlay() {{
                    document.getElementById("imageOverlay").style.display = "none";
                }}
                document.getElementById("graph_expln_button").addEventListener("click", function() {{
                    document.getElementById("dynamic_heading").style.fontWeight = "bold";
                    document.getElementById("dynamic_heading").innerText = "Analysis";
                    document.getElementById("dynamic_text").innerText = "{card_text}";
                    adjustCardHeight();
                }});
                document.getElementById("graph_expln_button2").addEventListener("click", function() {{
                    document.getElementById("dynamic_heading").style.fontWeight = "bold";
                    document.getElementById("dynamic_heading").innerText = "Recommendations";
                    document.getElementById("dynamic_text").innerText = "{expln_text}";
                    adjustCardHeight();
                }});
                document.getElementById("reset_button").addEventListener("click", function() {{
                    document.getElementById("dynamic_heading").innerText = "";
                    document.getElementById("dynamic_text").innerText = "";
                    adjustCardHeight();
                }});
                function adjustCardHeight() {{
                    var cardContainer = document.querySelector('.card-container');
                    cardContainer.style.height = 'auto';
                }}
            </script>
        '''
        return html_code_interactive
    
    # Add JavaScript to make the row height dynamic
    st.markdown(
        """
        <script>
            const card1Wrapper = document.getElementById('card1_wrapper');

            function setDynamicHeight() {
                const cardHeight = card1Wrapper.scrollHeight;
                card1Wrapper.style.height = cardHeight + 'px';
            }

            // Call the function when the page loads
            setDynamicHeight();

            // Call the function whenever the content inside the card changes
            card1Wrapper.addEventListener('DOMSubtreeModified', setDynamicHeight);
        </script>
        """,
        unsafe_allow_html=True
    )

    def style_custom_metric_cards(
        background_color: str = "#E8E8E8",
        border_size_px: int = 1,
        border_color: str = "#CCC",
        border_radius_px: int = 5,
        border_left_color: str = "#6d0606",
        box_shadow: bool = True,
    ):
        box_shadow_str = (
            "box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;"
            if box_shadow
            else "box-shadow: none !important;"
        )
        st.markdown(
            f"""
            <style>
                div[data-testid="custom-metric-container"] {{
                    background-color: {background_color};
                    border: {border_size_px}px solid {border_color};
                    padding: 1.5% 5% 5% 10%;
                    margin-top: -12px;
                    border-radius: {border_radius_px}px;
                    border-left: 0.5rem solid {border_left_color} !important;
                    {box_shadow_str}
                }}
                div[data-testid="custom-metric-container"] .metric-value {{
                    font-size: 36px;
                }}
                div[data-testid="custom-metric-container"] .metric-label {{
                    font-size: 18px;
                }}
                div[data-testid="custom-metric-container"] .metric-delta {{
                    font-size: 14px;
                    text-align: left;
                    white-space: pre-wrap;
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Helper function to save plots as base64-encoded images
    def save_plot_to_base64(fig):
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        return f"data:image/png;base64,{image_base64}"

    # Function to generate HTML for metric cards
    # def create_metric_card(label, value, description):
    #     # text_align = "center" if centered else "left"
    #     return f"""
    #     <div data-testid="custom-metric-container">
    #         <div class="metric-label" style="font-weight: bold;">{label}</div>
    #         <div class="metric-value" style="font-weight: normal;">{value}</div>
    #         <div class="custom-metric-delta">{description}</div>
    #     </div>
    #     """

    def create_metric_card(label, value, description, centered=False):
        text_align = "center" if centered else "left"
        return f"""
        <div data-testid="custom-metric-container" style="
            display: flex; 
            flex-direction: column; 
            justify-content: center; 
            align-items: {text_align}; 
            padding: 16px; 
            margin: 8px;
            border: 1px solid #ddd; 
            border-radius: 8px; 
            background-color: #f9f9f9; 
            width: 100%; 
            height: auto;
            box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);">
            <div class="metric-label" style="
                font-weight: bold; 
                font-size: 16px; 
                margin-bottom: 8px; 
                text-align: {text_align};">{label}</div>
            <div class="metric-value" style="
                font-weight: bold; 
                font-size: 24px; 
                margin-bottom: 8px; 
                color: #333; 
                text-align: {text_align};">{value}</div>
            <div class="custom-metric-delta" style="
                font-weight: normal; 
                font-size: 14px; 
                color: #777; 
                text-align: {text_align};">{description}</div>
        </div>
        """
    
    # def create_metric_card(label, value, description, centered=False):
    #     # Set text alignment
    #     text_align = "center" if centered else "left"
    #     return f"""
    #     <div data-testid="custom-metric-container" style="
    #         display: flex; 
    #         flex-direction: column; 
    #         justify-content: center; 
    #         align-items: {text_align}; 
    #         padding: 16px; 
    #         margin: 8px;
    #         border: 1px solid #ddd; 
    #         border-radius: 8px; 
    #         background-color: #f9f9f9; 
    #         width: 100%; 
    #         height: auto;
    #         box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);">
    #         <div class="metric-label" style="
    #             font-weight: bold; 
    #             font-size: 16px; 
    #             margin-bottom: 8px; 
    #             text-align: {text_align};">{label}</div>
    #         <div class="metric-value" style="
    #             font-weight: bold; 
    #             font-size: 24px; 
    #             margin-bottom: 8px; 
    #             color: #333; 
    #             text-align: {text_align};">{value}</div>
    #         <div class="custom-metric-delta" style="
    #             font-weight: normal; 
    #             font-size: 14px; 
    #             color: #777; 
    #             text-align: {text_align};">{description}</div>
    #     </div>
    #     """

    # Function to generate base64-encoded bar chart
    def create_bar_chart(x, y, title, x_label, y_label):
        fig, ax = plt.subplots()
        ax.bar(x, y, color='skyblue')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        return save_plot_to_base64(fig)

    with dbrddbrd:

        style_custom_metric_cards()

        add_vertical_space(3)

        colm1, colm2 = st.columns([1,1])

        ### DASHBOARD

        ### NON-SPECIFIC INDICATORS

        # 1. Metric card: SROI headline
        sroi = 1.30
        with colm1:
            st.markdown(
                f"""
                <div data-testid="custom-metric-container">
                    <div class="metric-label" style="font-weight: bold;">{"Indalo's SiAGIA SROI"}</div>
                    <div class="metric-value" style="font-weight: normal;">{f"{sroi:,.2f}"}</div>
                    <div class="custom-metric-delta">{"Overall SROI (SiAGIA) for 2023, or 30%."}</div>
                    <div class="custom-metric-delta" style="color: #E8E8E8;">{"Add line"}</div>
                    <div class="custom-metric-delta" style="color: #E8E8E8;">{"Add line"}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # 2. Metric card: non-financial SROI outcomes
        with colm2:
            st.markdown(
                f"""
                <div data-testid="custom-metric-container">
                    <div class="metric-label" style="font-weight: bold;">{"Non-monetary impacts - top 5 reported"}</div>
                    <div class="custom-metric-delta">{"1. Diversity in workplace"}</div>
                    <div class="custom-metric-delta">{"2. Access to markets"}</div>
                    <div class="custom-metric-delta">{"3. Attraction of new capital"}</div>
                    <div class="custom-metric-delta">{"4. Diversification of products"}</div>
                    <div class="custom-metric-delta">{"5. New markets, skills, management, capital investment, new products, finance and operations"}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        
        target_values = ["All", "General"]

        if any(value in selected_cohort for value in target_values):

            add_vertical_space(3)

            ### GENERAL INDICATORS (Indalo organizational reporting)

            # st.subheader("Key Business Trends")
            st.subheader("General indicators")

            st.text("Internal reporting and key business trends")

            colgen1, colgen2, colgen3 = st.columns(3)

            # 1. Gender Distribution by Age Graph
            plt.figure(figsize=(6,4))
            df_filtered_by_cohort.groupby('Gender')['Age'].mean().plot(kind='bar', color=['#207DCE', '#3182A8'])
            plt.title('Gender Distribution by Age')
            plt.xlabel('Gender')
            plt.ylabel('Average Age')
            plt.xticks(rotation=0)

            # Save the plot to buffer and encode it as base64
            buffer3 = BytesIO()
            plt.savefig(buffer3, format='png', bbox_inches='tight')
            buffer3.seek(0)
            data3 = base64.b64encode(buffer3.read()).decode('utf-8')
            data3 = 'data:image/png;base64,' + data3

            # Generate the graph card
            image_url1 = data3
            card_title1 = "Gender Distribution by Age"
            button1_text1 = "Analysis"
            button2_text1 = "Recommendations"
            button3_text1 = "Clear"
            card_text1 = "Clients are generally younger and male with the ratio of females decreasing with age."
            expln_text1 = "We recommend you offer products with aggressive risk ratios."

            html_code1 = generate_card_with_overlay(
                image_url1, 
                button1_text1, 
                button2_text1, 
                button3_text1, 
                card_text1, 
                expln_text1, 
                card_title1
            )

            # Render the graph card
            with colgen1:
                st.components.v1.html(f'<div id="card1_Wrapper">{html_code1}</div>', width=1200, height=550)

            # 2. Revenue vs Profit Graph
            plt.figure(figsize=(6,4))
            df_filtered_by_cohort.plot(x='Revenue amount in the past financial year', y='Profit amount in the past financial year', kind='scatter')
            plt.title('Revenue vs Profit')
            plt.xlabel('Revenue (R)')
            plt.ylabel('Profit (R)')

            # Save the plot to buffer and encode it as base64
            buffer4 = BytesIO()
            plt.savefig(buffer4, format='png', bbox_inches='tight')
            buffer4.seek(0)
            data4 = base64.b64encode(buffer4.read()).decode('utf-8')
            data4 = 'data:image/png;base64,' + data4

            # Generate the graph card for Revenue vs Profit
            image_url2 = data4
            card_title2 = "Revenue vs Profit"
            button1_text2 = "Analysis"
            button2_text2 = "Recommendations"
            button3_text2 = "Clear"
            card_text2 = "There is a positive correlation between revenue and profit for most businesses."
            expln_text2 = "Focusing on high revenue streams may increase profitability."

            html_code2 = generate_card_with_overlay(
                image_url2, 
                button1_text2, 
                button2_text2, 
                button3_text2, 
                card_text2, 
                expln_text2, 
                card_title2
            )

            # Render the second graph card
            with colgen2:
                st.components.v1.html(f'<div id="card2_Wrapper">{html_code2}</div>', width=1200, height=550)

            # Community where the business is based
            community_counts = df_filtered_by_cohort['Type of area based in'].value_counts()
            plt.figure(figsize=(6,4))
            # df_combined.plot(x=community_counts.index, y=community_counts.values, kind='bar')
            plt.bar(community_counts.index, community_counts.values, color='skyblue')
            plt.title('Business Community Distribution')
            plt.xlabel('Type of area')
            plt.ylabel('Number of businesses')
            plt.xticks(rotation=45)  # Rotate x-axis labels if necessary
            plt.tight_layout()  # Adjust layout to prevent overlap

            # Save the plot to buffer and encode it as base64
            community_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_comm = community_chart
            card_title_comm = "Area of enterprise"
            button1_text_comm = "Analysis"
            button2_text_comm = "Recommendations"
            button3_text_comm = "Clear"
            card_text_comm = "xxx"
            expln_text_comm = "yyy"

            html_code_comm = generate_card_with_overlay(
                image_url_comm, 
                button1_text_comm, 
                button2_text_comm, 
                button3_text_comm, 
                card_text_comm, 
                expln_text_comm, 
                card_title_comm
            )
            
            # Render the graph card
            with colgen3:
                st.components.v1.html(f'<div id="card1_Wrapper">{html_code_comm}</div>', width=1200, height=550)

            # Province
            
            # Example data
            provdata = {
                "Name of community or township": [
                    "Howick Dargle", "Akasia", "Bende Mutale Village", "Kraaibosch", "Janefurse", "Groblersdal", "Springs",
                    "Mamelodi", "Vaal River City", "Bosplaas", "Bende Mutale Village", "Lillydale Farm", "Cradock", "Soweto",
                    "Cape Town", "Mabibi", "Sokhulu", "Dapha", "Sokhulu", "Sokhulu", "Mabibi", "Dapha", "Dapha", "Zibi",
                    "Mpukane", "Mabheleni Village", "Nkawulweni", "Sitiyweni", "Msukeni", "Magxeni", "Gwadane Village",
                    "Tyiweni", "Ntlola"
                ]
            }

            # Load data into a DataFrame
            df_prov = pd.DataFrame(provdata, columns=["Name of community or township"])

            # Define a mapping of communities to provinces
            community_to_province = {
                "Howick Dargle": "KwaZulu-Natal",
                "Akasia": "Gauteng",
                "Bende Mutale Village": "Limpopo",
                "Kraaibosch": "Western Cape",
                "Janefurse": "Limpopo",
                "Groblersdal": "Limpopo",
                "Springs": "Gauteng",
                "Mamelodi": "Gauteng",
                "Vaal River City": "Gauteng",
                "Bosplaas": "North West",
                "Lillydale Farm": "Mpumalanga",
                "Cradock": "Eastern Cape",
                "Soweto": "Gauteng",
                "Cape Town": "Western Cape",
                "Mabibi": "KwaZulu-Natal",
                "Sokhulu": "KwaZulu-Natal",
                "Dapha": "KwaZulu-Natal",
                "Zibi": "Eastern Cape",
                "Mpukane": "Eastern Cape",
                "Mabheleni Village": "Eastern Cape",
                "Nkawulweni": "Eastern Cape",
                "Sitiyweni": "Eastern Cape",
                "Msukeni": "Eastern Cape",
                "Magxeni": "Eastern Cape",
                "Gwadane Village": "Eastern Cape",
                "Tyiweni": "Eastern Cape",
                "Ntlola": "Eastern Cape"
            }

            # Initialize geolocator
            geolocator = Nominatim(user_agent="dynamic_province_mapper")
            cached_results = {}

            # Function to determine the province dynamically using geopy, with fallback to the mapping
            def get_province(community):
                if community in cached_results:
                    return cached_results[community]
                # Fallback to mapping if the community is in the predefined mapping
                if community in community_to_province:
                    return community_to_province[community]
                try:
                    # Attempt dynamic lookup via geopy
                    location = geolocator.geocode(community + ", South Africa", timeout=10)
                    time.sleep(1)  # Rate limit
                    if location:
                        for component in location.address.split(","):
                            if "Province" in component or component.strip() in [
                                "Eastern Cape", "Western Cape", "Northern Cape", "Gauteng", "KwaZulu-Natal",
                                "Free State", "Limpopo", "Mpumalanga", "North West"
                            ]:
                                cached_results[community] = component.strip()
                                return component.strip()
                    # If geopy fails, return Unknown
                    cached_results[community] = "Unknown"
                    return "Unknown"
                except GeocoderTimedOut:
                    cached_results[community] = "Unknown"
                    return "Unknown"

            # Apply the function to determine provinces dynamically
            df_prov['Province'] = df_prov['Name of community or township'].apply(get_province)

            # Count organizations by province
            province_counts = df_prov['Province'].value_counts()

            # Create a pie chart
            plt.figure(figsize=(8, 6))

            # Format each slice label as "number (percentage)"
            labels = [f"{count} ({percentage:.0f}%)" for count, percentage in zip(
                province_counts.values, 
                100 * province_counts.values / province_counts.values.sum()
            )]

            # Create the pie chart
            plt.pie(
                province_counts.values,
                labels=labels,
                autopct=None,  # Disable matplotlib's default percentage
                startangle=140,
                colors=plt.cm.tab20.colors
            )

            # Add a legend for provinces
            plt.legend(
                labels=province_counts.index,  # Provinces
                title="Province",
                loc="center left",
                bbox_to_anchor=(1, 0.5)
            )

            # Add the formatted labels to the slices
            # plt.title("Number of Organizations by Province")
            plt.axis('equal')  # Ensure the pie chart is a circle
            plt.tight_layout()

            # Save the plot as base64
            # Assuming `save_plot_to_base64` is already defined
            province_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_prov = province_chart
            card_title_prov = "Province of enterprise"
            button1_text_prov = "Analysis"
            button2_text_prov = "Recommendations"
            button3_text_prov = "Clear"
            card_text_prov = "This chart shows the distribution of organizations across provinces."
            expln_text_prov = "Understanding provincial distribution helps guide regional policies."

            html_code_prov = generate_card_with_overlay(
                image_url_prov, 
                button1_text_prov, 
                button2_text_prov, 
                button3_text_prov, 
                card_text_prov, 
                expln_text_prov, 
                card_title_prov
            )

            # Render the graph card in Streamlit
            with colgen1:
                st.components.v1.html(f'<div id="card1_Wrapper">{html_code_prov}</div>', width=1200, height=550)

            # # Degree of urbanisation (Urban, Peri-urban, Rural)
            # urbanisation_counts = df_combined['Degree of urbanisation of area of business'].value_counts()
            # urbanisation_chart = create_bar_chart(
            #     x=urbanisation_counts.index,
            #     y=urbanisation_counts.values,
            #     title='Degree of Urbanisation of Business Area',
            #     x_label='Area Type',
            #     y_label='Count'
            # )
            # results.append(f"<img src='{urbanisation_chart}' width='500'>")

            # --- Economic Indicators ---
            # Jobs created (full-time vs part-time)
            # full_time_jobs = df_filtered_by_cohort['No. of full-time employees employed by the enterprise'].sum()
            # part_time_jobs = df_filtered_by_cohort['No. of part time employees employed by the enterprise'].sum()
            # jobs_chart = create_bar_chart(
            #     x=['Full-Time', 'Part-Time'],
            #     y=[full_time_jobs, part_time_jobs],
            #     title='Jobs Created (Full-Time vs Part-Time)',
            #     x_label='Job Type',
            #     y_label='Number of Jobs'
            # )
            # results.append(f"<img src='{jobs_chart}' width='500'>")

            # # Temporary jobs created, Permanent jobs created (with gender/age breakdown where available)
            # temp_jobs = df_combined['No. of temporary jobs created (gender breakdown)'].sum() # Replace with correct column if available
            # perm_jobs = df_combined['No. of permanent jobs created'].sum()
            # results.append(create_metric_card("Temporary Jobs Created", temp_jobs, "Total temporary jobs created"))
            # results.append(create_metric_card("Permanent Jobs Created", perm_jobs, "Total permanent jobs created"))

            # # Salaries/Wages for Temporary and Permanent Jobs
            # temp_wages = df_combined['Total amount paid in salaries/wages for temporary jobs'].sum()
            # perm_wages = df_combined['Total amount paid in salaries/wages for permanent jobs'].sum()
            # results.append(create_metric_card("Salaries for Temporary Jobs", f"R {temp_wages:,.2f}", "Total wages for temporary jobs"))
            # results.append(create_metric_card("Salaries for Permanent Jobs", f"R {perm_wages:,.2f}", "Total wages for permanent jobs"))

            # # --- Financial Indicators ---
            # revenue = df_combined['Revenue amount in the past financial year'].sum()
            # profit = df_combined['Profit amount in the past financial year'].sum()
            # results.append(create_metric_card("Total Revenue", f"R {revenue:,.2f}", "Revenue generated in the last financial year"))
            # results.append(create_metric_card("Total Profit", f"R {profit:,.2f}", "Profit generated in the last financial year"))

            # # Additional financial metrics (Gross Profit, Taxes, etc.) could go here...

            # # --- Environmental Indicators ---
            # # Energy Savings, Water Savings, Waste Reduction
            # energy_saved = df_combined['Energy saved'].fillna(0).sum()
            # water_saved = df_combined['Water (litres) saved'].apply(pd.to_numeric, errors='coerce').fillna(0).sum()
            # waste_reduced = df_combined['Waste (tons) collected or recycled in the past 12 months '].apply(pd.to_numeric, errors='coerce').fillna(0).sum()
            
            # results.append(create_metric_card("Energy Saved", energy_saved, "Total energy saved across enterprises"))
            # results.append(create_metric_card("Water Saved", f"{water_saved:,.0f} liters", "Water saved in the past year"))
            # results.append(create_metric_card("Waste Reduced", f"{waste_reduced:,.2f} tons", "Total waste collected/recycled"))

            # # Display all metrics and graphs
            # for result in results:
            #     st.markdown(result, unsafe_allow_html=True)

            # Sector

            # Count organizations by sector
            sector_counts = df_filtered_by_cohort['Sector'].value_counts()

            # Create a pie chart
            plt.figure(figsize=(8, 6))

            # Format each slice label as "number (percentage)"
            labels = [f"{count} ({percentage:.0f}%)" for count, percentage in zip(
                sector_counts.values, 
                100 * sector_counts.values / sector_counts.values.sum()
            )]

            # Create the pie chart
            plt.pie(
                sector_counts.values,
                labels=labels,
                autopct=None,  # Disable matplotlib's default percentage
                startangle=140,
                colors=plt.cm.tab20.colors
            )

            # Add a legend for sectors
            plt.legend(
                labels=sector_counts.index,  # Sectors
                title="Sector",
                loc="center left",
                bbox_to_anchor=(1, 0.5)
            )

            # Add the formatted labels to the slices
            # plt.title("Number of Organizations by Sector")
            plt.axis('equal')  # Ensure the pie chart is a circle
            plt.tight_layout()

            # Save the plot as base64
            sector_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_sector = sector_chart
            card_title_sector = "Sector of enterprise"
            button1_text_sector = "Analysis"
            button2_text_sector = "Recommendations"
            button3_text_sector = "Clear"
            card_text_sector = "This chart shows the distribution of organizations across different sectors."
            expln_text_sector = "Understanding sector distribution helps identify industry trends and focus areas."

            html_code_sector = generate_card_with_overlay(
                image_url_sector, 
                button1_text_sector, 
                button2_text_sector, 
                button3_text_sector, 
                card_text_sector, 
                expln_text_sector, 
                card_title_sector
            )

            # Render the graph card in Streamlit
            with colgen2:
                st.components.v1.html(f'<div id="card2_Wrapper">{html_code_sector}</div>', width=1200, height=550)

            # CIPC registration

            # Preprocess the field to handle text-based data and missing responses
            df_filtered_by_cohort['Is your enterprise registered with the Companies and Intellectual Property Commission (CIPC)?'] = (
                df_filtered_by_cohort['Is your enterprise registered with the Companies and Intellectual Property Commission (CIPC)?']
                .astype(str)
                .str.strip()
            )

            # Replace NaN or empty strings with "Missing"
            df_filtered_by_cohort['Is your enterprise registered with the Companies and Intellectual Property Commission (CIPC)?'].replace(
                {"nan": "Missing", "": "Missing"}, inplace=True
            )

            # Count occurrences of each response, including missing responses
            cipc_counts = df_filtered_by_cohort['Is your enterprise registered with the Companies and Intellectual Property Commission (CIPC)?'].value_counts()

            # Create a pie chart
            plt.figure(figsize=(8, 6))

            # Format each slice label as "number (percentage)"
            labels = [f"{count} ({percentage:.0f}%)" for count, percentage in zip(
                cipc_counts.values, 
                100 * cipc_counts.values / cipc_counts.values.sum()
            )]

            # Create the pie chart
            plt.pie(
                cipc_counts.values,
                labels=labels,
                autopct=None,  # Disable matplotlib's default percentage
                startangle=140,
                colors=plt.cm.tab20.colors
            )

            # Add a legend for the responses
            plt.legend(
                labels=cipc_counts.index,  # Responses
                title="CIPC Registration",
                loc="center left",
                bbox_to_anchor=(1, 0.5)
            )

            # Add the formatted labels to the slices
            plt.axis('equal')  # Ensure the pie chart is a circle
            plt.tight_layout()

            # Save the plot as base64
            cipc_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_cipc = cipc_chart
            card_title_cipc = "CIPC Registration"
            button1_text_cipc = "Analysis"
            button2_text_cipc = "Recommendations"
            button3_text_cipc = "Clear"
            card_text_cipc = "This chart includes the percentage of enterprises registered with CIPC, including missing responses."
            expln_text_cipc = "Accounting for missing responses ensures a comprehensive analysis of enterprise registration status."

            html_code_cipc = generate_card_with_overlay(
                image_url_cipc, 
                button1_text_cipc, 
                button2_text_cipc, 
                button3_text_cipc, 
                card_text_cipc, 
                expln_text_cipc, 
                card_title_cipc
            )

            # Render the graph card in Streamlit
            with colgen3:
                st.components.v1.html(f'<div id="card3_Wrapper">{html_code_cipc}</div>', width=1200, height=550)

            # Type of enterprise

            # Preprocess the field to handle text-based data and missing responses
            df_filtered_by_cohort['Legal status of enterprise'] = (
                df_filtered_by_cohort['Legal status of enterprise']
                .astype(str)
                .str.strip()
            )

            # Replace NaN or empty strings with "Missing"
            df_filtered_by_cohort['Legal status of enterprise'].replace(
                {"nan": "Missing", "": "Missing"}, inplace=True
            )

            # Count occurrences of each response, including missing responses
            legal_status_counts = df_filtered_by_cohort['Legal status of enterprise'].value_counts()

            # Create a pie chart
            plt.figure(figsize=(8, 6))

            # Format each slice label as "number (percentage)"
            labels = [f"{count} ({percentage:.0f}%)" for count, percentage in zip(
                legal_status_counts.values, 
                100 * legal_status_counts.values / legal_status_counts.values.sum()
            )]

            # Create the pie chart
            plt.pie(
                legal_status_counts.values,
                labels=labels,
                autopct=None,  # Disable matplotlib's default percentage
                startangle=140,
                colors=plt.cm.tab20.colors
            )

            # Add a legend for the responses
            plt.legend(
                labels=legal_status_counts.index,  # Responses
                title="Legal Status",
                loc="center left",
                bbox_to_anchor=(1, 0.5)
            )

            # Add the formatted labels to the slices
            plt.axis('equal')  # Ensure the pie chart is a circle
            plt.tight_layout()

            # Save the plot as base64
            legal_status_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_legal = legal_status_chart
            card_title_legal = "Legal Status of Enterprise"
            button1_text_legal = "Analysis"
            button2_text_legal = "Recommendations"
            button3_text_legal = "Clear"
            card_text_legal = "This chart includes the distribution of enterprises by legal status, including missing responses."
            expln_text_legal = "Understanding legal status helps in determining compliance and business type distribution."

            html_code_legal = generate_card_with_overlay(
                image_url_legal, 
                button1_text_legal, 
                button2_text_legal, 
                button3_text_legal, 
                card_text_legal, 
                expln_text_legal, 
                card_title_legal
            )

            # Render the graph card in Streamlit
            with colgen1:
                st.components.v1.html(f'<div id="card4_Wrapper">{html_code_legal}</div>', width=1200, height=550)

            # Business stage

            # Preprocess the field to handle text-based data and missing responses
            df_filtered_by_cohort['Business stage'] = (
                df_filtered_by_cohort['Business stage']
                .astype(str)
                .str.strip()
            )

            # Replace NaN or empty strings with "Missing"
            df_filtered_by_cohort['Business stage'].replace(
                {"nan": "Missing", "": "Missing"}, inplace=True
            )

            # Count occurrences of each response, including missing responses
            business_stage_counts = df_filtered_by_cohort['Business stage'].value_counts()

            # Create a pie chart
            plt.figure(figsize=(8, 6))

            # Format each slice label as "number (percentage)"
            labels = [f"{count} ({percentage:.0f}%)" for count, percentage in zip(
                business_stage_counts.values, 
                100 * business_stage_counts.values / business_stage_counts.values.sum()
            )]

            # Create the pie chart
            plt.pie(
                business_stage_counts.values,
                labels=labels,
                autopct=None,  # Disable matplotlib's default percentage
                startangle=140,
                colors=plt.cm.tab20.colors
            )

            # Add a legend for the responses
            plt.legend(
                labels=business_stage_counts.index,  # Responses
                title="Business Stage",
                loc="center left",
                bbox_to_anchor=(1, 0.5)
            )

            # Add the formatted labels to the slices
            plt.axis('equal')  # Ensure the pie chart is a circle
            plt.tight_layout()

            # Save the plot as base64
            business_stage_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_business = business_stage_chart
            card_title_business = "Business Stage of Enterprises"
            button1_text_business = "Analysis"
            button2_text_business = "Recommendations"
            button3_text_business = "Clear"
            card_text_business = "This chart includes the distribution of enterprises by business stage, including missing responses."
            expln_text_business = "Understanding the business stage helps assess development levels and support needs."

            html_code_business = generate_card_with_overlay(
                image_url_business, 
                button1_text_business, 
                button2_text_business, 
                button3_text_business, 
                card_text_business, 
                expln_text_business, 
                card_title_business
            )

            # Render the graph card in Streamlit
            with colgen2:
                st.components.v1.html(f'<div id="card5_Wrapper">{html_code_business}</div>', width=1200, height=550)

            # Business plan

            # Type of Formal Business Plan

            # Preprocess the field to handle text-based data and missing responses
            df_filtered_by_cohort['Do you have a formal business plan'] = (
                df_filtered_by_cohort['Do you have a formal business plan']
                .astype(str)
                .str.strip()
            )

            # Replace NaN or empty strings with "Missing"
            df_filtered_by_cohort['Do you have a formal business plan'].replace(
                {"nan": "Missing", "": "Missing"}, inplace=True
            )

            # Count occurrences of each response, including missing responses
            business_plan_counts = df_filtered_by_cohort['Do you have a formal business plan'].value_counts()

            # Create a pie chart
            plt.figure(figsize=(8, 6))

            # Format each slice label as "number (percentage)"
            labels = [f"{count} ({percentage:.0f}%)" for count, percentage in zip(
                business_plan_counts.values, 
                100 * business_plan_counts.values / business_plan_counts.values.sum()
            )]

            # Create the pie chart
            plt.pie(
                business_plan_counts.values,
                labels=labels,
                autopct=None,  # Disable matplotlib's default percentage
                startangle=140,
                colors=plt.cm.tab20.colors
            )

            # Add a legend for the responses
            plt.legend(
                labels=business_plan_counts.index,  # Responses
                title="Formal Business Plan",
                loc="center left",
                bbox_to_anchor=(1, 0.5)
            )

            # Add the formatted labels to the slices
            plt.axis('equal')  # Ensure the pie chart is a circle
            plt.tight_layout()

            # Save the plot as base64
            business_plan_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_plan = business_plan_chart
            card_title_plan = "Formal Business Plan"
            button1_text_plan = "Analysis"
            button2_text_plan = "Recommendations"
            button3_text_plan = "Clear"
            card_text_plan = "This chart includes the distribution of enterprises indicating whether they have a formal business plan."
            expln_text_plan = "Understanding the prevalence of formal business plans helps gauge strategic planning among enterprises."

            html_code_plan = generate_card_with_overlay(
                image_url_plan, 
                button1_text_plan, 
                button2_text_plan, 
                button3_text_plan, 
                card_text_plan, 
                expln_text_plan, 
                card_title_plan
            )

            # Render the graph card in Streamlit
            with colgen3:
                st.components.v1.html(f'<div id="card6_Wrapper">{html_code_plan}</div>', width=1200, height=550)

            # Frequency of Business Plan Updates

            # Preprocess the field to handle text-based data and missing responses
            df_filtered_by_cohort['If yes, how often is your business plan updated or revised over a 12-month period?'] = (
                df_filtered_by_cohort['If yes, how often is your business plan updated or revised over a 12-month period?']
                .astype(str)
                .str.strip()
            )

            # Replace NaN or empty strings with "Missing"
            df_filtered_by_cohort['If yes, how often is your business plan updated or revised over a 12-month period?'].replace(
                {"nan": "Missing", "": "Missing"}, inplace=True
            )

            # Count occurrences of each response, including missing responses
            business_plan_update_counts = df_filtered_by_cohort['If yes, how often is your business plan updated or revised over a 12-month period?'].value_counts()

            # Create a pie chart
            plt.figure(figsize=(8, 6))

            # Format each slice label as "number (percentage)"
            labels = [f"{count} ({percentage:.0f}%)" for count, percentage in zip(
                business_plan_update_counts.values, 
                100 * business_plan_update_counts.values / business_plan_update_counts.values.sum()
            )]

            # Create the pie chart
            plt.pie(
                business_plan_update_counts.values,
                labels=labels,
                autopct=None,  # Disable matplotlib's default percentage
                startangle=140,
                colors=plt.cm.tab20.colors
            )

            # Add a legend for the responses
            plt.legend(
                labels=business_plan_update_counts.index,  # Responses
                title="Business Plan Updates",
                loc="center left",
                bbox_to_anchor=(1, 0.5)
            )

            # Add the formatted labels to the slices
            plt.axis('equal')  # Ensure the pie chart is a circle
            plt.tight_layout()

            # Save the plot as base64
            business_plan_update_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_update = business_plan_update_chart
            card_title_update = "Frequency of Business Plan Updates"
            button1_text_update = "Analysis"
            button2_text_update = "Recommendations"
            button3_text_update = "Clear"
            card_text_update = "This chart includes the distribution of responses for how often business plans are updated or revised over a 12-month period."
            expln_text_update = "Understanding the frequency of business plan updates helps assess planning discipline and responsiveness to change."

            html_code_update = generate_card_with_overlay(
                image_url_update, 
                button1_text_update, 
                button2_text_update, 
                button3_text_update, 
                card_text_update, 
                expln_text_update, 
                card_title_update
            )

            # Render the graph card in Streamlit
            with colgen1:
                st.components.v1.html(f'<div id="card7_Wrapper">{html_code_update}</div>', width=1200, height=550)
    
        target_values = ["All", "Core indicators", "Economic"]

        if any(value in selected_cohort for value in target_values):

            ### CORE INDICATORS

            add_vertical_space(3)

            st.subheader("Core indicators")

            st.text("Economic indicators")

            colecon1, colecon2, colecon3 = st.columns(3)

            plt.figure(figsize=(6, 4))

            # Create a bar plot for full-time vs part-time employees
            ax = df_filtered_by_cohort[['No. of full-time employees employed by the enterprise', 'No. of part time employees employed by the enterprise']].sum().plot(kind='bar')

            # Rename the x-axis labels to 'No. FTEs' and 'No. PTEs'
            ax.set_xticklabels(['No. FTEs', 'No. PTEs'], rotation=45)

            plt.title('Full Time vs Part Time Employees')
            plt.ylabel('Number of Employees')
            plt.xticks(rotation=45)

            # Save the plot to buffer and encode it as base64
            buffer5 = BytesIO()
            plt.savefig(buffer5, format='png', bbox_inches='tight')
            buffer5.seek(0)
            data5 = base64.b64encode(buffer5.read()).decode('utf-8')
            data5 = 'data:image/png;base64,' + data5

            # Generate the graph card for Full Time vs Part Time Employees
            image_url3 = data5
            card_title3 = "Full Time vs Part Time Employees"
            button1_text3 = "Analysis"
            button2_text3 = "Recommendations"
            button3_text3 = "Clear"
            card_text3 = "This chart shows the comparison between full-time and part-time employees."
            expln_text3 = "Analyzing workforce distribution helps to optimize resource allocation."

            html_code3 = generate_card_with_overlay(
                image_url3, 
                button1_text3, 
                button2_text3, 
                button3_text3, 
                card_text3, 
                expln_text3, 
                card_title3
            )

            # Render the third graph card
            with colecon1:
                st.components.v1.html(f'<div id="card3_Wrapper">{html_code3}</div>', width=1200, height=550)

            # Total pay for full-time/part-time employees
            # df_filtered_by_cohort['No. of full-time employees employed by the enterprise'] = df_filtered_by_cohort['No. of full-time employees employed by the enterprise'].fillna(0)
            # df_filtered_by_cohort['Average salary of the full-time employees'] = df_filtered_by_cohort['Average salary of the full-time employees'].fillna(0)

            df_filtered_by_cohort['Total Pay (Full-Time Employees)'] = (
                df_filtered_by_cohort['No. of full-time employees employed by the enterprise'] *
                df_filtered_by_cohort['Average salary of the full-time employees']
            )

            # df_filtered_by_cohort['No. of part time employees employed by the enterprise'] = df_filtered_by_cohort['No. of part time employees employed by the enterprise'].fillna(0)
            # df_filtered_by_cohort['Average salary of the part time employees'] = df_filtered_by_cohort['Average salary of the part time employees'].fillna(0)
            
            df_filtered_by_cohort['Total Pay (Part-Time Employees)'] = (
                df_filtered_by_cohort['No. of part time employees employed by the enterprise'] *
                df_filtered_by_cohort['Average salary of the part time employees']
            )

            df_filtered_by_cohort['Total Pay FTE/PTE'] = (
                df_filtered_by_cohort['Total Pay (Full-Time Employees)'] +
                df_filtered_by_cohort['Total Pay (Part-Time Employees)']
            )

            df_filtered_by_cohort['Total Pay (Owners)'] = (
                df_filtered_by_cohort['No. of owners or partners paid by the enterprise'] *
                df_filtered_by_cohort['Average salary of the owners/partners ']
            )

            plt.figure(figsize=(6, 4))

            # Create a bar plot for full-time vs part-time employees
            ax = df_filtered_by_cohort[['Total Pay (Full-Time Employees)', 'Total Pay (Part-Time Employees)']].sum().plot(kind='bar')

            # Rename the x-axis labels to 'No. FTEs' and 'No. PTEs'
            ax.set_xticklabels(['FTE pay', 'PTE pay'], rotation=45)

            plt.title('Full Time vs Part Time Employees Pay')
            plt.ylabel('Total pay')
            plt.xticks(rotation=45)

            # Save the plot to buffer and encode it as base64
            paychart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_pay = paychart
            card_title_pay = "Total Pay for Employees"
            button1_text_pay = "Analysis"
            button2_text_pay = "Recommendations"
            button3_text_pay = "Clear"
            card_text_pay = "Total pay for FTE/PTE employees"
            expln_text_pay = "This chart shows the total salary paid to full-time and part-time employees."

            html_code_pay = generate_card_with_overlay(
                image_url_pay, 
                button1_text_pay, 
                button2_text_pay, 
                button3_text_pay, 
                card_text_pay, 
                expln_text_pay, 
                card_title_pay
            )

            # Render the graph card
            with colecon2:
                st.components.v1.html(f'<div id="card1_Wrapper">{html_code_pay}</div>', width=1200, height=550)

            ### Dynamic graph - breakdown of employment by demographics

            # Assuming your dataframe is named `df_filtered_by_cohort`
            # Data preparation: Aggregate employee counts across relevant categories
            employee_data = pd.DataFrame({
                'Category': [
                    'Full-Time Employees',
                    'Part-Time Employees',
                    'Female Employees',
                    'Youth Employees (18-35)',
                    'Volunteers',
                    "Disabled"
                ],
                'Count': [
                    df_filtered_by_cohort["No. of full-time employees employed by the enterprise"].sum(),
                    df_filtered_by_cohort["No. of part time employees employed by the enterprise"].sum(),
                    df_filtered_by_cohort["No. of paid female employees employed in the enterprise"].sum(),
                    df_filtered_by_cohort["No. of paid youth (between 18-35) employees employed in the enterprise"].sum(),
                    df_filtered_by_cohort["No. of people working at the enterprise as volunteers"].sum(),
                    df_filtered_by_cohort["How many persons of disabilities does the enterprise employ (full-time, part-time or volunteers)"].sum()
                ]
            })

            employee_data['Category'] = employee_data['Category'].str.replace("Employees", "").str.strip()

            # Create an interactive bar chart with Plotly
            fig = px.bar(
                employee_data,
                x='Category',
                y='Count',
                # title='Employee Distribution Across Categories',
                labels={'Count': 'Number of Employees', 'Category': 'Employee Category'},
                text='Count',
                hover_data=['Count'],
            )

            # Adjust layout to move title and x-axis label
            fig.update_layout(
                # title=dict(
                #     text='Employee Distribution Across Categories',
                #     y=0.92,  # Move title down slightly
                #     x=0.5,
                #     font=dict(size=18)
                # ),
                xaxis=dict(
                    title=dict(
                        text=' ', # Remove axis label "Employee Category"
                        standoff=30  # Move x-axis label down
                    ),
                    tickangle=45,
                    tickfont=dict(size=10)
                ),
                yaxis_title="Number of Employees",
                margin=dict(l=121, r=40, t=80, b=60),  # Adjust margins
                width=375,  # Match the card's image dimensions
                height=275
            )

            # Add filtering directly on the graph
            fig.update_traces(marker_color='skyblue')
            fig.update_layout(updatemenus=[
                dict(
                    buttons=[
                        dict(
                            args=[{"y": [employee_data['Count']]}],
                            label="All Categories",
                            method="update"
                        ),
                        dict(
                            args=[{"y": [employee_data.loc[employee_data['Category'] == 'Full-Time Employees', 'Count']]}],
                            label="Full-Time Employees",
                            method="update"
                        ),
                        dict(
                            args=[{"y": [employee_data.loc[employee_data['Category'] == 'Part-Time Employees', 'Count']]}],
                            label="Part-Time Employees",
                            method="update"
                        ),
                        dict(
                            args=[{"y": [employee_data.loc[employee_data['Category'] == 'Female Employees', 'Count']]}],
                            label="Female Employees",
                            method="update"
                        ),
                        dict(
                            args=[{"y": [employee_data.loc[employee_data['Category'] == 'Youth Employees (18-35)', 'Count']]}],
                            label="Youth Employees (18-35)",
                            method="update"
                        ),
                        dict(
                            args=[{"y": [employee_data.loc[employee_data['Category'] == 'Volunteers', 'Count']]}],
                            label="Volunteers",
                            method="update"
                        ),
                        dict(
                            args=[{"y": [employee_data.loc[employee_data['Category'] == 'Disabled', 'Count']]}],
                            label="Disabled",
                            method="update"
                        )
                    ],
                    direction="down",
                    showactive=True,
                    x=1.3,  # Center dropdown below the graph
                    y=1.3   # Move dropdown below the graph
                )
            ])

            # Generate HTML for the Plotly chart
            plotly_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": True})

            # Define the card details
            card_title3 = "Employee Distribution"
            button1_text3 = "Analysis"
            button2_text3 = "Recommendations"
            button3_text3 = "Clear"
            card_text3 = "This interactive chart shows the employee distribution across categories, including permanent/temporary, gender, youth, volunteers, and people with disabilities."
            expln_text3 = "Dynamically analyze workforce demographics to make informed decisions."

            accesstomarket = generate_card_with_overlay_interactive(
                plotly_html,
                button1_text3,
                button2_text3,
                button3_text3,
                card_text3,
                expln_text3,
                card_title3

            )

            # Render the graph card in Streamlit
            with colecon1:
                st.components.v1.html(f'<div id="card2_Wrapper">{accesstomarket}</div>', width=1200, height=550)

            # # Render the graph card
            # with colecon2:
            #     st.components.v1.html(f'<div id="card3_Wrapper" style="padding: 20px;">{html_code3}</div>', width=400, height=600)

            # Dynamic pay graph

            # Assuming your dataframe is named `df_filtered_by_cohort`
            # Data preparation: Aggregate pay across relevant categories
            pay_data = pd.DataFrame({
                'Category': [
                    'Full-Time Employees',
                    'Part-Time Employees',
                    'Owners'
                ],
                'Total Pay': [
                    df_filtered_by_cohort["Total Pay (Full-Time Employees)"].sum(),
                    df_filtered_by_cohort["Total Pay (Part-Time Employees)"].sum(),
                    df_filtered_by_cohort['Total Pay (Owners)'].sum()
                ]
            })

            pay_data['Category'] = pay_data['Category'].str.replace("Employees", "").str.strip()

            # Create an interactive bar chart with Plotly
            fig = px.bar(
                pay_data,
                x='Category',
                y='Total Pay',
                labels={'Total Pay': 'Total Pay Amount', 'Category': 'Employee Category'},
                text='Total Pay',
                hover_data=['Total Pay']
            )

            # Adjust layout to move title and x-axis label
            fig.update_layout(
                xaxis=dict(
                    title=dict(
                        text=' ', # Remove axis label "Employee Category"
                        standoff=30  # Move x-axis label down
                    ),
                    tickangle=45,
                    tickfont=dict(size=10)
                ),
                yaxis_title="Total Pay Amount",
                margin=dict(l=121, r=40, t=80, b=60),  # Adjust margins
                width=375,  # Match the card's image dimensions
                height=275
            )

            # Add filtering directly on the graph
            fig.update_traces(marker_color='skyblue')
            fig.update_layout(updatemenus=[
                dict(
                    buttons=[
                        dict(
                            args=[{"y": [pay_data['Total Pay']]}],
                            label="All Categories",
                            method="update"
                        ),
                        dict(
                            args=[{"y": [pay_data.loc[pay_data['Category'] == 'Full-Time', 'Total Pay']]}],
                            label="Full-Time Employees",
                            method="update"
                        ),
                        dict(
                            args=[{"y": [pay_data.loc[pay_data['Category'] == 'Part-Time', 'Total Pay']]}],
                            label="Part-Time Employees",
                            method="update"
                        ),
                        dict(
                            args=[{"y": [pay_data.loc[pay_data['Category'] == 'Owners', 'Total Pay']]}],
                            label="Owners",
                            method="update"
                        )
                    ],
                    direction="down",
                    showactive=True,
                    x=1.3,  # Center dropdown below the graph
                    y=1.3   # Move dropdown below the graph
                )
            ])

            # Generate HTML for the Plotly chart
            plotly_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": True})

            # Define the card details
            card_title_pay = "Total Pay for Employees"
            button1_text_pay = "Analysis"
            button2_text_pay = "Recommendations"
            button3_text_pay = "Clear"
            card_text_pay = "This interactive chart shows the total salary paid to full-time and part-time employees."
            expln_text_pay = "Analyze pay distribution across employee types dynamically."

            interactive_pay_chart = generate_card_with_overlay_interactive(
                plotly_html,
                button1_text_pay,
                button2_text_pay,
                button3_text_pay,
                card_text_pay,
                expln_text_pay,
                card_title_pay
            )

            # Render the graph card in Streamlit
            with colecon2:
                st.components.v1.html(f'<div id="card1_Wrapper">{interactive_pay_chart}</div>', width=1200, height=550)

            # Potential to Maintain and Create Jobs

            # Preprocess the field to handle text-based data and missing responses
            df_filtered_by_cohort['Does your enterprise have the potential to maintain existing jobs and create new ones for your community?'] = (
                df_filtered_by_cohort['Does your enterprise have the potential to maintain existing jobs and create new ones for your community?']
                .astype(str)
                .str.strip()
            )

            # Replace NaN or empty strings with "Missing"
            df_filtered_by_cohort['Does your enterprise have the potential to maintain existing jobs and create new ones for your community?'].replace(
                {"nan": "Missing", "": "Missing"}, inplace=True
            )

            # Count occurrences of each response, including missing responses
            job_potential_counts = df_filtered_by_cohort['Does your enterprise have the potential to maintain existing jobs and create new ones for your community?'].value_counts()

            # Create a bar chart
            plt.figure(figsize=(10, 6))
            bars = plt.bar(
                job_potential_counts.index.astype(str),  # Categories (convert to string for consistent labels)
                job_potential_counts.values,  # Counts
                color=plt.cm.tab20.colors
            )

            # Add labels to each bar
            for bar in bars:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{bar.get_height()}",
                    ha='center',
                    va='bottom'
                )

            # Add titles and labels
            plt.title("Potential to Maintain and Create Jobs", fontsize=14)
            plt.xlabel("Response", fontsize=12)
            plt.ylabel("Number of Enterprises", fontsize=12)
            plt.xticks(
                ticks=range(len(job_potential_counts.index)),
                labels=job_potential_counts.index,
                rotation=45,
                ha="right"
            )
            plt.tight_layout()

            # Save the plot as base64
            job_potential_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_job_potential = job_potential_chart
            card_title_job_potential = "Job Maintenance and Creation Potential"
            button1_text_job_potential = "Analysis"
            button2_text_job_potential = "Recommendations"
            button3_text_job_potential = "Clear"
            card_text_job_potential = "This chart shows the distribution of responses regarding the potential of enterprises to maintain existing jobs and create new ones for their community."
            expln_text_job_potential = "Understanding job creation potential helps assess the socioeconomic impact of enterprises on their communities."

            html_code_job_potential = generate_card_with_overlay(
                image_url_job_potential, 
                button1_text_job_potential, 
                button2_text_job_potential, 
                button3_text_job_potential, 
                card_text_job_potential, 
                expln_text_job_potential, 
                card_title_job_potential
            )

            # Render the graph card in Streamlit
            with colecon3:
                st.components.v1.html(f'<div id="card12_Wrapper">{html_code_job_potential}</div>', width=1200, height=550)

            # Revenue increasing or decreasing

            # Preprocess the field to handle missing responses
            df_filtered_by_cohort['Compared to the previous financial year, has your revenue in the past 12 months increased, decreased or stayed roughly the same?'] = (
                df_filtered_by_cohort['Compared to the previous financial year, has your revenue in the past 12 months increased, decreased or stayed roughly the same?']
                .astype(str)
                .str.strip()
            )

            # Replace NaN or empty strings with "Missing"
            df_filtered_by_cohort['Compared to the previous financial year, has your revenue in the past 12 months increased, decreased or stayed roughly the same?'].replace(
                {"nan": "Missing", "": "Missing"}, inplace=True
            )

            # Count occurrences of each unique value, including missing responses
            revenue_change_counts = df_filtered_by_cohort[
                'Compared to the previous financial year, has your revenue in the past 12 months increased, decreased or stayed roughly the same?'
            ].value_counts()

            # Sort the counts by index (optional for consistent order)
            revenue_change_counts = revenue_change_counts.sort_index()

            # Create a bar chart
            plt.figure(figsize=(10, 6))
            bars = plt.bar(
                revenue_change_counts.index,  # Categories
                revenue_change_counts.values,  # Counts
                color=plt.cm.tab20.colors
            )

            # Add labels to each bar
            for bar in bars:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{bar.get_height()}",
                    ha='center',
                    va='bottom'
                )

            # Add titles and labels
            plt.title("Revenue Change Compared to Previous Financial Year", fontsize=14)
            plt.xlabel("Revenue Change Category", fontsize=12)
            plt.ylabel("Number of Enterprises", fontsize=12)
            plt.xticks(rotation=45, ha="right")  # Rotate category labels for readability
            plt.tight_layout()

            # Save the plot as base64
            revenue_change_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_revenue = revenue_change_chart
            card_title_revenue = "Enterprises' Reported Revenue Changes"
            button1_text_revenue = "Analysis"
            button2_text_revenue = "Recommendations"
            button3_text_revenue = "Clear"
            card_text_revenue = "This chart shows the distribution of enterprises based on revenue changes over the past year, including missing values."
            expln_text_revenue = "Understanding revenue trends helps assess enterprise performance and challenges over time."

            html_code_revenue_change = generate_card_with_overlay(
                image_url_revenue, 
                button1_text_revenue, 
                button2_text_revenue, 
                button3_text_revenue, 
                card_text_revenue, 
                expln_text_revenue, 
                card_title_revenue
            )

            # Render the graph card in Streamlit
            with colecon1:
                st.components.v1.html(f'<div id="card6_Wrapper">{html_code_revenue_change}</div>', width=1200, height=550)

            # Sales Comparison to Previous Year

            # Preprocess the field to handle text-based data and missing responses
            df_filtered_by_cohort['Compared to the previous 12 months, has the sale of your products or services increased, decreased or stayed roughly the same?'] = (
                df_filtered_by_cohort['Compared to the previous 12 months, has the sale of your products or services increased, decreased or stayed roughly the same?']
                .astype(str)
                .str.strip()
            )

            # Replace NaN or empty strings with "Missing"
            df_filtered_by_cohort['Compared to the previous 12 months, has the sale of your products or services increased, decreased or stayed roughly the same?'].replace(
                {"nan": "Missing", "": "Missing"}, inplace=True
            )

            # Count occurrences of each response, including missing responses
            sales_comparison_counts = df_filtered_by_cohort['Compared to the previous 12 months, has the sale of your products or services increased, decreased or stayed roughly the same?'].value_counts()

            # Create a pie chart
            plt.figure(figsize=(8, 6))

            # Format each slice label as "number (percentage)"
            labels = [f"{count} ({percentage:.0f}%)" for count, percentage in zip(
                sales_comparison_counts.values, 
                100 * sales_comparison_counts.values / sales_comparison_counts.values.sum()
            )]

            # Create the pie chart
            plt.pie(
                sales_comparison_counts.values,
                labels=labels,
                autopct=None,  # Disable matplotlib's default percentage
                startangle=140,
                colors=plt.cm.tab20.colors
            )

            # Add a legend for the responses
            plt.legend(
                labels=sales_comparison_counts.index,  # Responses
                title="Sales Comparison",
                loc="center left",
                bbox_to_anchor=(1, 0.5)
            )

            # Add the formatted labels to the slices
            plt.axis('equal')  # Ensure the pie chart is a circle
            plt.tight_layout()

            # Save the plot as base64
            sales_comparison_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_sales = sales_comparison_chart
            card_title_sales = "Sales Comparison to Previous Year"
            button1_text_sales = "Analysis"
            button2_text_sales = "Recommendations"
            button3_text_sales = "Clear"
            card_text_sales = "This chart includes the distribution of responses comparing sales to the previous 12 months."
            expln_text_sales = "Understanding sales trends helps assess business performance and identify growth opportunities or challenges."

            html_code_sales = generate_card_with_overlay(
                image_url_sales, 
                button1_text_sales, 
                button2_text_sales, 
                button3_text_sales, 
                card_text_sales, 
                expln_text_sales, 
                card_title_sales
            )

            # Render the graph card in Streamlit
            with colecon2:
                st.components.v1.html(f'<div id="card8_Wrapper">{html_code_sales}</div>', width=1200, height=550)

            # Revenue Expectations for Next Year

            # Preprocess the field to handle text-based data and missing responses
            df_filtered_by_cohort['In the next 12 months, do you expect your revenue to increase, decrease or stay roughly the same?'] = (
                df_filtered_by_cohort['In the next 12 months, do you expect your revenue to increase, decrease or stay roughly the same?']
                .astype(str)
                .str.strip()
            )

            # Replace NaN or empty strings with "Missing"
            df_filtered_by_cohort['In the next 12 months, do you expect your revenue to increase, decrease or stay roughly the same?'].replace(
                {"nan": "Missing", "": "Missing"}, inplace=True
            )

            # Count occurrences of each response, including missing responses
            revenue_expectation_counts = df_filtered_by_cohort['In the next 12 months, do you expect your revenue to increase, decrease or stay roughly the same?'].value_counts()

            # Create a pie chart
            plt.figure(figsize=(8, 6))

            # Format each slice label as "number (percentage)"
            labels = [f"{count} ({percentage:.0f}%)" for count, percentage in zip(
                revenue_expectation_counts.values, 
                100 * revenue_expectation_counts.values / revenue_expectation_counts.values.sum()
            )]

            # Create the pie chart
            plt.pie(
                revenue_expectation_counts.values,
                labels=labels,
                autopct=None,  # Disable matplotlib's default percentage
                startangle=140,
                colors=plt.cm.tab20.colors
            )

            # Add a legend for the responses
            plt.legend(
                labels=revenue_expectation_counts.index,  # Responses
                title="Revenue Expectations",
                loc="center left",
                bbox_to_anchor=(1, 0.5)
            )

            # Add the formatted labels to the slices
            plt.axis('equal')  # Ensure the pie chart is a circle
            plt.tight_layout()

            # Save the plot as base64
            revenue_expectation_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_revenue = revenue_expectation_chart
            card_title_revenue = "Revenue Expectations for Next Year"
            button1_text_revenue = "Analysis"
            button2_text_revenue = "Recommendations"
            button3_text_revenue = "Clear"
            card_text_revenue = "This chart includes the distribution of responses regarding revenue expectations for the next 12 months."
            expln_text_revenue = "Understanding revenue expectations helps forecast growth trends and identify optimism or challenges in the market."

            html_code_revenue = generate_card_with_overlay(
                image_url_revenue, 
                button1_text_revenue, 
                button2_text_revenue, 
                button3_text_revenue, 
                card_text_revenue, 
                expln_text_revenue, 
                card_title_revenue
            )

            # Render the graph card in Streamlit
            with colecon3:
                st.components.v1.html(f'<div id="card9_Wrapper">{html_code_revenue}</div>', width=1200, height=550)

            # Revenue Plans for Next Year (Monthly)

            # Preprocess the field to handle text-based data and missing responses
            df_filtered_by_cohort['How much do you plan to make in the next 12 months? (monthly)'] = (
                df_filtered_by_cohort['How much do you plan to make in the next 12 months? (monthly)']
                .astype(str)
                .str.strip()
            )

            # Replace NaN or empty strings with "Missing"
            df_filtered_by_cohort['How much do you plan to make in the next 12 months? (monthly)'].replace(
                {"nan": "Missing", "": "Missing"}, inplace=True
            )

            # DEBUG: Print raw data
            print("Raw Data:")
            print(df_filtered_by_cohort['How much do you plan to make in the next 12 months? (monthly)'])

            # Convert numeric values where possible, keeping "Missing" as None
            def process_revenue(value):
                try:
                    return float(value)  # Handle numeric values
                except ValueError:
                    return None  # Non-numeric values are treated as None

            df_filtered_by_cohort['Processed Revenue (Monthly)'] = df_filtered_by_cohort[
                'How much do you plan to make in the next 12 months? (monthly)'
            ].apply(process_revenue)

            # DEBUG: Check numeric conversion
            print("\nProcessed Revenue (Numeric):")
            print(df_filtered_by_cohort['Processed Revenue (Monthly)'])

            # Categorize revenue into predefined ranges
            def categorize_revenue(value):
                if value is None or value == "" or value == "Missing":
                    return "Missing"
                elif value <= 1000:
                    return "Up to R1,000"
                elif value <= 5000:
                    return "R1,001 - R5,000"
                elif value <= 10000:
                    return "R5,001 - R10,000"
                elif value <= 20000:
                    return "R10,001 - R20,000"
                elif value > 20000:
                    return "More than R20,000"
                else:
                    return "Missing"

            df_filtered_by_cohort['Revenue Category'] = df_filtered_by_cohort[
                'Processed Revenue (Monthly)'
            ].apply(categorize_revenue)

            # DEBUG: Check categorization
            print("\nCategorized Data:")
            print(df_filtered_by_cohort[['Processed Revenue (Monthly)', 'Revenue Category']])

            # Count occurrences of each revenue category
            revenue_counts = df_filtered_by_cohort['Revenue Category'].value_counts()

            # Order categories on the x-axis
            category_order = ["Up to R1,000", "R1,001 - R5,000", "R5,001 - R10,000", "R10,001 - R20,000", "More than R20,000", "Missing"]
            revenue_counts = revenue_counts.reindex(category_order).fillna(0)  # Reorder and fill missing categories with 0

            # DEBUG: Print counts
            print("\nRevenue Counts:")
            print(revenue_counts)

            # Create a bar chart
            plt.figure(figsize=(10, 6))
            bars = plt.bar(
                revenue_counts.index,  # Ordered categories
                revenue_counts.values,  # Counts
                color=plt.cm.tab20.colors
            )

            # Add labels to each bar
            for bar in bars:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{int(bar.get_height())}",
                    ha='center',
                    va='bottom'
                )

            # Add titles and labels
            plt.title("Planned Monthly Revenue for the Next 12 Months", fontsize=14)
            plt.xlabel("Revenue Range (Monthly)", fontsize=12)
            plt.ylabel("Number of Enterprises", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            # Save the plot as base64
            revenue_bar_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_revenue = revenue_bar_chart
            card_title_revenue = "Planned Monthly Revenue for Next Year"
            button1_text_revenue = "Analysis"
            button2_text_revenue = "Recommendations"
            button3_text_revenue = "Clear"
            card_text_revenue = "This chart shows the distribution of planned monthly revenue ranges for enterprises in the next 12 months."
            expln_text_revenue = "Analyzing planned revenue ranges helps assess financial expectations and business growth prospects."

            html_code_revenue = generate_card_with_overlay(
                image_url_revenue, 
                button1_text_revenue, 
                button2_text_revenue, 
                button3_text_revenue, 
                card_text_revenue, 
                expln_text_revenue, 
                card_title_revenue
            )

            # Render the graph card in Streamlit
            with colecon1:
                st.components.v1.html(f'<div id="card10_Wrapper">{html_code_revenue}</div>', width=1200, height=550)

            add_vertical_space(3)

            st.text("Diversity")

            coldiv1, coldiv2, coldiv3 = st.columns(3)

            # Group by 'Type of area based in' and sum the total pay
            total_pay_by_area = df_filtered_by_cohort.groupby('Type of area based in')['Total Pay FTE/PTE'].sum()

            # Create a pie chart
            plt.figure(figsize=(6, 4))
            plt.pie(
            total_pay_by_area.values, 
            labels=None,  # Remove labels from the pie chart
            autopct='%1.1f%%', 
            startangle=140, 
            colors=plt.cm.tab20.colors
            )

            # Add a legend for the areas
            plt.legend(
                total_pay_by_area.index, 
                title="Type of Area", 
                loc="center left", 
                bbox_to_anchor=(1, 0.5)  # Position the legend to the right of the pie chart
            )
            plt.title('Total Pay by Type of Area')
            plt.axis('equal')  # Ensure the pie chart is a circle
            plt.tight_layout()

            # Save the plot to a buffer as base64
            pie_chart_base64 = save_plot_to_base64(plt)

            # Generate the graph card
            areapay = generate_card_with_overlay(
                pie_chart_base64,
                "Analysis",
                "Recommendations",
                "Clear",
                "Distribution of total pay by area",
                "This pie chart shows the proportion of total pay distributed by the type of area where the businesses are located.",
                "Total Pay by Type of Area"
            )

            # Render the graph card in Streamlit
            with coldiv1:
                st.components.v1.html(f'<div id="card2_Wrapper">{areapay}</div>', width=1200, height=550)

            ### Skills, training, development

            st.text("Skills, training, development")

            colskills1, colskills2, colskills3 = st.columns(3)

            # Count the number of enterprises in each skills category
            skills_counts = df_filtered_by_cohort['Do you (or team members) have formal skills and experience in the field of your business?'].value_counts()

            # Calculate the percentage of enterprises providing a response
            total_responses = skills_counts.sum()
            total_enterprises = len(df_filtered_by_cohort)
            response_percentage = (total_responses / total_enterprises) * 100

            # Create the pie chart
            plt.figure(figsize=(8, 6))
            plt.pie(
                skills_counts.values,
                labels=None,  # Use legend instead of direct labels
                autopct='%1.1f%%',
                startangle=140,
                colors=plt.cm.tab20.colors
            )
            plt.legend(
                skills_counts.index,
                title="Skills and Experience",
                loc="center left",
                bbox_to_anchor=(1, 0.5)
            )
            plt.title("Skills and Training Distribution")
            plt.axis('equal')  # Ensure the pie chart is a circle

            # Add a textbox on the bottom right
            text_box_content = f"Response Rate: {response_percentage:.1f}%"
            plt.text(
                1, -0.2, text_box_content, fontsize=18, color="black",
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"),
                horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes
            )

            plt.tight_layout()

            # Save the pie chart as base64
            skills_chart = save_plot_to_base64(plt)

            # Generate the graph card
            skills_training_card = generate_card_with_overlay(
                skills_chart,
                "Analysis",
                "Recommendations",
                "Clear",
                "Skills and Training",
                "This pie chart shows the proportion of enterprises reporting formal skills and experience in the field of their business, categorized as 'Yes,' 'No,' or 'In the process.'",
                "Skills and Training"
            )

            # Render the graph card in Streamlit
            with colskills1:
                st.components.v1.html(f'<div id="skillsTrainingCard_Wrapper">{skills_training_card}</div>', width=1200, height=550)

            # Number of operational years

            # Map categorical values to numerical equivalents
            replacement_map = {
                "4 - 6 years": 5,
                "Not yet started": 0,
                "1 - 3 years": 2
            }

            # Replace values in the 'No. of operational years' column
            df_filtered_by_cohort['No. of operational years'] = (
                df_filtered_by_cohort['No. of operational years']
                .replace(replacement_map)  # Replace specific text values
                .apply(pd.to_numeric, errors='coerce')  # Convert all values to numeric, set invalid to NaN
                .round()  # Round all numeric values
            )

            # Replace NaN values with "Missing"
            df_filtered_by_cohort['No. of operational years'] = df_filtered_by_cohort['No. of operational years'].fillna("Missing")

            # Count occurrences of each unique value, including "Missing"
            operational_years_counts = df_filtered_by_cohort['No. of operational years'].value_counts()

            # Separate "Missing" from the numeric values for sorting
            missing_count = operational_years_counts.pop("Missing") if "Missing" in operational_years_counts else 0
            numeric_counts = operational_years_counts.sort_index()  # Sort numeric values
            operational_years_counts = pd.concat([numeric_counts, pd.Series({"Missing": missing_count})])  # Add "Missing" back

            # Create a bar chart
            plt.figure(figsize=(10, 6))
            bars = plt.bar(
                operational_years_counts.index.astype(str),  # Convert indices to strings for proper labeling
                operational_years_counts.values,
                color=plt.cm.tab20.colors
            )

            # # Add labels to each bar
            # for bar in bars:
            #     plt.text(
            #         bar.get_x() + bar.get_width() / 2,
            #         bar.get_height() + 1,
            #         f"{bar.get_height()}",
            #         ha='center',
            #         va='bottom'
            #     )

            # Add titles and labels
            # plt.title("Number of Operational Years of Enterprises", fontsize=14)
            plt.xlabel("Operational Years", fontsize=12)
            plt.ylabel("Number of Enterprises", fontsize=12)
            plt.tight_layout()

            # Save the plot as base64
            operational_years_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_operational = operational_years_chart
            card_title_operational = "Operational Years of Enterprise"
            button1_text_operational = "Analysis"
            button2_text_operational = "Recommendations"
            button3_text_operational = "Clear"
            card_text_operational = "This chart shows the distribution of enterprises by number of operational years, including missing values."
            expln_text_operational = "Analyzing operational years provides insights into enterprise maturity and trends."

            html_code_operational = generate_card_with_overlay(
                image_url_operational, 
                button1_text_operational, 
                button2_text_operational, 
                button3_text_operational, 
                card_text_operational, 
                expln_text_operational, 
                card_title_operational
            )

            # Render the graph card in Streamlit
            with colskills2:
                st.components.v1.html(f'<div id="card5_Wrapper">{html_code_operational}</div>', width=1200, height=550)

            # Planned Employment in the Next 12 Months

            # Preprocess the column to handle text-based data and missing responses
            df_filtered_by_cohort['How many people do you plan to provide employment to within the next 12 months?'] = (
                df_filtered_by_cohort['How many people do you plan to provide employment to within the next 12 months?']
                .astype(str)
                .str.strip()
            )

            # Define a function to categorize data (ranges and numeric values)
            def categorize_and_convert(value):
                if value == "1-2":
                    return "1-2", 1.5  # Midpoint of 1-2
                elif value == "2-3":
                    return "2-3", 2.5  # Midpoint of 2-3
                elif value == "3-4":
                    return "3-4", 3.5  # Midpoint of 3-4
                elif value == "More than 4":
                    return "More than 4", 5.0  # Conservative estimate for "More than 4"
                elif value.isdigit():  # Handle raw numeric values
                    num = int(value)
                    if 1 <= num <= 2:
                        return "1-2", 1.5
                    elif 2 < num <= 3:
                        return "2-3", 2.5
                    elif 3 < num <= 4:
                        return "3-4", 3.5
                    elif num > 4:
                        return "More than 4", 5.0
                return "Missing", pd.NA  # Handle unexpected or missing values

            # Apply the function to categorize and calculate midpoints
            categorized_data = df_filtered_by_cohort[
                'How many people do you plan to provide employment to within the next 12 months?'
            ].apply(categorize_and_convert)

            # Split into two columns: categories and numeric midpoints
            df_filtered_by_cohort['Employment Category'], df_filtered_by_cohort['Planned Employment (Numeric)'] = zip(*categorized_data)

            # Count occurrences of each category
            employment_counts = df_filtered_by_cohort['Employment Category'].value_counts()

            # Order categories on the x-axis
            category_order = ["1-2", "2-3", "3-4", "More than 4", "Missing"]
            employment_counts = employment_counts.reindex(category_order).fillna(0)  # Reorder and fill missing categories with 0

            # Calculate total and average planned employment using numeric midpoints
            total_jobs = df_filtered_by_cohort['Planned Employment (Numeric)'].sum(skipna=True)
            average_jobs = df_filtered_by_cohort['Planned Employment (Numeric)'].mean(skipna=True)

            # Create a bar chart
            plt.figure(figsize=(10, 6))
            bars = plt.bar(
                employment_counts.index,  # Ordered categories
                employment_counts.values,  # Counts
                color=plt.cm.tab20.colors
            )

            # Add labels to each bar
            for bar in bars:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{int(bar.get_height())}",
                    ha='center',
                    va='bottom'
                )

            # Add titles and labels
            plt.title("Planned Employment in the Next 12 Months", fontsize=14)
            plt.xlabel("Employment Range", fontsize=12)
            plt.ylabel("Number of Enterprises", fontsize=12)
            plt.xticks(rotation=45, ha="right")

            # Add a text box at the bottom for total and average jobs
            plt.text(
                0.5, -0.2, 
                f"Total Planned Jobs: {int(total_jobs)}\nAverage Planned Jobs: {average_jobs:.2f}", 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', edgecolor='black')
            )

            plt.tight_layout()

            # Save the plot as base64
            employment_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_employment = employment_chart
            card_title_employment = "Planned Employment in the Next 12 Months"
            button1_text_employment = "Analysis"
            button2_text_employment = "Recommendations"
            button3_text_employment = "Clear"
            card_text_employment = "This chart shows the distribution of planned employment ranges by enterprises over the next 12 months."
            expln_text_employment = "Understanding planned employment levels helps assess potential job growth and economic impact in the community."

            html_code_employment = generate_card_with_overlay(
                image_url_employment, 
                button1_text_employment, 
                button2_text_employment, 
                button3_text_employment, 
                card_text_employment, 
                expln_text_employment, 
                card_title_employment
            )

            # Render the graph card in Streamlit
            with colskills3:
                st.components.v1.html(f'<div id="card14_Wrapper">{html_code_employment}</div>', width=1200, height=550)

            # Plan to Add Products/Services

            # Preprocess the field to handle text-based data and missing responses
            df_filtered_by_cohort['Do you plan to add products/services to your current offering?'] = (
                df_filtered_by_cohort['Do you plan to add products/services to your current offering?']
                .astype(str)
                .str.strip()
            )

            # Replace NaN or empty strings with "Missing"
            df_filtered_by_cohort['Do you plan to add products/services to your current offering?'].replace(
                {"nan": "Missing", "": "Missing"}, inplace=True
            )

            # Categorize responses (Yes/No/Maybe/Missing)
            def categorize_plan(value):
                if value.lower() in ["yes", "y", "true"]:
                    return "Yes"
                elif value.lower() in ["no", "n", "false"]:
                    return "No"
                elif value.lower() in ["maybe", "unsure", "not sure"]:
                    return "Maybe"
                else:
                    return "Missing"

            df_filtered_by_cohort['Plan Category'] = df_filtered_by_cohort[
                'Do you plan to add products/services to your current offering?'
            ].apply(categorize_plan)

            # Count occurrences of each category
            plan_counts = df_filtered_by_cohort['Plan Category'].value_counts()

            # Order categories on the x-axis
            category_order = ["Yes", "No", "Maybe", "Missing"]
            plan_counts = plan_counts.reindex(category_order).fillna(0)  # Reorder and fill missing categories with 0

            # DEBUG: Print counts
            print("\nPlan Counts:")
            print(plan_counts)

            # Create a bar chart
            plt.figure(figsize=(10, 6))
            bars = plt.bar(
                plan_counts.index,  # Ordered categories
                plan_counts.values,  # Counts
                color=plt.cm.tab20.colors
            )

            # Add labels to each bar
            for bar in bars:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{int(bar.get_height())}",
                    ha='center',
                    va='bottom'
                )

            # Add titles and labels
            plt.title("Plans to Add Products/Services", fontsize=14)
            plt.xlabel("Response", fontsize=12)
            plt.ylabel("Number of Businesses", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            # Save the plot as base64
            plan_bar_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_plan = plan_bar_chart
            card_title_plan = "Plans to Add Products/Services"
            button1_text_plan = "Analysis"
            button2_text_plan = "Recommendations"
            button3_text_plan = "Clear"
            card_text_plan = "This chart shows the distribution of responses for whether businesses plan to add products/services to their current offering."
            expln_text_plan = "Understanding plans to expand offerings helps assess growth strategies and market adaptability."

            html_code_plan = generate_card_with_overlay(
                image_url_plan, 
                button1_text_plan, 
                button2_text_plan, 
                button3_text_plan, 
                card_text_plan, 
                expln_text_plan, 
                card_title_plan
            )

            # Render the graph card in Streamlit
            with colskills1:
                st.components.v1.html(f'<div id="card13_Wrapper">{html_code_plan}</div>', width=1200, height=550)

            # Preprocess the column for enterprise plans
            df_filtered_by_cohort['Does your enterprise plan to do any of the following over the next two to three years?'] = (
                df_filtered_by_cohort['Does your enterprise plan to do any of the following over the next two to three years?']
                .astype(str)
                .str.strip()
            )

            # Replace NaN or empty strings with "Missing"
            df_filtered_by_cohort['Does your enterprise plan to do any of the following over the next two to three years?'].replace(
                {"nan": "Missing", "": "Missing"}, inplace=True
            )

            # Split comma-separated values into individual categories
            plans_exploded = df_filtered_by_cohort[
                'Does your enterprise plan to do any of the following over the next two to three years?'
            ].str.split(',').explode()

            # Clean up whitespace and count occurrences of each category
            plans_exploded = plans_exploded.str.strip()
            plan_counts = plans_exploded.value_counts()

            # DEBUG: Print counts
            print("\nEnterprise Plans Counts:")
            print(plan_counts)

            # Create a pie chart
            plt.figure(figsize=(10, 8))
            plt.pie(
                plan_counts.values,
                labels=plan_counts.index,
                autopct='%1.1f%%',  # Show percentages
                startangle=140,
                colors=plt.cm.tab20.colors[:len(plan_counts)]
            )

            # Add a title
            plt.title("Enterprise Plans Over the Next 2-3 Years", fontsize=16)
            plt.tight_layout()

            # Save the plot as base64
            enterprise_plans_pie_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_enterprise_plans = enterprise_plans_pie_chart
            card_title_enterprise_plans = "Enterprise Plans Over the Next 2-3 Years"
            button1_text_enterprise_plans = "Analysis"
            button2_text_enterprise_plans = "Recommendations"
            button3_text_enterprise_plans = "Clear"
            card_text_enterprise_plans = "This chart shows the distribution of enterprise plans for the next two to three years."
            expln_text_enterprise_plans = "Understanding enterprise plans helps assess growth strategies and key focus areas."

            html_code_enterprise_plans = generate_card_with_overlay(
                image_url_enterprise_plans, 
                button1_text_enterprise_plans, 
                button2_text_enterprise_plans, 
                button3_text_enterprise_plans, 
                card_text_enterprise_plans, 
                expln_text_enterprise_plans, 
                card_title_enterprise_plans
            )

            # Render the graph card in Streamlit
            with colskills2:
                st.components.v1.html(f'<div id="card20_Wrapper">{html_code_enterprise_plans}</div>', width=1200, height=650)

            # Count the number of enterprises in each "Access to market" category
            access_to_market_counts = df_filtered_by_cohort['Access to market'].value_counts()

            # Calculate the percentage of enterprises providing a response
            total_responses_access_to_market = access_to_market_counts.sum()
            total_enterprises_access_to_market = len(df_filtered_by_cohort)
            response_percentage_access_to_market = (total_responses_access_to_market / total_enterprises_access_to_market) * 100

            # Create the pie chart
            plt.figure(figsize=(8, 6))
            plt.pie(
                access_to_market_counts.values,
                labels=None,  # Use legend instead of direct labels
                autopct='%1.1f%%',
                startangle=140,
                colors=plt.cm.tab20.colors
            )
            plt.legend(
                access_to_market_counts.index,
                title="Access to Market",
                loc="center left",
                bbox_to_anchor=(1, 0.5)
            )
            plt.title("Access to Market Distribution")
            plt.axis('equal')  # Ensure the pie chart is a circle

            # Add a textbox on the bottom right
            text_box_content_access_to_market = f"Response Rate: {response_percentage_access_to_market:.1f}%"
            plt.text(
                1, -0.2, text_box_content_access_to_market, fontsize=18, color="black",
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"),
                horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes
            )

            plt.tight_layout()

            # Save the pie chart as base64
            access_to_market_chart = save_plot_to_base64(plt)

            # Generate the graph card
            accesstomarket = generate_card_with_overlay(
                access_to_market_chart,
                "Analysis",
                "Recommendations",
                "Clear",
                "Access to Market",
                "This pie chart shows the proportion of enterprises reporting enhanced access to market for their products or services.",
                "Access to Market"
            )

            # Render the graph card in Streamlit
            with colskills3:
                st.components.v1.html(f'<div id="accessToMarketCard_Wrapper">{accesstomarket}</div>', width=1200, height=550)

            # Count the number of enterprises in each "Anticipating access to market" category
            future_access_to_market_counts = df_filtered_by_cohort['Are you anticipating accessing new markets in the next six months?'].value_counts()

            # Calculate the percentage of enterprises providing a response
            future_total_responses_access_to_market = future_access_to_market_counts.sum()
            future_total_enterprises_access_to_market = len(df_filtered_by_cohort)
            future_response_percentage_access_to_market = (future_total_responses_access_to_market / future_total_enterprises_access_to_market) * 100

            # Create the pie chart
            plt.figure(figsize=(8, 6))
            plt.pie(
                future_access_to_market_counts.values,
                labels=None,  # Use legend instead of direct labels
                autopct='%1.1f%%',
                startangle=140,
                colors=plt.cm.tab20.colors
            )
            plt.legend(
                future_access_to_market_counts.index,
                title="Changing market access",
                loc="center left",
                bbox_to_anchor=(1, 0.5)
            )
            plt.title("Anticipating changing access to market")
            plt.axis('equal')  # Ensure the pie chart is a circle

            # Add a textbox on the bottom right
            text_box_future_content_access_to_market = f"Response Rate: {future_response_percentage_access_to_market:.1f}%"
            plt.text(
                1, -0.2, text_box_future_content_access_to_market, fontsize=18, color="black",
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"),
                horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes
            )

            plt.tight_layout()

            # Save the pie chart as base64
            future_access_to_market_chart = save_plot_to_base64(plt)

            # Generate the graph card
            futureaccesstomarket = generate_card_with_overlay(
                future_access_to_market_chart,
                "Analysis",
                "Recommendations",
                "Clear",
                "Anticipating Accessing New Markets (6 mo)",
                "This pie chart shows the proportion of enterprises reporting anticipating enhanced access to market for their products or services.",
                "Anticipating Accessing New Markets (6 mo)"
            )

            # Render the graph card in Streamlit
            with colskills1:
                st.components.v1.html(f'<div id="accessToMarketCard_Wrapper">{futureaccesstomarket}</div>', width=1200, height=550)

        target_values = ["All", "Core indicators", "Environmental"]

        if any(value in selected_cohort for value in target_values):

            add_vertical_space(3)

            st.text("Environmental indicators")

            colenv1, colenv2, colenv3 = st.columns(3)

            # Energy saved

            # Preprocess the 'Energy saved' column to handle missing responses and string values
            df_filtered_by_cohort['Energy saved'] = df_filtered_by_cohort['Energy saved'].apply(
                lambda x: x if isinstance(x, (int, float)) else "Missing"  # Convert non-numeric values to "Missing"
            )

            # Count occurrences of each unique value, including missing responses
            energy_saved_counts = df_filtered_by_cohort['Energy saved'].value_counts()

            # Sort the counts by index (optional for consistent order, numeric first, then "Missing")
            if "Missing" in energy_saved_counts:
                missing_count = energy_saved_counts.pop("Missing")
                energy_saved_counts = energy_saved_counts.sort_index()  # Sort numeric values
                # energy_saved_counts = energy_saved_counts.append(pd.Series({"Missing": missing_count}))  # Add "Missing" back
                energy_saved_counts = pd.concat([energy_saved_counts, pd.Series({"Missing": missing_count})]) # Add "Missing" back

            else:
                energy_saved_counts = energy_saved_counts.sort_index()

            # Create a bar chart
            plt.figure(figsize=(10, 6))
            bars = plt.bar(
                energy_saved_counts.index.astype(str),  # Categories (convert to string for consistent labels)
                energy_saved_counts.values,  # Counts
                color=plt.cm.tab20.colors
            )

            # Add labels to each bar
            for bar in bars:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{bar.get_height()}",
                    ha='center',
                    va='bottom'
                )

            # Add titles and labels
            plt.title("Energy Saved Distribution", fontsize=14)
            plt.xlabel("Energy Saved (or Missing)", fontsize=12)
            plt.ylabel("Number of Enterprises", fontsize=12)
            plt.xticks(rotation=45, ha="right")  # Rotate category labels for readability
            plt.tight_layout()

            # Save the plot as base64
            energy_saved_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_energy = energy_saved_chart
            card_title_energy = "Energy Saved"
            button1_text_energy = "Analysis"
            button2_text_energy = "Recommendations"
            button3_text_energy = "Clear"
            card_text_energy = "This chart shows the distribution of energy saved by enterprises, with missing values categorized appropriately. Certain enterprises did not confirm to data handling criteria and their responses have been lost."
            expln_text_energy = "Analyzing energy savings helps assess sustainability efforts and identify areas for improvement."

            html_code_energy = generate_card_with_overlay(
                image_url_energy, 
                button1_text_energy, 
                button2_text_energy, 
                button3_text_energy, 
                card_text_energy, 
                expln_text_energy, 
                card_title_energy
            )

            # Render the graph card in Streamlit
            with colenv1:
                st.components.v1.html(f'<div id="card7_Wrapper">{html_code_energy}</div>', width=1200, height=550)

            # Water saved

            # Preprocess the 'Water (litres) saved' column to extract numeric values from strings
            def extract_numeric(value):
                if isinstance(value, (int, float)):
                    return value  # Keep numeric values as-is
                elif isinstance(value, str):
                    match = re.search(r"\d+", value)  # Extract numeric content from the string
                    return float(match.group()) if match else "Missing"  # Convert to float if numeric content exists
                else:
                    return "Missing"  # Non-numeric and non-string values are treated as "Missing"

            df_filtered_by_cohort['Water (litres) saved'] = df_filtered_by_cohort['Water (litres) saved'].apply(extract_numeric)

            # Replace any "Missing" labels for consistency
            df_filtered_by_cohort['Water (litres) saved'] = df_filtered_by_cohort['Water (litres) saved'].replace("Missing", pd.NA)

            # Count occurrences of each unique value, including missing responses
            water_saved_counts = round(df_filtered_by_cohort['Water (litres) saved'].value_counts(dropna=False))

            # Sort the counts for numeric values first, then "Missing"
            if pd.NA in water_saved_counts:
                missing_count = water_saved_counts.pop(pd.NA)
                water_saved_counts = water_saved_counts.sort_index()  # Sort numeric values
                # water_saved_counts = water_saved_counts.append(pd.Series({"Missing": missing_count}))  # Add "Missing" back
                water_saved_counts = pd.concat([water_saved_counts, pd.Series({"Missing": missing_count})]) # Add "Missing" back

            else:
                water_saved_counts = water_saved_counts.sort_index()

            # Create a bar chart
            plt.figure(figsize=(10, 6))
            bars = plt.bar(
                water_saved_counts.index.astype(str),  # Categories (convert to string for consistent labels)
                water_saved_counts.values,  # Counts
                color=plt.cm.tab20.colors
            )

            # # Add labels to each bar
            # for bar in bars:
            #     plt.text(
            #         bar.get_x() + bar.get_width() / 2,
            #         bar.get_height() + 1,
            #         f"{bar.get_height()}",
            #         ha='center',
            #         va='bottom'
            #     )

            # Add titles and labels
            plt.title("Water Saved Distribution", fontsize=14)
            plt.xlabel("Water Saved (Litres or Missing)", fontsize=12)
            plt.ylabel("Number of Enterprises", fontsize=12)
            # plt.xticks(rotation=45, ha="right")  # Rotate category labels for readability
            plt.xticks(ticks=range(len(water_saved_counts.index)), labels=[str(int(x)) if isinstance(x, (int, float)) and not pd.isna(x) else str(x) for x in water_saved_counts.index], rotation=45, ha="right")
            plt.tight_layout()

            # Save the plot as base64
            water_saved_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_water = water_saved_chart
            card_title_water = "Water Saved"
            button1_text_water = "Analysis"
            button2_text_water = "Recommendations"
            button3_text_water = "Clear"
            card_text_water = "This chart shows the distribution of water saved (in litres) by enterprises, including missing values."
            expln_text_water = "Analyzing water savings helps assess sustainability efforts and resource management."

            html_code_water = generate_card_with_overlay(
                image_url_water, 
                button1_text_water, 
                button2_text_water, 
                button3_text_water, 
                card_text_water, 
                expln_text_water, 
                card_title_water
            )

            # Render the graph card in Streamlit
            with colenv2:
                st.components.v1.html(f'<div id="card8_Wrapper">{html_code_water}</div>', width=1200, height=550)

            # Water used

            # Preprocess the 'Water used per month' column to extract numeric values from strings
            def extract_numeric(value):
                if isinstance(value, (int, float)):
                    return value  # Keep numeric values as-is
                elif isinstance(value, str):
                    match = re.search(r"\d+", value)  # Extract numeric content from the string
                    return float(match.group()) if match else "Missing"  # Convert to float if numeric content exists
                else:
                    return "Missing"  # Non-numeric and non-string values are treated as "Missing"

            df_filtered_by_cohort['Water used per month'] = df_filtered_by_cohort['Water used per month'].apply(extract_numeric)

            # Replace any "Missing" labels for consistency
            df_filtered_by_cohort['Water used per month'] = df_filtered_by_cohort['Water used per month'].replace("Missing", pd.NA)

            # Count occurrences of each unique value, including missing responses
            water_used_counts = df_filtered_by_cohort['Water used per month'].value_counts(dropna=False)

            # Sort the counts for numeric values first, then "Missing"
            if pd.NA in water_used_counts:
                missing_count = water_used_counts.pop(pd.NA)
                water_used_counts = water_used_counts.sort_index()  # Sort numeric values
                # water_used_counts = water_used_counts.append(pd.Series({"Missing": missing_count}))  # Add "Missing" back
                water_used_counts = pd.concat([water_used_counts, pd.Series({"Missing": missing_count})]) # Add "Missing" back

            else:
                water_used_counts = water_used_counts.sort_index()

            # Create a bar chart
            plt.figure(figsize=(10, 6))
            bars = plt.bar(
                water_used_counts.index.astype(str),  # Categories (convert to string for consistent labels)
                water_used_counts.values,  # Counts
                color=plt.cm.tab20.colors
            )

            # # Add labels to each bar
            # for bar in bars:
            #     plt.text(
            #         bar.get_x() + bar.get_width() / 2,
            #         bar.get_height() + 1,
            #         f"{bar.get_height()}",
            #         ha='center',
            #         va='bottom'
            #     )

            # Add titles and labels
            plt.title("Water Used Per Month Distribution", fontsize=14)
            plt.xlabel("Water Used Per Month (Litres or Missing)", fontsize=12)
            plt.ylabel("Number of Enterprises", fontsize=12)
            plt.xticks(
                ticks=range(len(water_used_counts.index)),
                labels=[str(int(x)) if isinstance(x, (int, float)) and not pd.isna(x) else str(x) for x in water_used_counts.index],
                rotation=45,
                ha="right"
            )
            plt.tight_layout()

            # Save the plot as base64
            water_used_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_water_used = water_used_chart
            card_title_water_used = "Water Used Per Month"
            button1_text_water_used = "Analysis"
            button2_text_water_used = "Recommendations"
            button3_text_water_used = "Clear"
            card_text_water_used = "This chart shows the distribution of water used per month by enterprises, including missing values."
            expln_text_water_used = "Analyzing water usage helps assess sustainability efforts and resource management."

            html_code_water_used = generate_card_with_overlay(
                image_url_water_used, 
                button1_text_water_used, 
                button2_text_water_used, 
                button3_text_water_used, 
                card_text_water_used, 
                expln_text_water_used, 
                card_title_water_used
            )

            # Render the graph card in Streamlit
            with colenv3:
                st.components.v1.html(f'<div id="card9_Wrapper">{html_code_water_used}</div>', width=1200, height=550)

            # Waste collected

            # Preprocess the 'Waste (tons) collected or recycled in the past 12 months' column to extract numeric values from strings
            def extract_numeric(value):
                if isinstance(value, (int, float)):
                    return value  # Keep numeric values as-is
                elif isinstance(value, str):
                    match = re.search(r"\d+", value)  # Extract numeric content from the string
                    return float(match.group()) if match else "Missing"  # Convert to float if numeric content exists
                else:
                    return "Missing"  # Non-numeric and non-string values are treated as "Missing"

            df_filtered_by_cohort['Waste (tons) collected or recycled in the past 12 months'] = df_filtered_by_cohort['Waste (tons) collected or recycled in the past 12 months'].apply(extract_numeric)

            # Replace any "Missing" labels for consistency
            df_filtered_by_cohort['Waste (tons) collected or recycled in the past 12 months'] = df_filtered_by_cohort['Waste (tons) collected or recycled in the past 12 months'].replace("Missing", pd.NA)

            # Count occurrences of each unique value, including missing responses
            waste_collected_counts = df_filtered_by_cohort['Waste (tons) collected or recycled in the past 12 months'].value_counts(dropna=False)

            # Sort the counts for numeric values first, then "Missing"
            if pd.NA in waste_collected_counts:
                missing_count = waste_collected_counts.pop(pd.NA)
                waste_collected_counts = waste_collected_counts.sort_index()  # Sort numeric values
                # waste_collected_counts = waste_collected_counts.append(pd.Series({"Missing": missing_count}))  # Add "Missing" back
                waste_collected_counts = pd.concat([waste_collected_counts, pd.Series({"Missing": missing_count})]) # Add "Missing" back

            else:
                waste_collected_counts = waste_collected_counts.sort_index()

            # Create a bar chart
            plt.figure(figsize=(10, 6))
            bars = plt.bar(
                waste_collected_counts.index.astype(str),  # Categories (convert to string for consistent labels)
                waste_collected_counts.values,  # Counts
                color=plt.cm.tab20.colors
            )

            # # Add labels to each bar
            # for bar in bars:
            #     plt.text(
            #         bar.get_x() + bar.get_width() / 2,
            #         bar.get_height() + 1,
            #         f"{bar.get_height()}",
            #         ha='center',
            #         va='bottom'
            #     )

            # Add titles and labels
            plt.title("Waste Collected or Recycled in the Past 12 Months", fontsize=14)
            plt.xlabel("Waste Collected or Recycled (Tons or Missing)", fontsize=12)
            plt.ylabel("Number of Enterprises", fontsize=12)
            plt.xticks(
                ticks=range(len(waste_collected_counts.index)),
                labels=[str(int(x)) if isinstance(x, (int, float)) and not pd.isna(x) else str(x) for x in waste_collected_counts.index],
                rotation=45,
                ha="right"
            )
            plt.tight_layout()

            # Save the plot as base64
            waste_collected_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_waste = waste_collected_chart
            card_title_waste = "Waste Collected or Recycled"
            button1_text_waste = "Analysis"
            button2_text_waste = "Recommendations"
            button3_text_waste = "Clear"
            card_text_waste = "This chart shows the distribution of waste (tons) collected or recycled by enterprises over the past 12 months, including missing values."
            expln_text_waste = "Analyzing waste collection and recycling helps assess sustainability efforts and resource management."

            html_code_waste = generate_card_with_overlay(
                image_url_waste, 
                button1_text_waste, 
                button2_text_waste, 
                button3_text_waste, 
                card_text_waste, 
                expln_text_waste, 
                card_title_waste
            )

            # Render the graph card in Streamlit
            with colenv1:
                st.components.v1.html(f'<div id="card10_Wrapper">{html_code_waste}</div>', width=1200, height=550)

            # Understanding of Ecosystem-based Adaptation

            # Preprocess the field to handle text-based data and missing responses
            df_filtered_by_cohort['From your perspective how much do you understand the concept of Ecosystem-based Adaptation?'] = (
                df_filtered_by_cohort['From your perspective how much do you understand the concept of Ecosystem-based Adaptation?']
                .astype(str)
                .str.strip()
            )

            # Replace NaN or empty strings with "Missing"
            df_filtered_by_cohort['From your perspective how much do you understand the concept of Ecosystem-based Adaptation?'].replace(
                {"nan": "Missing", "": "Missing"}, inplace=True
            )

            # Count occurrences of each response, including missing responses
            understanding_counts = df_filtered_by_cohort['From your perspective how much do you understand the concept of Ecosystem-based Adaptation?'].value_counts()

            # Create a bar chart
            plt.figure(figsize=(10, 6))
            bars = plt.bar(
                understanding_counts.index.astype(str),  # Categories (convert to string for consistent labels)
                understanding_counts.values,  # Counts
                color=plt.cm.tab20.colors
            )

            # Add labels to each bar
            for bar in bars:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{bar.get_height()}",
                    ha='center',
                    va='bottom'
                )

            # Add titles and labels
            plt.title("Understanding of Ecosystem-based Adaptation", fontsize=14)
            plt.xlabel("Understanding Level", fontsize=12)
            plt.ylabel("Number of Enterprises", fontsize=12)
            plt.xticks(
                ticks=range(len(understanding_counts.index)),
                labels=understanding_counts.index,
                rotation=45,
                ha="right"
            )
            plt.tight_layout()

            # Save the plot as base64
            understanding_chart = save_plot_to_base64(plt)

            # Generate the graph card
            image_url_understanding = understanding_chart
            card_title_understanding = "Understanding of Ecosystem-based Adaptation"
            button1_text_understanding = "Analysis"
            button2_text_understanding = "Recommendations"
            button3_text_understanding = "Clear"
            card_text_understanding = "This chart shows the distribution of how enterprises rate their understanding of the concept of Ecosystem-based Adaptation."
            expln_text_understanding = "Understanding the levels of awareness of Ecosystem-based Adaptation helps identify knowledge gaps and training needs."

            html_code_understanding = generate_card_with_overlay(
                image_url_understanding, 
                button1_text_understanding, 
                button2_text_understanding, 
                button3_text_understanding, 
                card_text_understanding, 
                expln_text_understanding, 
                card_title_understanding
            )

            # Render the graph card in Streamlit
            with colenv2:
                st.components.v1.html(f'<div id="card11_Wrapper">{html_code_understanding}</div>', width=1200, height=550)

        target_values = ["All", "General", "Non-core indicators"]

        if any(value in selected_cohort for value in target_values):

            st.subheader("Non-core indicators")

            add_vertical_space(3)

            target_values = ["All", "General", "Non-core indicators", "Economic"]

            if any(value in selected_cohort for value in target_values):

                st.text("Economic indicators")

                coleconnonc1, coleconnonc2, coleconnonc3 = st.columns(3)

                # Preprocess the "What type of finance did you seek?" column
                df_filtered_by_cohort['What type of finance did you seek? Please include all types of finance including cases where you failed to obtain it'] = (
                    df_filtered_by_cohort['What type of finance did you seek? Please include all types of finance including cases where you failed to obtain it']
                    .astype(str)
                    .str.strip()
                )

                # Replace NaN, empty strings, or invalid data with "Missing"
                df_filtered_by_cohort['What type of finance did you seek? Please include all types of finance including cases where you failed to obtain it'].replace(
                    {"nan": "Missing", "": "Missing"}, inplace=True
                )

                # Split comma-separated values into individual categories
                type_exploded = df_filtered_by_cohort[
                    'What type of finance did you seek? Please include all types of finance including cases where you failed to obtain it'
                ].str.split(',').explode()

                # Clean up whitespace and ensure all categories are included
                type_exploded = type_exploded.str.strip()

                # Count occurrences of each type of finance, including "Missing"
                type_counts = type_exploded.value_counts()

                # DEBUG: Print counts
                print("\nType of Finance Counts:")
                print(type_counts)

                # Create the bar chart
                plt.figure(figsize=(12, 8))
                bars = plt.bar(
                    type_counts.index,
                    type_counts.values,
                    color=plt.cm.tab20.colors[:len(type_counts)]
                )

                # Add labels to each bar
                for bar in bars:
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{int(bar.get_height())}",
                        ha='center',
                        va='bottom',
                        fontsize=10
                    )

                # Add chart details
                plt.title("Types of Finance Sought", fontsize=16)
                plt.xlabel("Type of Finance", fontsize=14)
                plt.ylabel("Number of Businesses", fontsize=14)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                # Save the plot as base64
                finance_chart = save_plot_to_base64(plt)

                # Generate the graph card
                image_url_finance = finance_chart
                card_title_finance = "Number Seeking External Finance (With Type)"
                button1_text_finance = "Analysis"
                button2_text_finance = "Recommendations"
                button3_text_finance = "Clear"
                card_text_finance = "This chart shows the distribution of types of finance sought, including multiple categories per respondent."
                expln_text_finance = "Understanding the types of finance sought helps assess business financial needs and access to funding."

                html_code_finance = generate_card_with_overlay(
                    image_url_finance, 
                    button1_text_finance, 
                    button2_text_finance, 
                    button3_text_finance, 
                    card_text_finance, 
                    expln_text_finance, 
                    card_title_finance
                )

                # Render the graph card in Streamlit
                with coleconnonc1:
                    st.components.v1.html(f'<div id="card19_Wrapper">{html_code_finance}</div>', width=1200, height=650)

                # Value of Funding Requested

                # Replace NaN or empty strings with "Missing"
                df_filtered_by_cohort['Value of funding requested'] = df_filtered_by_cohort['Value of funding requested'].replace(
                    {"nan": "Missing", "": "Missing"}
                )

                # Count occurrences of each funding range, including missing values
                funding_counts = df_filtered_by_cohort['Value of funding requested'].value_counts()

                # Order categories for the bar chart
                category_order = funding_counts.index.tolist()  # Dynamically extract unique categories from the data
                if "Missing" in category_order:
                    category_order.remove("Missing")
                category_order.append("Missing")  # Ensure "Missing" is at the end

                # Reorder counts
                funding_counts = funding_counts.reindex(category_order).fillna(0)

                # DEBUG: Print counts
                print("\nFunding Counts:")
                print(funding_counts)

                # Create a bar chart
                plt.figure(figsize=(10, 6))
                bars = plt.bar(
                    funding_counts.index,  # Ordered categories
                    funding_counts.values,  # Counts
                    color=plt.cm.tab20.colors
                )

                # Add labels to each bar
                for bar in bars:
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f"{int(bar.get_height())}",
                        ha='center',
                        va='bottom'
                    )

                # Add titles and labels
                plt.title("Value of Funding Requested", fontsize=14)
                plt.xlabel("Funding Range", fontsize=12)
                plt.ylabel("Number of Requests", fontsize=12)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                # Save the plot as base64
                funding_bar_chart = save_plot_to_base64(plt)

                # Generate the graph card
                image_url_funding = funding_bar_chart
                card_title_funding = "Value of Funding Requested"
                button1_text_funding = "Analysis"
                button2_text_funding = "Recommendations"
                button3_text_funding = "Clear"
                card_text_funding = "This chart shows the distribution of funding requests by value range, including missing values."
                expln_text_funding = "Understanding funding requests helps identify the range of financial needs among businesses."

                html_code_funding = generate_card_with_overlay(
                    image_url_funding, 
                    button1_text_funding, 
                    button2_text_funding, 
                    button3_text_funding, 
                    card_text_funding, 
                    expln_text_funding, 
                    card_title_funding
                )

                # Render the graph card in Streamlit
                with coleconnonc2:
                    st.components.v1.html(f'<div id="card15_Wrapper">{html_code_funding}</div>', width=1200, height=550)

                # Reason for Requesting Funding

                # Replace NaN, empty strings, or any invalid data with "Missing"
                df_filtered_by_cohort['Reason for requesting funding'] = df_filtered_by_cohort[
                    'Reason for requesting funding'
                ].replace({np.nan: "Missing", "": "Missing", None: "Missing"})

                # Split comma-separated values into individual categories
                reason_exploded = df_filtered_by_cohort['Reason for requesting funding'].str.split(',').explode()

                # Strip whitespace and ensure "Missing" is handled
                reason_exploded = reason_exploded.str.strip()
                reason_exploded.replace("", "Missing", inplace=True)

                # Count occurrences of each reason, including "Missing"
                reason_counts = reason_exploded.value_counts()

                # Order categories for the bar chart
                category_order = reason_counts.index.tolist()  # Dynamically extract unique categories
                if "Missing" in category_order:
                    category_order.remove("Missing")
                category_order.append("Missing")  # Ensure "Missing" is at the end

                # Reorder counts
                reason_counts = reason_counts.reindex(category_order).fillna(0)

                # DEBUG: Print counts
                print("\nReason Counts:")
                print(reason_counts)

                # Create a bar chart
                plt.figure(figsize=(10, 6))
                bars = plt.bar(
                    reason_counts.index,  # Ordered categories
                    reason_counts.values,  # Counts
                    color=plt.cm.tab20.colors
                )

                # Add labels to each bar
                for bar in bars:
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f"{int(bar.get_height())}",
                        ha='center',
                        va='bottom'
                    )

                # Add titles and labels
                plt.title("Reasons for Requesting Funding", fontsize=14)
                plt.xlabel("Reason", fontsize=12)
                plt.ylabel("Number of Requests", fontsize=12)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                # Save the plot as base64
                reason_bar_chart = save_plot_to_base64(plt)

                # Generate the graph card
                image_url_reason = reason_bar_chart
                card_title_reason = "Reasons for Requesting Funding"
                button1_text_reason = "Analysis"
                button2_text_reason = "Recommendations"
                button3_text_reason = "Clear"
                card_text_reason = "This chart shows the distribution of reasons for requesting funding, including multiple categories and missing values."
                expln_text_reason = "Understanding reasons for requesting funding helps assess the diverse financial needs of businesses."

                html_code_reason = generate_card_with_overlay(
                    image_url_reason, 
                    button1_text_reason, 
                    button2_text_reason, 
                    button3_text_reason, 
                    card_text_reason, 
                    expln_text_reason, 
                    card_title_reason
                )

                # Render the graph card in Streamlit
                with coleconnonc3:
                    st.components.v1.html(f'<div id="card16_Wrapper">{html_code_reason}</div>', width=1200, height=550)

            target_values = ["All", "General", "Non-core indicators", "Environmental"]

            if any(value in selected_cohort for value in target_values):

                st.text("Environmental indicators")

                colenvnonc1, colenvnonc2, colenvnonc3 = st.columns(3)

                # Preprocess the 'Petrol used per month' column to extract numeric values from strings
                def extract_numeric(value):
                    if isinstance(value, (int, float)):
                        return value  # Keep numeric values as-is
                    elif isinstance(value, str):
                        match = re.search(r"\d+", value)  # Extract numeric content from the string
                        return float(match.group()) if match else "Missing"  # Convert to float if numeric content exists
                    else:
                        return "Missing"  # Non-numeric and non-string values are treated as "Missing"

                df_filtered_by_cohort['Petrol used per month'] = df_filtered_by_cohort['Petrol used per month'].apply(extract_numeric)

                # Replace any "Missing" labels for consistency
                df_filtered_by_cohort['Petrol used per month'] = df_filtered_by_cohort['Petrol used per month'].replace("Missing", pd.NA)

                # Count occurrences of each unique value, including missing responses
                petrol_used_counts = df_filtered_by_cohort['Petrol used per month'].value_counts(dropna=False)

                # Sort the counts for numeric values first, then "Missing"
                if pd.NA in petrol_used_counts:
                    missing_count = petrol_used_counts.pop(pd.NA)
                    petrol_used_counts = petrol_used_counts.sort_index()  # Sort numeric values
                    # petrol_used_counts = petrol_used_counts.append(pd.Series({"Missing": missing_count}))  # Add "Missing" back
                    petrol_used_counts = pd.concat([petrol_used_counts, pd.Series({"Missing": missing_count})]) # Add "Missing" back

                else:
                    petrol_used_counts = petrol_used_counts.sort_index()

                # Create a bar chart
                plt.figure(figsize=(10, 6))
                bars = plt.bar(
                    petrol_used_counts.index.astype(str),  # Categories (convert to string for consistent labels)
                    petrol_used_counts.values,  # Counts
                    color=plt.cm.tab20.colors
                )

                # Add labels to each bar
                for bar in bars:
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f"{bar.get_height()}",
                        ha='center',
                        va='bottom'
                    )

                # Add titles and labels
                plt.title("Petrol Used Per Month Distribution", fontsize=14)
                plt.xlabel("Petrol Used Per Month (Litres or Missing)", fontsize=12)
                plt.ylabel("Number of Enterprises", fontsize=12)
                plt.xticks(
                    ticks=range(len(petrol_used_counts.index)),
                    labels=[str(int(x)) if isinstance(x, (int, float)) and not pd.isna(x) else str(x) for x in petrol_used_counts.index],
                    rotation=45,
                    ha="right"
                )
                plt.tight_layout()

                # Save the plot as base64
                petrol_used_chart = save_plot_to_base64(plt)

                # Generate the graph card
                image_url_petrol = petrol_used_chart
                card_title_petrol = "Petrol Used Per Month"
                button1_text_petrol = "Analysis"
                button2_text_petrol = "Recommendations"
                button3_text_petrol = "Clear"
                card_text_petrol = "This chart shows the distribution of petrol usage (in litres) per month by enterprises, including missing values."
                expln_text_petrol = "Analyzing petrol usage helps assess resource consumption and identify sustainability efforts."

                html_code_petrol = generate_card_with_overlay(
                    image_url_petrol, 
                    button1_text_petrol, 
                    button2_text_petrol, 
                    button3_text_petrol, 
                    card_text_petrol, 
                    expln_text_petrol, 
                    card_title_petrol
                )

                # Render the graph card in Streamlit
                with colenvnonc1:
                    st.components.v1.html(f'<div id="card11_Wrapper">{html_code_petrol}</div>', width=1200, height=550)

                # SDGs Addressed by Enterprises

                # Split the comma-separated values into individual SDGs
                df_filtered_by_cohort['Which of the 17 SDGs is your enterprise addressing? You may select more than one box'] = (
                    df_filtered_by_cohort['Which of the 17 SDGs is your enterprise addressing? You may select more than one box']
                    .astype(str)
                    .str.strip()
                )

                # Expand the values into a flat list for counting
                sdgs_flat_list = df_filtered_by_cohort['Which of the 17 SDGs is your enterprise addressing? You may select more than one box'].str.split(',').explode()

                # Strip whitespace around SDG values
                sdgs_flat_list = sdgs_flat_list.str.strip()

                # Count occurrences of each SDG
                sdg_counts = sdgs_flat_list.value_counts()

                # Create a bar chart
                plt.figure(figsize=(12, 8))
                bars = plt.bar(
                    sdg_counts.index,  # SDGs
                    sdg_counts.values,  # Counts
                    color=plt.cm.tab20.colors
                )

                # Add labels to each bar
                for bar in bars:
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f"{bar.get_height()}",
                        ha='center',
                        va='bottom'
                    )

                # Add titles and labels
                plt.title("Sustainable Development Goals (SDGs) Addressed by Enterprises", fontsize=14)
                plt.xlabel("SDG", fontsize=12)
                plt.ylabel("Number of Enterprises", fontsize=12)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                # Save the plot as base64
                sdg_chart = save_plot_to_base64(plt)

                # Generate the graph card
                image_url_sdg = sdg_chart
                card_title_sdg = "SDGs Addressed by Enterprises"
                button1_text_sdg = "Analysis"
                button2_text_sdg = "Recommendations"
                button3_text_sdg = "Clear"
                card_text_sdg = "This chart shows the distribution of the 17 SDGs addressed by enterprises, based on their responses."
                expln_text_sdg = "Analyzing the alignment with SDGs helps understand enterprises' focus areas and contributions to global sustainability goals."

                html_code_sdg = generate_card_with_overlay(
                    image_url_sdg, 
                    button1_text_sdg, 
                    button2_text_sdg, 
                    button3_text_sdg, 
                    card_text_sdg, 
                    expln_text_sdg, 
                    card_title_sdg
                )

                # Render the graph card in Streamlit
                with colenvnonc2:
                    st.components.v1.html(f'<div id="card13_Wrapper">{html_code_sdg}</div>', width=1200, height=550)

                # Preprocess the column for climate strategy
                df_filtered_by_cohort['Does your enterprise have a climate strategy to reduce its own emissions and become more resilient to climate impacts?'] = (
                    df_filtered_by_cohort['Does your enterprise have a climate strategy to reduce its own emissions and become more resilient to climate impacts?']
                    .astype(str)
                    .str.strip()
                )

                # Replace NaN or empty strings with "Missing"
                df_filtered_by_cohort['Does your enterprise have a climate strategy to reduce its own emissions and become more resilient to climate impacts?'].replace(
                    {"nan": "Missing", "": "Missing"}, inplace=True
                )

                # Count occurrences of each response, including "Missing"
                climate_strategy_counts = df_filtered_by_cohort[
                    'Does your enterprise have a climate strategy to reduce its own emissions and become more resilient to climate impacts?'
                ].value_counts()

                # DEBUG: Print counts
                print("\nClimate Strategy Counts:")
                print(climate_strategy_counts)

                # Create the bar chart
                plt.figure(figsize=(10, 6))
                bars = plt.bar(
                    climate_strategy_counts.index,
                    climate_strategy_counts.values,
                    color=plt.cm.tab20.colors[:len(climate_strategy_counts)]
                )

                # Add labels to each bar
                for bar in bars:
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f"{int(bar.get_height())}",
                        ha='center',
                        va='bottom',
                        fontsize=10
                    )

                # Add chart details
                plt.title("Climate Strategy Adoption by Enterprises", fontsize=16)
                plt.xlabel("Response", fontsize=14)
                plt.ylabel("Number of Enterprises", fontsize=14)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                # Save the plot as base64
                climate_strategy_bar_chart = save_plot_to_base64(plt)

                # Generate the graph card
                image_url_climate_strategy = climate_strategy_bar_chart
                card_title_climate_strategy = "Climate Strategy Adoption by Enterprises"
                button1_text_climate_strategy = "Analysis"
                button2_text_climate_strategy = "Recommendations"
                button3_text_climate_strategy = "Clear"
                card_text_climate_strategy = "This chart shows the distribution of responses to whether enterprises have a climate strategy to reduce emissions and build resilience."
                expln_text_climate_strategy = "Understanding climate strategy adoption helps assess enterprise sustainability and resilience to climate impacts."

                html_code_climate_strategy = generate_card_with_overlay(
                    image_url_climate_strategy, 
                    button1_text_climate_strategy, 
                    button2_text_climate_strategy, 
                    button3_text_climate_strategy, 
                    card_text_climate_strategy, 
                    expln_text_climate_strategy, 
                    card_title_climate_strategy
                )

                # Render the graph card in Streamlit
                with colenvnonc3:
                    st.components.v1.html(f'<div id="card21_Wrapper">{html_code_climate_strategy}</div>', width=1200, height=650)

                # Adopting climate smart/adaptive products or services

                # Preprocess the column for climate smart/adaptive products or services
                df_filtered_by_cohort['Have you introduced new or significantly improved climate smart/adaptive products or services in the past twelve months?'] = (
                    df_filtered_by_cohort['Have you introduced new or significantly improved climate smart/adaptive products or services in the past twelve months?']
                    .astype(str)
                    .str.strip()
                )

                # Replace NaN or empty strings with "Missing"
                df_filtered_by_cohort['Have you introduced new or significantly improved climate smart/adaptive products or services in the past twelve months?'].replace(
                    {"nan": "Missing", "": "Missing"}, inplace=True
                )

                # Count occurrences of each response, including "Missing"
                climate_products_counts = df_filtered_by_cohort[
                    'Have you introduced new or significantly improved climate smart/adaptive products or services in the past twelve months?'
                ].value_counts()

                # DEBUG: Print counts
                print("\nClimate Smart/Adaptive Products Counts:")
                print(climate_products_counts)

                # Create the bar chart
                plt.figure(figsize=(10, 6))
                bars = plt.bar(
                    climate_products_counts.index,
                    climate_products_counts.values,
                    color=plt.cm.tab20.colors[:len(climate_products_counts)]
                )

                # Add labels to each bar
                for bar in bars:
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f"{int(bar.get_height())}",
                        ha='center',
                        va='bottom',
                        fontsize=10
                    )

                # Add chart details
                plt.title("Introduction of Climate Smart/Adaptive Products or Services", fontsize=16)
                plt.xlabel("Response", fontsize=14)
                plt.ylabel("Number of Enterprises", fontsize=14)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                # Save the plot as base64
                climate_products_bar_chart = save_plot_to_base64(plt)

                # Generate the graph card
                image_url_climate_products = climate_products_bar_chart
                card_title_climate_products = "Climate Smart/Adaptive Products or Services"
                button1_text_climate_products = "Analysis"
                button2_text_climate_products = "Recommendations"
                button3_text_climate_products = "Clear"
                card_text_climate_products = "This chart shows the distribution of responses to whether enterprises have introduced climate smart/adaptive products or services in the past 12 months."
                expln_text_climate_products = "Understanding innovation in climate smart/adaptive products helps assess enterprise sustainability efforts and innovation trends."

                html_code_climate_products = generate_card_with_overlay(
                    image_url_climate_products, 
                    button1_text_climate_products, 
                    button2_text_climate_products, 
                    button3_text_climate_products, 
                    card_text_climate_products, 
                    expln_text_climate_products, 
                    card_title_climate_products
                )

                # Render the graph card in Streamlit
                with colenvnonc1:
                    st.components.v1.html(f'<div id="card22_Wrapper">{html_code_climate_products}</div>', width=1200, height=650)

            target_values = ["All", "General", "Non-core indicators", "Social"]

            if any(value in selected_cohort for value in target_values):

                st.text("Social indicators")

                colsocnonc1, colsocnonc2, colsocnonc3 = st.columns(3)

                # Household Size

                # Ensure the column is numeric, handling errors and missing values
                df_filtered_by_cohort['How many people are within your household'] = pd.to_numeric(
                    df_filtered_by_cohort['How many people are within your household'], errors='coerce'
                )

                # Fill missing numeric values with "Missing"
                df_filtered_by_cohort['Processed Household Size'] = df_filtered_by_cohort[
                    'How many people are within your household'
                ]

                # Categorize household sizes
                def categorize_household(value):
                    if value is None or value == "":
                        return "Missing"
                    if value == 1:
                        return "1 Person"
                    elif value == 2:
                        return "2 People"
                    elif value >=3 and value <= 4:
                        return "3-4 People"
                    elif value >=5 and value <= 9:
                        return "5-9 People"
                    elif value >=10 and value <= 20:
                        return "10-20 People"
                    elif value > 20:
                        return "More than 20 people"
                    else:
                        return "Missing"

                df_filtered_by_cohort['Household Size Category'] = df_filtered_by_cohort[
                    'Processed Household Size'
                ].apply(categorize_household)

                # Count occurrences of each household size category
                household_counts = df_filtered_by_cohort['Household Size Category'].value_counts()

                # Order categories on the x-axis
                category_order = ["1 Person", "2 People", "3-4 People", "5-9 People", "10-20 People", "More than 20 people", "Missing"]
                household_counts = household_counts.reindex(category_order).fillna(0)  # Reorder and fill missing categories with 0

                # DEBUG: Print counts
                print("\nHousehold Counts:")
                print(household_counts)

                # Create a bar chart
                plt.figure(figsize=(10, 6))
                bars = plt.bar(
                    household_counts.index,  # Ordered categories
                    household_counts.values,  # Counts
                    color=plt.cm.tab20.colors
                )

                # Add labels to each bar
                for bar in bars:
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f"{int(bar.get_height())}",
                        ha='center',
                        va='bottom'
                    )

                # Add titles and labels
                plt.title("Household Size Distribution", fontsize=14)
                plt.xlabel("Household Size", fontsize=12)
                plt.ylabel("Number of Households", fontsize=12)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                # Save the plot as base64
                household_bar_chart = save_plot_to_base64(plt)

                # Generate the graph card
                image_url_household = household_bar_chart
                card_title_household = "Household Size Distribution"
                button1_text_household = "Analysis"
                button2_text_household = "Recommendations"
                button3_text_household = "Clear"
                card_text_household = "This chart shows the distribution of household sizes, including missing values."
                expln_text_household = "Analyzing household size distribution helps understand demographic patterns and resource allocation needs."

                html_code_household = generate_card_with_overlay(
                    image_url_household, 
                    button1_text_household, 
                    button2_text_household, 
                    button3_text_household, 
                    card_text_household, 
                    expln_text_household, 
                    card_title_household
                )

                # Render the graph card in Streamlit
                with colsocnonc1:
                    st.components.v1.html(f'<div id="card11_Wrapper">{html_code_household}</div>', width=1200, height=550)

                # Ensure the column is numeric, handling errors and missing values
                df_filtered_by_cohort['How many people benefit from your business?'] = pd.to_numeric(
                    df_filtered_by_cohort['How many people benefit from your business?'], errors='coerce'
                )

                # Fill missing numeric values with "Missing"
                df_filtered_by_cohort['Processed Beneficiaries'] = df_filtered_by_cohort[
                    'How many people benefit from your business?'
                ]

                # Categorize the number of beneficiaries
                def categorize_beneficiaries(value):
                    if pd.isna(value):
                        return "Missing"
                    elif value == 1:
                        return "1 Person"
                    elif value == 2:
                        return "2 People"
                    elif 3 <= value <= 10:
                        return "3-10 People"
                    elif 11 <= value <= 50:
                        return "11-50 People"
                    elif value > 50:
                        return "More than 50 People"
                    else:
                        return "Missing"

                df_filtered_by_cohort['Beneficiary Category'] = df_filtered_by_cohort[
                    'Processed Beneficiaries'
                ].apply(categorize_beneficiaries)

                # Count occurrences of each beneficiary category
                beneficiary_counts = df_filtered_by_cohort['Beneficiary Category'].value_counts()

                # Order categories on the x-axis
                category_order = ["1 Person", "2 People", "3-10 People", "11-50 People", "More than 50 People", "Missing"]
                beneficiary_counts = beneficiary_counts.reindex(category_order).fillna(0)  # Reorder and fill missing categories with 0

                # DEBUG: Print counts
                print("\nBeneficiary Counts:")
                print(beneficiary_counts)

                # Create a bar chart
                plt.figure(figsize=(10, 6))
                bars = plt.bar(
                    beneficiary_counts.index,  # Ordered categories
                    beneficiary_counts.values,  # Counts
                    color=plt.cm.tab20.colors
                )

                # Add labels to each bar
                for bar in bars:
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f"{int(bar.get_height())}",
                        ha='center',
                        va='bottom'
                    )

                # Add titles and labels
                plt.title("Number of Beneficiaries", fontsize=14)
                plt.xlabel("Beneficiary Range", fontsize=12)
                plt.ylabel("Number of Businesses", fontsize=12)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                # Save the plot as base64
                beneficiary_bar_chart = save_plot_to_base64(plt)

                # Generate the graph card
                image_url_beneficiary = beneficiary_bar_chart
                card_title_beneficiary = "Beneficiary Distribution"
                button1_text_beneficiary = "Analysis"
                button2_text_beneficiary = "Recommendations"
                button3_text_beneficiary = "Clear"
                card_text_beneficiary = "This chart shows the distribution of people benefiting from businesses, including missing values."
                expln_text_beneficiary = "Understanding the number of beneficiaries helps assess the social impact of businesses."

                html_code_beneficiary = generate_card_with_overlay(
                    image_url_beneficiary, 
                    button1_text_beneficiary, 
                    button2_text_beneficiary, 
                    button3_text_beneficiary, 
                    card_text_beneficiary, 
                    expln_text_beneficiary, 
                    card_title_beneficiary
                )

                # Render the graph card in Streamlit
                with colsocnonc2:
                    st.components.v1.html(f'<div id="card12_Wrapper">{html_code_beneficiary}</div>', width=1200, height=550)

    with dbrdmetrics:

        style_custom_metric_cards()
        
        add_vertical_space(3)

        # Function to count non-missing values for a selected variable
        def count_non_missing(df, column):
            if column in df.columns:
                return df[column].notna().sum()  # Count non-NA values
            else:
                return 0

        # Dynamically get all columns from the DataFrame
        all_columns = df_filtered_by_cohort.columns.tolist()

        # User selects the variable from all columns
        selected_variable = st.selectbox("Select a variable to analyze:", all_columns)

        # Count non-missing values for the selected variable
        reported_count = count_non_missing(df_filtered_by_cohort, selected_variable)

        # Create the metric card dynamically
        colmissing2, colmissing2, colmissing3 = st.columns([1,3,1])  # Adjust the layout as needed
        colmissing2.markdown(
            create_metric_card(
                selected_variable,  # Use the selected variable as the title
                f"{reported_count:,.0f}",  # Display the count of non-missing values
                f"Number of enterprises with reported information for {selected_variable}"
            , centered=True),
            unsafe_allow_html=True,
        )

        add_vertical_space(3)

        target_values = ["All", "General"]

        if any(value in selected_cohort for value in target_values):

            st.subheader("General indicators")

            add_vertical_space(3)

            st.text("Internal reporting and key business trends")

            # Totals

            # totalindalovator = df_combined[df_combined['Cohort'] == 'Indalovator'].count()
            # totalindalogrow = df_combined[df_combined['Cohort'] =='Indalogrow'].count()
            # totalindaloexcel = df_combined[df_combined['Cohort'] =='Indaloexcel'].count()

            # --- Indalovator/Indalogrow Programs ---
            # Supported businesses in each program
            indalovator_count = df_consolidated[df_consolidated['Cohort'] == 'Indalovator'].shape[0]
            indalogrow_count = df_consolidated[df_consolidated['Cohort']=='Indalogrow'].shape[0]
            indaloaccel_count = df_consolidated[df_consolidated['Cohort']=='Indaloaccel'].shape[0]
            indalouncat_count = df_consolidated[df_consolidated['Cohort']=='Uncategorized'].shape[0]
            total_businesses = indalovator_count + indalogrow_count + indaloaccel_count + indalouncat_count
            st.markdown(create_metric_card("Total enterprises supported", total_businesses, f"Businesses supported under all programs, with {indalouncat_count} currently uncategorized", centered=True), unsafe_allow_html=True)
            add_vertical_space(3)
            colm1, colm2, colm3 = st.columns(3)
            colm1.markdown(create_metric_card("Indalovator Programme", indalovator_count, "Businesses supported under Indalovator"), unsafe_allow_html=True)
            colm2.markdown(create_metric_card("Indalogrow Programme", indalogrow_count, "Businesses supported under Indalogrow"), unsafe_allow_html=True)
            colm3.markdown(create_metric_card("IndaloAccel Programme", indaloaccel_count, "Businesses supported under IndaloAccel"), unsafe_allow_html=True)

            # print(totalindalovator)

            # # Metric Card: Indalovator
            # with colm1:
            #     st.markdown(
            #         f"""
            #         <div data-testid="custom-metric-container">
            #             <div class="metric-label" style="font-weight: bold;">{"Total enterprises in Indalovator"}</div>
            #             <div class="metric-value" style="font-weight: normal;">{f"{totalindalovator[0]}"}</div>
            #             <div class="custom-metric-delta" style="color: #E8E8E8;">{"Placeholder"}</div>
            #         </div>
            #         """,
            #         unsafe_allow_html=True
            #     )

            # # Metric Card: Indalogrow
            # with colm2:
            #     st.markdown(
            #         f"""
            #         <div data-testid="custom-metric-container">
            #             <div class="metric-label" style="font-weight: bold;">{"Total enterprises in Indalogrow"}</div>
            #             <div class="metric-value" style="font-weight: normal;">{f"{totalindalogrow[0]}"}</div>
            #             <div class="custom-metric-delta" style="color: #E8E8E8;">{"Placeholder"}</div>
            #         </div>
            #         """,
            #         unsafe_allow_html=True
            #     )

            # # Metric Card: Indaloexcel
            # with colm3:
            #     st.markdown(
            #         f"""
            #         <div data-testid="custom-metric-container">
            #             <div class="metric-label" style="font-weight: bold;">{"Total enterprises in Indaloexcel"}</div>
            #             <div class="metric-value" style="font-weight: normal;">{f"{totalindaloexcel[0]}"}</div>
            #             <div class="custom-metric-delta" style="color: #E8E8E8;">{"Placeholder"}</div>
            #         </div>
            #         """,
            #         unsafe_allow_html=True
            #     )

        target_values = ["All", "Core indicators", "Economic"]

        if any(value in selected_cohort for value in target_values):

            add_vertical_space(3)

            st.subheader("Core indicators")

            st.text("Economic indicators")

            colm1, colm2, colm3 = st.columns(3)

            # Revenue and Profit Metric Cards

            revenue = df_filtered_by_cohort['Revenue amount in the past financial year'].sum()
            profit = df_filtered_by_cohort['Profit amount in the past financial year'].sum()
            total_jobs = df_filtered_by_cohort['No. of full-time employees employed by the enterprise'].sum() + df_filtered_by_cohort['No. of part time employees employed by the enterprise'].sum()
            
            # Metric Card 1: Revenue
            with colm1:
                st.markdown(
                    f"""
                    <div data-testid="custom-metric-container">
                        <div class="metric-label" style="font-weight: bold;">{"Total revenue in past year"}</div>
                        <div class="metric-value" style="font-weight: normal;">{f"R {revenue:,.2f}"}</div>
                        <div class="custom-metric-delta" style="color: #E8E8E8;">{"Placeholder"}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Metric Card 2: Revenue
            with colm2:
                st.markdown(
                    f"""
                    <div data-testid="custom-metric-container">
                        <div class="metric-label" style="font-weight: bold;">{"Total profit in past year"}</div>
                        <div class="metric-value" style="font-weight: normal;">{f"R {profit:,.2f}"}</div>
                        <div class="custom-metric-delta" style="color: #E8E8E8;">{"Placeholder"}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Metric Card 3: Jobs
            with colm3:
                st.markdown(
                    f"""
                    <div data-testid="custom-metric-container">
                        <div class="metric-label" style="font-weight: bold;">{"Jobs created"}</div>
                        <div class="metric-value" style="font-weight: normal;">{f"{total_jobs:,.0f}"}</div>
                        <div class="custom-metric-delta">{"Full-time and part-time (combined)"}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            add_vertical_space(3)
            
            st.text("Diversity")

            # Women

            womenled = df_filtered_by_cohort[df_filtered_by_cohort['Gender'] == 'Female'].count()
            numfemales = round(df_filtered_by_cohort['No. of partners or directors that are women'].sum())

            # # Metric Card 3: Jobs
            # with colm1:
            #     st.markdown(
            #         f"""
            #         <div data-testid="custom-metric-container">
            #             <div class="metric-label" style="font-weight: bold;">{"Female-led enterprises"}</div>
            #             <div class="metric-value" style="font-weight: normal;">{f"{womenled[0]}"}</div>
            #             <div class="custom-metric-delta">{f"{numfemales} female-led enterprises"}</div>
            #         </div>
            #         """,
            #         unsafe_allow_html=True
            #     )            

            # Water Usage Metric
            # total_water = indalovator_data['Water used per month'].fillna(0).sum()
            # st.metric(label="Total Water Usage", value=f"{total_water:,.2f} Litres")

            # style_custom_metric_cards()

            colt1, colt2, colt3 = st.columns(3)

            # --- General Indicators ---
            # Women-owned businesses supported
            women_owned = df_filtered_by_cohort['No. of partners or directors that are women'].sum()
            totalowners = df_filtered_by_cohort['No. of owners or partners the business currently has'].sum()

            colt1.markdown(create_metric_card("Women-Owned Businesses", f"{women_owned:,.0f} ({womenled[0]} enterprises led by women)", "Number of women-owned businesses supported"), unsafe_allow_html=True)
            colt2.markdown(create_metric_card("Percentage of female partners", f"{round(women_owned*100/totalowners, 2):,.2f}%", "Ratio of female partners to total"), unsafe_allow_html=True)

            # # Businesses supporting people with disabilities
            # disabilities_support = df_combined['No. of businesses supported of people living with disabilities'].sum()
            # results.append(create_metric_card("Disability Support", disabilities_support, "Businesses supporting people with disabilities"))

            # Youth-owned businesses supported
            youth_supported = df_filtered_by_cohort['No. of paid youth (between 18-35) employees employed in the enterprise'].sum()
            count_youth_employing_enterprises = (df_filtered_by_cohort['No. of paid youth (between 18-35) employees employed in the enterprise'] > 0).sum()
            colt3.markdown(create_metric_card("Number of youths employed", f"{round(youth_supported)} ({count_youth_employing_enterprises} enterprises)", "Businesses employing youths supported"), unsafe_allow_html=True)

            add_vertical_space(3)

            disabled_supported = df_filtered_by_cohort['How many persons of disabilities does the enterprise employ (full-time, part-time or volunteers)'].sum()
            count_disabled_employing_enterprises = (df_filtered_by_cohort['How many persons of disabilities does the enterprise employ (full-time, part-time or volunteers)'] > 0).sum()
            colt1.markdown(create_metric_card("Number of disabled individuals employed", f"{round(disabled_supported)} ({count_disabled_employing_enterprises} enterprises)", "Businesses employing disabled individuals supported"), unsafe_allow_html=True)

            st.text("Skills, training, development")
            
            colskills1, colskills2, colskills3 = st.columns(3)

            # Calculate the total for "How many surrounding communities are directly/indirectly benefiting from your enterprise?"
            total_benefiting_communities = df_filtered_by_cohort['How many surrounding communities are directly/indirectly benefiting from your enterprise?'].sum()

            # Count the number of enterprises reporting data (non-missing values)
            num_reporting_enterprises = df_filtered_by_cohort['How many surrounding communities are directly/indirectly benefiting from your enterprise?'].notna().sum()

            # Generate the metric card
            metric_card_html = create_metric_card(
                "How many surrounding communities",
                f"{int(total_benefiting_communities)} ({num_reporting_enterprises} enterprises reporting)",
                "Communities benefiting from enterprises"
            )
            colskills1.markdown(metric_card_html, unsafe_allow_html=True)

            add_vertical_space(3)

            target_values = ["All", "Core indicators", "Financial"]

            if any(value in selected_cohort for value in target_values):

                st.text("Financial indicators")

                colfin1, colfin2, colfin3 = st.columns(3)

                # Revenue and Profit Metric Cards

                revenue = df_filtered_by_cohort['Revenue amount in the past financial year'].sum()
                profit = df_filtered_by_cohort['Profit amount in the past financial year'].sum()
                total_jobs = df_filtered_by_cohort['No. of full-time employees employed by the enterprise'].sum() + df_filtered_by_cohort['No. of part time employees employed by the enterprise'].sum()

                # Metric Card 1: Revenue
                with colfin1:
                    st.markdown(
                        f"""
                        <div data-testid="custom-metric-container">
                            <div class="metric-label" style="font-weight: bold;">{"Total revenue in past year"}</div>
                            <div class="metric-value" style="font-weight: normal;">{f"R {revenue:,.2f}"}</div>
                            <div class="custom-metric-delta" style="color: #E8E8E8;">{"Placeholder"}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Metric Card 2: Revenue
                with colfin2:
                    st.markdown(
                        f"""
                        <div data-testid="custom-metric-container">
                            <div class="metric-label" style="font-weight: bold;">{"Total profit in past year"}</div>
                            <div class="metric-value" style="font-weight: normal;">{f"R {profit:,.2f}"}</div>
                            <div class="custom-metric-delta" style="color: #E8E8E8;">{"Placeholder"}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            target_values = ["All", "Non-core indicators"]

            if any(value in selected_cohort for value in target_values):

                add_vertical_space(3)
                
                st.subheader("Non-core indicators")

                add_vertical_space(3)

                target_values = ["All", "Core indicators", "Economic"]

                if any(value in selected_cohort for value in target_values):

                    st.text("Economic indicators")

                    coleconnonc1, coleconnonc2, coleconnonc3 = st.columns(3)

                    # Preprocess the "Did you have any difficulties obtaining this finance?" column
                    df_filtered_by_cohort['Did you have any difficulties obtaining this finance from the sources you approached?'] = (
                        df_filtered_by_cohort['Did you have any difficulties obtaining this finance from the sources you approached?']
                        .astype(str)
                        .str.strip()
                    )

                    # Replace NaN or empty strings with "Missing"
                    df_filtered_by_cohort['Did you have any difficulties obtaining this finance from the sources you approached?'].replace(
                        {"nan": "Missing", "": "Missing"}, inplace=True
                    )

                    # Count the occurrences of "Yes" and "No"
                    difficulty_counts = df_filtered_by_cohort[
                        'Did you have any difficulties obtaining this finance from the sources you approached?'
                    ].value_counts()

                    # Extract the raw numbers for Yes and No
                    num_yes = difficulty_counts.get("Yes", 0)
                    num_no = difficulty_counts.get("No", 0)

                    # Calculate the ratio
                    if num_no > 0:
                        yes_to_no_ratio = f"{num_yes}:{num_no}"
                    else:
                        yes_to_no_ratio = "Undefined"  # Avoid division by zero

                    # Generate the metric card HTML
                    metric_card_html = create_metric_card(
                        "Finance Difficulty Ratio",
                        f"{yes_to_no_ratio} (Yes: {num_yes}, No: {num_no})",
                        "Ratio of Yes to No responses for finance difficulties"
                    )

                    # Render the metric card in Streamlit
                    with coleconnonc1:
                        st.markdown(metric_card_html, unsafe_allow_html=True)

                    from collections import Counter
                    from fuzzywuzzy import process

                    # Preprocess the column
                    df_filtered_by_cohort[' In instances where you experienced challenges obtaining finance, what reasons were given for your application being turned down or for receiving less finance than you requested?'] = (
                        df_filtered_by_cohort[' In instances where you experienced challenges obtaining finance, what reasons were given for your application being turned down or for receiving less finance than you requested?']
                        .astype(str)
                        .str.strip()
                    )

                    # Replace NaN or empty strings with "Missing"
                    df_filtered_by_cohort[' In instances where you experienced challenges obtaining finance, what reasons were given for your application being turned down or for receiving less finance than you requested?'].replace(
                        {"nan": "Missing", "": "Missing"}, inplace=True
                    )

                    # Extract all reasons as a list, excluding "Missing"
                    reasons = df_filtered_by_cohort[
                        ' In instances where you experienced challenges obtaining finance, what reasons were given for your application being turned down or for receiving less finance than you requested?'
                    ].tolist()
                    reasons = [reason for reason in reasons if reason.lower() != "missing"]

                    # Group similar reasons using fuzzy matching
                    grouped_reasons = []
                    grouped_reason_counts = Counter()

                    # Define some common groups for fuzzy matching
                    reason_categories = [
                        "Insufficient collateral", "Poor credit history", "Business too new", 
                        "Lack of profitability", "Incomplete documentation", "Did not meet the financial institution's criteria", "Credit score", "Risky sector", "No reason given", "Poor quality application",
                        "Other"
                    ]

                    # Fuzzy match each reason to predefined categories
                    for reason in reasons:
                        match, score = process.extractOne(reason, reason_categories)
                        if score > 50:  # Match threshold
                            grouped_reasons.append(match)
                            grouped_reason_counts[match] += 1
                        else:
                            grouped_reasons.append("Other")
                            grouped_reason_counts["Other"] += 1

                    # Get the top 3 reasons
                    top_reasons = grouped_reason_counts.most_common(10)

                    # Create the metric card content
                    if top_reasons:
                        top_reasons_text = "<br>".join([f"{reason}: {count}" for reason, count in top_reasons])
                    else:
                        top_reasons_text = "No reasons available."

                    # Split the top reasons into two groups: main body (top 3) and explanation (next 3)
                    main_reasons = top_reasons[:3]
                    next_reasons = top_reasons[3:6]

                    # Format the top three reasons for the main body
                    main_reasons_text = "<br>".join([f"{reason}: {count}" for reason, count in main_reasons])

                    # Format the next three reasons for the explanation
                    next_reasons_text = "<br>".join([f"{reason}: {count}" for reason, count in next_reasons]) if next_reasons else "No additional reasons available."

                    # Generate the metric card HTML
                    metric_card_html = create_metric_card(
                        "Top Reasons for Finance Challenges",
                        f"{main_reasons_text}",
                        f"The next most common reasons are:<br>{next_reasons_text}")

                    # Render the metric card in Streamlit
                    with coleconnonc2:
                        st.markdown(metric_card_html, unsafe_allow_html=True)

                add_vertical_space(3)

                target_values = ["All", "Core indicators", "Environmental"]

                if any(value in selected_cohort for value in target_values):
                
                    st.text("Environmental indicators")

                    colenvnonc1, colenvnonc2, colenvnonc3 = st.columns(3)

                    # Kilometres saved

                    # Function to extract numeric values from strings
                    def extract_numeric(value):
                        if isinstance(value, (int, float)):
                            return value  # Keep numeric values as-is
                        elif isinstance(value, str):
                            match = re.search(r"\d+", value)  # Extract numeric content from the string
                            return float(match.group()) if match else 0  # Convert to float if numeric content exists
                        else:
                            return 0  # Non-numeric and non-string values are treated as 0

                    # Apply numeric extraction to the field
                    df_filtered_by_cohort['Car kilometres saved because of sustainable transport or km reduced'] = (
                        df_filtered_by_cohort['Car kilometres saved because of sustainable transport or km reduced']
                        .apply(extract_numeric)
                    )

                    # Calculate the total kilometres saved
                    total_kilometres_saved = df_filtered_by_cohort['Car kilometres saved because of sustainable transport or km reduced'].sum()

                    # Count the number of enterprises reporting data (non-missing values)
                    num_reporting_enterprises = df_filtered_by_cohort['Car kilometres saved because of sustainable transport or km reduced'].notna().sum()

                    # Generate the metric card
                    metric_card_html = create_metric_card(
                        "Car kilometres saved",
                        f"{int(total_kilometres_saved):,} ({num_reporting_enterprises} enterprises reporting)",
                        "Kilometres saved due to sustainable transport or reduced"
                    )
                    colenvnonc1.markdown(metric_card_html, unsafe_allow_html=True)

                    # Ha used

                    # Ensure the column is numeric, coercing errors to NaN
                    df_filtered_by_cohort['ha of land planted on and amount of the land is used for organic produce'] = pd.to_numeric(
                        df_filtered_by_cohort['ha of land planted on and amount of the land is used for organic produce'], 
                        errors='coerce'
                    )

                    # Calculate the total hectares of land planted on (including organic produce)
                    total_hectares_planted = df_filtered_by_cohort['ha of land planted on and amount of the land is used for organic produce'].sum()
                    
                    # Count the number of enterprises reporting data (non-missing values)
                    num_reporting_enterprises_land = df_filtered_by_cohort['ha of land planted on and amount of the land is used for organic produce'].notna().sum()

                    # Generate the metric card
                    metric_card_html_land = create_metric_card(
                        "Hectares of Land Planted",
                        f"{total_hectares_planted:,.2f} ha ({num_reporting_enterprises_land} enterprises reporting)",
                        "Total hectares of land planted on, including organic produce"
                    )
                    colenvnonc2.markdown(metric_card_html_land, unsafe_allow_html=True)

    # with dbrdvaluemap:

        # # Define the iframe HTML code
        # iframe_code = """
        # <iframe title="SROI value map" width="1660" height="1000" src="https://docs.google.com/spreadsheets/d/1tOpeThGOkRGS7YQrnbh4tPb8sbFrAS9NeuqevEsP21o/edit?usp=sharing"" frameborder="10" allowFullScreen="true"></iframe>
        # """

        # # Use components.html to embed the iframe
        # components.html(iframe_code, height=1000, width=1600)

    with dbrdpyg:
            
        st.title("Custom analytics")

        df_tableau = df_filtered_by_cohort.copy()

        # Function to clean special characters and handle None values
        def clean_special_chars(df_tableau):
            for col in df_tableau.columns:
                df_tableau[col] = df_tableau[col].astype(str).fillna('').apply(lambda x: x.encode('utf-8', 'ignore').decode('utf-8'))
            return df_tableau

        # Clean the dataframe
        df_cleaned = clean_special_chars(df_tableau)

        # Ensure the dataframe is properly encoded in UTF-8
        df_cleaned = df_cleaned.apply(lambda x: x.str.encode('utf-8').str.decode('utf-8'))

        # Use the cleaned dataframe with mitosheet
        from mitosheet.streamlit.v1 import spreadsheet
        spreadsheet(df_cleaned)

        st.write("")

        # Title for the dashboard
        st.title("Custom visualizations")

        # Display data types and any problematic values
        # st.write("Data Types:")
        # st.write(df_tableau.dtypes)

        # Separate numeric and non-numeric columns
        numeric_cols = df_tableau.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df_tableau.select_dtypes(exclude=[np.number]).columns

        # Check for NaN and Infinite values in numeric columns
        st.write("Checking for NaN or Infinite values in numeric columns...")
        nan_count_numeric = df_tableau[numeric_cols].isna().sum().sum()
        inf_count_numeric = np.isinf(df_tableau[numeric_cols]).sum().sum()

        # Check for NaN values in non-numeric columns
        st.write("Checking for NaN values in non-numeric columns...")
        nan_count_non_numeric = df_tableau[non_numeric_cols].isna().sum().sum()

        st.write(f"NaN count in numeric columns: {nan_count_numeric}")
        st.write(f"Infinite count in numeric columns: {inf_count_numeric}")
        st.write(f"NaN count in non-numeric columns: {nan_count_non_numeric}")

        # Fill NaN/Infinite values with appropriate values
        if nan_count_numeric > 0 or inf_count_numeric > 0 or nan_count_non_numeric > 0:
            st.write("Filling NaN and Infinite values...")
            df_tableau[numeric_cols] = df_tableau[numeric_cols].replace([np.inf, -np.inf], np.nan)
            df_tableau[numeric_cols] = df_tableau[numeric_cols].fillna(0)
            
            # Handle non-numeric columns
            for col in non_numeric_cols:
                if pd.api.types.is_categorical_dtype(df_tableau[col]):
                    df_tableau[col] = df_tableau[col].astype(str).fillna('missing')
                else:
                    df_tableau[col] = df_tableau[col].fillna('missing')

        # Convert all columns to appropriate types
        for col in df_tableau.columns:
            if df_tableau[col].dtype == np.object:
                df_tableau[col] = df_tableau[col].astype(str)

        # Establish communication between pygwalker and Streamlit
        from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
        init_streamlit_comm()

        # Get an instance of pygwalker's renderer and cache it to manage memory effectively
        @st.cache_resource
        def get_pyg_renderer(dataframe: pd.DataFrame) -> StreamlitRenderer:
            return StreamlitRenderer(dataframe, spec="./gw_config.json", debug=False)

        renderer = get_pyg_renderer(df_tableau)

        # Render the data exploration interface
        renderer.render_explore()

        # Additionally, create an instance of StreamlitRenderer for direct use if needed
        pyg_app = StreamlitRenderer(df_tableau)

    with dbrdanalytics:

        # Step 1: Handling NaN and infinite values
        # Replace infinite values with NaN
        df_filtered_by_cohort.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop columns with too many NaN values
        threshold = 0.5  # Drop columns with more than 50% missing values
        # df.dropna(thresh=int(threshold * len(df)), axis=1, inplace=True)

        # Fill remaining NaN values with the median of the column
        df_filtered_by_cohort.fillna(df_filtered_by_cohort.median(), inplace=True)

        # Step 2: Correlation Analysis by chosen cohort
        correlation_matrix = df_filtered_by_cohort.corr()
        st.header("Correlation matrix")
        st.write(correlation_matrix)
        
        st.header("Indalo analytics")
        
        st.info("The objective is to determine, for a given variable of interest, which other variables are most impactful, and to determine to the best degree how to manage them.")

        def generate_card_with_overlay(image_url, button1_text, button2_text, button3_text, card_text, expln_text, card_title):
            html_code = f'''
                <style>
                    .card-container {{
                        display: flex;
                        border: 1px solid #fff; /* Change to white background */
                        width: 375px;
                        overflow: hidden;
                        flex-direction: column;
                        margin-bottom: 20px;
                        background-color: #f0f0f0; /* Grey background */
                        padding: 15px; /* Optional padding */
                        border-radius: 10px; /* Optional rounded corners */
                    }}
                    .card-left {{
                        padding: 10px;
                        cursor: pointer;
                        font-family: 'Roboto', sans-serif;
                        font-size: 12px;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: flex-start;
                    }}
                    .card-left img {{
                        max-width: 100%;
                        max-height: 100%;
                    }}
                    .card-right {{
                        padding: 10px;
                        font-family: 'Roboto', sans-serif;
                        font-size: 14px;
                        overflow-y: hidden;
                    }}
                    .overlay {{
                        display: none;
                        position: fixed;
                        top: 1px;
                        left: 0;
                        width: 90%;
                        height: 90%;
                        background-color: transparent; /* Transparent background */
                        z-index: 1;
                        justify-content: center;
                        align-items: center;
                    }}
                    .overlay-image {{
                        max-width: 80%; /* Adjust the width of the overlay image */
                        max-height: 80%; /* Adjust the height of the overlay image */
                    }}
                    .card-title {{
                        text-align: center;
                        margin-top: 10px;
                        font-family: 'Roboto', sans-serif;
                        font-size: 14px;
                    }}
                    .button-container {{
                        display: flex;
                        justify-content: center;
                    }}
                    button {{
                        background-color: #207DCE;
                        color: white;
                        padding: 10px 15px;
                        border: none;
                        border-radius: 15px; /* Make buttons rounded */
                        cursor: pointer;
                        margin: 5px;
                    }}
                </style>
                <div class="card-container" style="font-family: 'Roboto', sans-serif;">
                    <h3 class="card-title">{card_title}</h3>
                    <div class="card-left" onclick="openOverlay()">
                        <img src="{image_url}" style="width: 375px; height: 250px; margin-top: 25px;">
                    </div>
                    <div class="card-right">
                        <div class="button-container">
                            <button id="graph_expln_button">{button1_text}</button>
                            <button id="graph_expln_button2">{button2_text}</button>
                            <button id="reset_button">{button3_text}</button>
                        </div>
                        <p id="dynamic_heading"></p>
                        <p class="card-text" id="dynamic_text">{" "}</p>
                        <p class="card-text"><small class="text-muted">Last updated 3 mins ago</small></p>
                    </div>
                </div>
                <div class="overlay" id="imageOverlay" onclick="closeOverlay()">
                    <img src="{image_url}" class="overlay-image">
                </div>
                <script>
                    function openOverlay() {{
                        document.getElementById("imageOverlay").style.display = "block";
                    }}
                    function closeOverlay() {{
                        document.getElementById("imageOverlay").style.display = "none";
                    }}
                    document.getElementById("graph_expln_button").addEventListener("click", function() {{
                        document.getElementById("dynamic_heading").style.fontWeight = "bold";
                        document.getElementById("dynamic_heading").innerText = "Analysis";
                        document.getElementById("dynamic_text").innerText = "{card_text}";
                        adjustCardHeight();
                    }});
                    document.getElementById("graph_expln_button2").addEventListener("click", function() {{
                        document.getElementById("dynamic_heading").style.fontWeight = "bold";
                        document.getElementById("dynamic_heading").innerText = "Recommendations";
                        document.getElementById("dynamic_text").innerText = "{expln_text}";
                        adjustCardHeight();
                    }});
                    document.getElementById("reset_button").addEventListener("click", function() {{
                        document.getElementById("dynamic_heading").innerText = "";
                        document.getElementById("dynamic_text").innerText = "";
                        adjustCardHeight();
                    }});
                    function adjustCardHeight() {{
                        var cardContainer = document.querySelector('.card-container');
                        cardContainer.style.height = 'auto';
                    }}
                </script>
            '''
            return html_code

        def generate_card_with_overlay_interactive(image_url, button1_text, button2_text, button3_text, card_text, expln_text, card_title):        
            html_code_interactive = f'''
                <style>
                    .card-container {{
                        display: flex;
                        border: 1px solid #fff; /* Change to white background */
                        width: 375px;
                        overflow: hidden;
                        flex-direction: column;
                        margin-bottom: 20px;
                        background-color: #f0f0f0; /* Grey background */
                        padding: 15px; /* Optional padding */
                        border-radius: 10px; /* Optional rounded corners */
                    }}
                    .card-left {{
                        padding: 10px;
                        cursor: pointer;
                        font-family: 'Roboto', sans-serif;
                        font-size: 12px;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: flex-start;
                    }}
                    .card-left img {{
                        max-width: 100%;
                        max-height: 100%;
                    }}
                    .card-right {{
                        padding: 10px;
                        font-family: 'Roboto', sans-serif;
                        font-size: 14px;
                        overflow-y: hidden;
                    }}
                    .overlay {{
                        display: none;
                        position: fixed;
                        top: 1px;
                        left: 0;
                        width: 90%;
                        height: 90%;
                        background-color: transparent; /* Transparent background */
                        z-index: 1;
                        justify-content: center;
                        align-items: center;
                    }}
                    .overlay-image {{
                        max-width: 80%; /* Adjust the width of the overlay image */
                        max-height: 80%; /* Adjust the height of the overlay image */
                    }}
                    .card-title {{
                        text-align: center;
                        margin-top: 10px;
                        font-family: 'Roboto', sans-serif;
                        font-size: 14px;
                    }}
                    .button-container {{
                        display: flex;
                        justify-content: center;
                    }}
                    button {{
                        background-color: #207DCE;
                        color: white;
                        padding: 10px 15px;
                        border: none;
                        border-radius: 15px; /* Make buttons rounded */
                        cursor: pointer;
                        margin: 5px;
                    }}
                </style>
                <div class="card-container" style="font-family: 'Roboto', sans-serif;">
                    <h3 class="card-title">{card_title}</h3>
                    <div class="card-left" onclick="openOverlay()">
                        {image_url}
                    </div>
                    <div class="card-right">
                        <div class="button-container">
                            <button id="graph_expln_button">{button1_text}</button>
                            <button id="graph_expln_button2">{button2_text}</button>
                            <button id="reset_button">{button3_text}</button>
                        </div>
                        <p id="dynamic_heading"></p>
                        <p class="card-text" id="dynamic_text">{" "}</p>
                        <p class="card-text"><small class="text-muted">Last updated 3 mins ago</small></p>
                    </div>
                </div>
                <!--<div class="overlay" id="imageOverlay" onclick="closeOverlay()">
                    {image_url}
                </div>-->
                <script>
                    function openOverlay() {{
                        document.getElementById("imageOverlay").style.display = "block";
                    }}
                    function closeOverlay() {{
                        document.getElementById("imageOverlay").style.display = "none";
                    }}
                    document.getElementById("graph_expln_button").addEventListener("click", function() {{
                        document.getElementById("dynamic_heading").style.fontWeight = "bold";
                        document.getElementById("dynamic_heading").innerText = "Analysis";
                        document.getElementById("dynamic_text").innerText = "{card_text}";
                        adjustCardHeight();
                    }});
                    document.getElementById("graph_expln_button2").addEventListener("click", function() {{
                        document.getElementById("dynamic_heading").style.fontWeight = "bold";
                        document.getElementById("dynamic_heading").innerText = "Recommendations";
                        document.getElementById("dynamic_text").innerText = "{expln_text}";
                        adjustCardHeight();
                    }});
                    document.getElementById("reset_button").addEventListener("click", function() {{
                        document.getElementById("dynamic_heading").innerText = "";
                        document.getElementById("dynamic_text").innerText = "";
                        adjustCardHeight();
                    }});
                    function adjustCardHeight() {{
                        var cardContainer = document.querySelector('.card-container');
                        cardContainer.style.height = 'auto';
                    }}
                </script>
            '''
            return html_code_interactive
        
        # Add JavaScript to make the row height dynamic
        st.markdown(
            """
            <script>
                const card1Wrapper = document.getElementById('card1_wrapper');

                function setDynamicHeight() {
                    const cardHeight = card1Wrapper.scrollHeight;
                    card1Wrapper.style.height = cardHeight + 'px';
                }

                // Call the function when the page loads
                setDynamicHeight();

                // Call the function whenever the content inside the card changes
                card1Wrapper.addEventListener('DOMSubtreeModified', setDynamicHeight);
            </script>
            """,
            unsafe_allow_html=True
        )

        def style_custom_metric_cards(
            background_color: str = "#E8E8E8",
            border_size_px: int = 1,
            border_color: str = "#CCC",
            border_radius_px: int = 5,
            border_left_color: str = "#ffcc66",
            box_shadow: bool = True,
        ):
            box_shadow_str = (
                "box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;"
                if box_shadow
                else "box-shadow: none !important;"
            )
            st.markdown(
                f"""
                <style>
                    div[data-testid="custom-metric-container"] {{
                        background-color: {background_color};
                        border: {border_size_px}px solid {border_color};
                        padding: 1.5% 5% 5% 10%;
                        margin-top: -12px;
                        border-radius: {border_radius_px}px;
                        border-left: 0.5rem solid {border_left_color} !important;
                        {box_shadow_str}
                    }}
                    div[data-testid="custom-metric-container"] .metric-value {{
                        font-size: 36px;
                    }}
                    div[data-testid="custom-metric-container"] .metric-label {{
                        font-size: 18px;
                    }}
                    div[data-testid="custom-metric-container"] .metric-delta {{
                        font-size: 14px;
                        text-align: left;
                        white-space: pre-wrap;
                    }}
                </style>
                """,
                unsafe_allow_html=True,
            )

        def create_metric_card(label, value, description, centered=False):
            text_align = "center" if centered else "left"
            return f"""
            <div data-testid="custom-metric-container" style="
                display: flex; 
                flex-direction: column; 
                justify-content: center; 
                align-items: {text_align}; 
                padding: 16px; 
                margin: 8px;
                border: 1px solid #ddd; 
                border-radius: 8px; 
                background-color: #f9f9f9; 
                width: 100%; 
                height: auto;
                box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);">
                <div class="metric-label" style="
                    font-weight: bold; 
                    font-size: 16px; 
                    margin-bottom: 8px; 
                    text-align: {text_align};">{label}</div>
                <div class="metric-value" style="
                    font-weight: bold; 
                    font-size: 24px; 
                    margin-bottom: 8px; 
                    color: #333; 
                    text-align: {text_align};">{value}</div>
                <div class="custom-metric-delta" style="
                    font-weight: normal; 
                    font-size: 14px; 
                    color: #777; 
                    text-align: {text_align};">{description}</div>
            </div>
            """
        
        style_custom_metric_cards() # Adds the styling, e.g. left-hand bar on card

        def generate_analysis_card(df, varx, expln):
            """
            Generate an analysis card for a given variable, either categorical or numerical.

            Args:
                df (DataFrame): The original dataframe.
                varx (str): The variable to analyze.

            Returns:
                str: The generated HTML for the card.
            """
            if pd.api.types.is_categorical_dtype(df[varx]) or df[varx].dtype == 'object':
                # Categorical Variable: Aggregate counts
                category_data = df[varx].value_counts().reset_index()
                category_data.columns = ['Category', 'Count']

                # Get unique categories
                categories = category_data['Category'].tolist()

                # Create an interactive bar chart
                fig = px.bar(
                    category_data,
                    x='Category',
                    y='Count',
                    text='Count',
                    labels={'Category': varx, 'Count': 'Frequency'},
                    hover_data=['Count'],
                )

                # Update layout for consistent styling
                fig.update_layout(
                    xaxis=dict(
                        title=dict(
                            text=' ',
                            standoff=30
                        ),
                        tickangle=45,
                        tickfont=dict(size=10)
                    ),
                    yaxis_title=f"Count of {varx}",
                    margin=dict(l=121, r=40, t=80, b=60),
                    width=375,
                    height=275
                )
                fig.update_traces(marker_color='skyblue')

                # Dynamically create buttons for each category
                buttons = [
                    dict(
                        args=[{"x": [category_data['Category']], "y": [category_data.loc[category_data['Category'] == category, 'Count']]}],
                        label=str(category),
                        method="update"
                    )
                    for category in categories
                ]

                # Add "All Categories" button
                buttons.insert(
                    0,
                    dict(
                        args=[{"x": [category_data['Category']], "y": [category_data['Count']]}],
                        label="All Categories",
                        method="update"
                    )
                )

                # Add dropdown for filtering
                fig.update_layout(
                    updatemenus=[
                        dict(
                            buttons=buttons,
                            direction="down",
                            showactive=True,
                            x=1.3,
                            y=1.3
                        )
                    ]
                )

                # Generate HTML for the interactive Plotly chart
                plotly_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": True})

                # Card Details for Interactive Graph
                card_title = f"Interactive Analysis of {varx}"
                button1_text = "Analysis"
                button2_text = "Recommendations"
                button3_text = "Clear"
                card_text = f"This interactive chart visualizes the distribution of {varx}. Analyze patterns and trends dynamically."
                expln_text = expln

                # Use the interactive card function
                card_html = generate_card_with_overlay_interactive(
                    plotly_html,
                    button1_text,
                    button2_text,
                    button3_text,
                    card_text,
                    expln_text,
                    card_title
                )

            else:
                # Numerical Variable: Create a static graph
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(df[varx], bins=20, color='lightgreen', alpha=0.7)
                ax.set_title(f"Distribution of {varx}")
                ax.set_xlabel(varx)
                ax.set_ylabel("Frequency")

                # Save the static graph to a buffer and encode it as base64
                buffer = BytesIO()
                fig.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close()

                # Card Details for Static Graph
                card_title = f"Static Analysis of {varx}"
                button1_text = "Analysis"
                button2_text = "Recommendations"
                button3_text = "Clear"
                card_text = f"This static chart visualizes the distribution of {varx}. Analyze patterns and trends at a glance."
                expln_text = expln

                # Use the static card function
                card_html = generate_card_with_overlay(
                    f"data:image/png;base64,{image_base64}",
                    button1_text,
                    button2_text,
                    button3_text,
                    card_text,
                    expln_text,
                    card_title
                )

            return card_html

        # if st.button("Run Analysis"):

        # Temporary encoding of categorical variables
        temp_df = df_filtered_by_cohort.copy()

        for col in df_filtered_by_cohort.columns:
            if pd.api.types.is_categorical_dtype(df_filtered_by_cohort[col]) or df_filtered_by_cohort[col].dtype == 'object':
                temp_df[col] = temp_df[col].astype('category').cat.codes

        columns_to_drop = [
            "Cohort",
            "Name",
            "Age",
            "Email",
            "Contact number",
            "Enterprise Name",
            "Enterprise Address",
            "Name of community or township",
            "Business model",
            "Products/services and innovations"
        ]
        temp_df = temp_df.drop(columns=columns_to_drop, errors='ignore')

        # User selects a variable
        selected_variable = st.selectbox("Select a variable for dashboarding:", temp_df.columns)

        # Perform correlation analysis
        correlations = temp_df.corr()[selected_variable].drop(selected_variable)
        top_correlated = correlations.abs().sort_values(ascending=False).head(3)

        # Display metric cards for each top correlated variable
        metriccol1, metriccol2, metriccol3 = st.columns(3)
        metric_cols = [metriccol1, metriccol2, metriccol3]  # Store the column objects in a list

        for colnum, (var, corr_value) in enumerate(top_correlated.items()):
            with metric_cols[colnum]:  # Use the column based on the current index
                metric_card = create_metric_card(
                    label=f"Correlation with {var}",
                    value=f"{corr_value:.2f}",
                    description=f"Correlation with {selected_variable}"
                )
                st.markdown(metric_card, unsafe_allow_html=True)
                add_vertical_space(3)

        # Use columns for each correlated variable
        graphcol1, graphcol2, graphcol3 = st.columns(3)

        # Handle the first correlated variable
        with graphcol1:

            varx = top_correlated.index[0]  # Use the first correlated variable

            expln = f"Recommendations and insights based on the distribution of {varx}."

            # Call the function to generate the card HTML
            card_html = generate_analysis_card(temp_df, varx, expln)

            # Render the card in Streamlit
            st.components.v1.html(f'<div id="card_Wrapper">{card_html}</div>', width=1200, height=550)

        # Handle the second correlated variable
        with graphcol2:

            # var2 = top_correlated.index[1]
            # fig2, ax2 = plt.subplots()
            # ax2.scatter(df[selected_variable], df[var2], alpha=0.5)
            # ax2.set_title(f"{selected_variable} vs {var2}")
            # ax2.set_xlabel(selected_variable)
            # ax2.set_ylabel(var2)

            # # Save plot to base64
            # buffer2 = BytesIO()
            # fig2.savefig(buffer2, format='png', bbox_inches='tight')
            # buffer2.seek(0)
            # image2_base64 = base64.b64encode(buffer2.read()).decode('utf-8')
            # plt.close()

            # # Display plot using a custom card
            # card2_html = generate_card_with_overlay(
            #     image_url=f"data:image/png;base64,{image2_base64}",
            #     button1_text="Analysis",
            #     button2_text="Recommendations",
            #     button3_text="Reset",
            #     card_text=f"Correlation value: {top_correlated[var2]:.2f}",
            #     expln_text="Provide recommendations for variable 2.",
            #     card_title=f"{var2} Correlation"
            # )
            # st.markdown(card2_html, unsafe_allow_html=True)
            
            varx = top_correlated.index[1]  # Use the second correlated variable

            expln = f"Recommendations and insights based on the distribution of {varx}."

            # Call the function to generate the card HTML
            card_html = generate_analysis_card(temp_df, varx, expln)

            # Render the card in Streamlit
            st.components.v1.html(f'<div id="card_Wrapper">{card_html}</div>', width=1200, height=550)

        # Handle the third correlated variable
        with graphcol3:

            # var3 = top_correlated.index[2]
            # fig3, ax3 = plt.subplots()
            # ax3.scatter(df[selected_variable], df[var3], alpha=0.5)
            # ax3.set_title(f"{selected_variable} vs {var3}")
            # ax3.set_xlabel(selected_variable)
            # ax3.set_ylabel(var3)

            # # Save plot to base64
            # buffer3 = BytesIO()
            # fig3.savefig(buffer3, format='png', bbox_inches='tight')
            # buffer3.seek(0)
            # image3_base64 = base64.b64encode(buffer3.read()).decode('utf-8')
            # plt.close()

            # # Display plot using a custom card
            # card3_html = generate_card_with_overlay(
            #     image_url=f"data:image/png;base64,{image3_base64}",
            #     button1_text="Analysis",
            #     button2_text="Recommendations",
            #     button3_text="Reset",
            #     card_text=f"Correlation value: {top_correlated[var3]:.2f}",
            #     expln_text="Provide recommendations for variable 3.",
            #     card_title=f"{var3} Correlation"
            # )
            # st.markdown(card3_html, unsafe_allow_html=True)

            varx = top_correlated.index[2]  # Use the third correlated variable

            expln = f"Recommendations and insights based on the distribution of {varx}."

            # Call the function to generate the card HTML
            card_html = generate_analysis_card(temp_df, varx, expln)

            # Render the card in Streamlit
            st.components.v1.html(f'<div id="card_Wrapper">{card_html}</div>', width=1200, height=550)
        
    with dbrdmap:

        # Load your Excel file (replace with the path to your file)
        file_path = 'SiAGIA Baseline Report.xlsx'
        excel_data = pd.ExcelFile(file_path)
        df_combined = excel_data.parse('Combined')

        # List of community names
        data = {
            'Community': [
                'Howick Dargle', 'Akasia', 'Bende Mutale Village', 'Kraaibosch', 
                'Janefurse', 'Groblersdal', 'Springs', 'Mamelodi', 'Vaal River City', 'Bosplaas'
            ],
            # Optional: Additional data for hover tooltips
            'Province': [
                'KwaZulu-Natal', 'Gauteng', 'Limpopo', 'Western Cape',
                'Limpopo', 'Mpumalanga', 'Gauteng', 'Gauteng', 'Gauteng', 'North West'
            ],
            'Socioeconomic_Index': [
                0.8, 0.6, 0.4, 0.7, 0.5, 0.55, 0.65, 0.7, 0.6, 0.45
            ]
        }
        df = pd.DataFrame(data)

        # Function to get latitude and longitude using OpenCage API or fallback to predefined data
        def get_coordinates_opencage(community):
            api_key = "79cb6573ea98435189f05d6e7737d8db"  # Replace with your OpenCage API key
            try:
                response = requests.get(
                    f"https://api.opencagedata.com/geocode/v1/json?q={community}&key={api_key}"
                )
                if response.status_code == 200:
                    results = response.json()
                    if results['results']:
                        geometry = results['results'][0]['geometry']
                        return pd.Series([geometry['lat'], geometry['lng']])
                return pd.Series([None, None])  # No results
            except Exception as e:
                print(f"Error retrieving location for {community}: {e}")
                return pd.Series([None, None])

        # Uncomment if you have static coordinates for fallback
        # predefined_coordinates = {
        #     'Howick Dargle': (-29.5, 30.2),
        #     'Akasia': (-25.8, 28.1),
        #     ...
        # }

        # Apply the geocoding function
        df[['Latitude', 'Longitude']] = df['Community'].apply(get_coordinates_opencage)

        # Remove rows with missing coordinates
        df = df.dropna(subset=['Latitude', 'Longitude'])

        # Convert DataFrame to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude'])
        )

        # Create an interactive Plotly map with hover functionality
        fig = px.scatter_mapbox(
            gdf,
            lat='Latitude',
            lon='Longitude',
            hover_name='Community',
            hover_data={
                'Province': True,
                'Socioeconomic_Index': True
            },
            zoom=5,
            center={"lat": gdf['Latitude'].mean(), "lon": gdf['Longitude'].mean()},
            mapbox_style="carto-positron"
        )

        # Render the map in Streamlit
        st.title("Community Map")
        st.plotly_chart(fig)
