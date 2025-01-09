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
import tkinter
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
from bioinfokit.analys import stat
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
import pandas_profiling
from pandas_profiling import ProfileReport
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

#add an import to Hydralit
from hydralit import HydraHeadApp
from hydralit import HydraApp

#create a wrapper class
class indalodashboardshome(HydraHeadApp):

#wrap all your code in this method and you should be done

    def run(self):

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
        df_loaded = dataingestion.readdata(init)

        vars = list(df_loaded.columns.values.tolist())

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

        df=df_loaded

        st.title("Indalo analytics")

        dbrddbrd, dbrdmetrics, dbrdanalytics, dbrdpyg = st.tabs([":red[Dashboard]", ":red[Metrics]", ":red[Analytics]", ":red[Custom visualizations]"])

        with dbrddbrd:

            # Define the iframe HTML code
            iframe_code = """
            <iframe title="KPI Dashboard" width="1600" height="1000" src="https://app.powerbi.com/reportEmbed?reportId=5a1b43d3-cf89-4bf6-9fe3-0381b1ed5dd9&autoAuth=true&ctid=cbaaaed8-dfec-4c03-b204-7e7911e2049b" frameborder="0" allowFullScreen="true"></iframe>
            """

            # Use components.html to embed the iframe
            components.html(iframe_code, height=1000, width=1600)

            add_vertical_space(10)

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
            
            colm1, colm2, colm3 = st.columns(3)

            with colm1:
                # 1. Sales per customer distribution analysis
                plt.figure(figsize=(10, 6))
                plt.hist(df['Sales per customer'], bins=30, color='skyblue', edgecolor='black')
                plt.title('Distribution of Sales per Customer')
                plt.xlabel('Sales per Customer')
                plt.ylabel('Frequency')
                plt.grid(True)

                # Save the image
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                image_data = buffer.getvalue()
                encoded_data = base64.b64encode(image_data).decode('utf-8')
                image_url1 = f'data:image/png;base64,{encoded_data}'
                card_title1 = "Distribution of Sales per Customer"
                button1_text1 = "Analysis"
                button2_text1 = "Recommendations"
                button3_text1 = "Clear"
                card_text1 = "This graph shows the distribution of sales per customer."
                expln_text1 = "Analyze the spread and frequency of sales values to understand customer purchasing patterns."
                html_code1 = generate_card_with_overlay(image_url1, button1_text1, button2_text1, button3_text1, card_text1, expln_text1, card_title1)
                from streamlit.components.v1 import html
                html(f'<div id="card1_Wrapper">{html_code1}</div>', width=1200, height=550)

            with colm2:
                # 2. Late delivery risk analysis
                late_delivery_risk_counts = df['Late_delivery_risk'].value_counts()

                plt.figure(figsize=(10, 6))
                late_delivery_risk_counts.plot(kind='bar', color='salmon', edgecolor='black')
                plt.title('Late Delivery Risk Analysis')
                plt.xlabel('Late Delivery Risk')
                plt.ylabel('Count')
                plt.grid(True)

                # Save the image
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                image_data = buffer.getvalue()
                encoded_data = base64.b64encode(image_data).decode('utf-8')
                image_url2 = f'data:image/png;base64,{encoded_data}'
                card_title2 = "Late Delivery Risk Analysis"
                button1_text2 = "Analysis"
                button2_text2 = "Recommendations"
                button3_text2 = "Clear"
                card_text2 = "This graph shows the count of late delivery risks."
                expln_text2 = "Monitor and reduce late delivery risks to improve customer satisfaction."
                html_code2 = generate_card_with_overlay(image_url2, button1_text2, button2_text2, button3_text2, card_text2, expln_text2, card_title2)
                from streamlit.components.v1 import html
                html(f'<div id="card2_Wrapper">{html_code2}</div>', width=1200, height=550)

            with colm3:

                # 3. Category-wise sales analysis
                category_sales = df.groupby('Category Name')['Sales per customer'].sum().sort_values(ascending=False)

                plt.figure(figsize=(12, 8))
                category_sales.plot(kind='bar', color='lightgreen', edgecolor='black')
                plt.title('Total Sales per Category')
                plt.xlabel('Category Name')
                plt.ylabel('Total Sales')
                plt.grid(True)
                plt.xticks(rotation=45, ha='right')

                # Save the image
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                image_data = buffer.getvalue()
                encoded_data = base64.b64encode(image_data).decode('utf-8')
                image_url3 = f'data:image/png;base64,{encoded_data}'
                card_title3 = "Total Sales per Category"
                button1_text3 = "Analysis"
                button2_text3 = "Recommendations"
                button3_text3 = "Clear"
                card_text3 = "This graph shows the total sales per category."
                expln_text3 = "Identify the best performing categories and focus on them to maximize sales."
                html_code3 = generate_card_with_overlay(image_url3, button1_text3, button2_text3, button3_text3, card_text3, expln_text3, card_title3)
                from streamlit.components.v1 import html
                html(f'<div id="card3_Wrapper">{html_code3}</div>', width=1200, height=550)

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
                                    font-size: 14px;
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

        with dbrdmetrics:

                style_custom_metric_cards()

                # Insight 1: Total Orders and On-time Delivery Rate
                total_orders = df.shape[0]
                on_time_deliveries = df[df['On Time In Full'] == 1].shape[0]
                on_time_delivery_rate = (on_time_deliveries / total_orders) * 100

                # Insight 2: Average Sales per Customer
                average_sales_per_customer = df['Sales per customer'].mean()

                # Insight 3: Total Benefit and Average Benefit per Order
                total_benefit = df['Benefit per order'].sum()
                average_benefit_per_order = df['Benefit per order'].mean()

                col1, col2, col3 = st.columns(3)

                with col1:

                    # Metric Card 1: Total Orders and On-time Delivery Rate
                    label1 = 'Total Orders and On-time Delivery Rate'
                    value1 = f"{total_orders} total orders"
                    delta1_1 = f"On-time deliveries: {on_time_deliveries}"
                    delta1_2 = f"On-time delivery rate: {on_time_delivery_rate:.2f}%"
                    delta1_3 = ""

                    # Display the metric card

                    # Metric Card 1: Total Orders and On-time Delivery Rate
                    st.markdown(
                        f"""
                        <div data-testid="custom-metric-container">
                            <div class="metric-label">{label1}</div>
                            <div class="metric-value">{value1}</div>
                            <div class="custom-metric-delta">{delta1_1}</div>
                            <div class="custom-metric-delta">{delta1_2}</div>
                            <div class="custom-metric-delta">{delta1_3}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:

                    # Metric Card 2: Average Sales per Customer
                    label2 = 'Average Sales per Customer'
                    value2 = f"R{average_sales_per_customer:.2f} average sales"
                    delta2_1 = ""
                    delta2_2 = ""
                    delta2_3 = ""

                    # Metric Card 2: Average Sales per Customer
                    st.markdown(
                        f"""
                        <div data-testid="custom-metric-container">
                            <div class="metric-label">{label2}</div>
                            <div class="metric-value">{value2}</div>
                            <div class="custom-metric-delta">{delta2_1}</div>
                            <div class="custom-metric-delta">{delta2_2}</div>
                            <div class="custom-metric-delta">{delta2_3}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col3:

                    # Metric Card 3: Total Benefit and Average Benefit per Order
                    label3 = 'Total Benefit and Average Benefit per Order'
                    value3 = f"R{total_benefit:.2f} total benefit"
                    delta3_1 = f"Average benefit per order: R{average_benefit_per_order:.2f}"
                    delta3_2 = ""
                    delta3_3 = ""

                    # Metric Card 3: Total Benefit and Average Benefit per Order
                    st.markdown(
                        f"""
                        <div data-testid="custom-metric-container">
                            <div class="metric-label">{label3}</div>
                            <div class="metric-value">{value3}</div>
                            <div class="custom-metric-delta">{delta3_1}</div>
                            <div class="custom-metric-delta">{delta3_2}</div>
                            <div class="custom-metric-delta">{delta3_3}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        with dbrdpyg:
             
            st.title("Custom analytics")

            df_tableau = df.copy()

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
            
            import seaborn as sns
            import ppscore as pps
            import statsmodels.api as sm

            # Step 1: Handling NaN and infinite values
            # Replace infinite values with NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Drop columns with too many NaN values
            threshold = 0.5  # Drop columns with more than 50% missing values
            # df.dropna(thresh=int(threshold * len(df)), axis=1, inplace=True)

            # Fill remaining NaN values with the median of the column
            df.fillna(df.median(), inplace=True)

            # Step 2: Correlation Analysis for Sales and Profit
            correlation_matrix = df.corr()
            sales_correlation = correlation_matrix['Sales per customer'].drop('Sales per customer').sort_values(ascending=False)
            profit_correlation = correlation_matrix['Benefit per order'].drop('Benefit per order').sort_values(ascending=False)

            # Step 3: PPS Analysis
            pps_sales = pps.predictors(df, 'Sales per customer').sort_values(by='ppscore', ascending=False)
            pps_profit = pps.predictors(df, 'Benefit per order').sort_values(by='ppscore', ascending=False)

            # Step 4: Regression Analysis
            # Selecting top correlated variables for regression
            top_sales_predictors = sales_correlation.index[:5].tolist()
            top_profit_predictors = profit_correlation.index[:5].tolist()

            # Sales Regression
            X_sales = df[top_sales_predictors]
            y_sales = df['Sales per customer']
            X_sales = sm.add_constant(X_sales)  # adding a constant
            model_sales = sm.OLS(y_sales, X_sales).fit()

            # Profit Regression
            X_profit = df[top_profit_predictors]
            y_profit = df['Benefit per order']
            X_profit = sm.add_constant(X_profit)  # adding a constant
            model_profit = sm.OLS(y_profit, X_profit).fit()

            # Streamlit Code
            st.title('Sales and Profit Analysis')

            # Correlation Analysis
            st.header('1. Correlation Analysis')
            st.markdown("### Top Correlations for Sales and Profit")

            top_sales_corr = sales_correlation.index[0]
            top_profit_corr = profit_correlation.index[0]

            # Visualizing Top Correlations
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            sns.barplot(x=sales_correlation.head(5).values, y=sales_correlation.head(5).index, ax=axes[0], palette='Blues_d')
            axes[0].set_title('Top 5 Correlations with Sales per Customer')
            axes[0].set_xlabel('Correlation Coefficient')
            axes[0].set_ylabel('')

            sns.barplot(x=profit_correlation.head(5).values, y=profit_correlation.head(5).index, ax=axes[1], palette='Reds_d')
            axes[1].set_title('Top 5 Correlations with Benefit per Order')
            axes[1].set_xlabel('Correlation Coefficient')
            axes[1].set_ylabel('')

            st.pyplot(fig)

            st.info(f"**Insight on Sales per Customer:** The variable most strongly correlated with Sales per Customer is `{top_sales_corr}` with a correlation coefficient of {sales_correlation[top_sales_corr]:.2f}. This suggests that changes in `{top_sales_corr}` are closely related to changes in sales, indicating a potential area to focus on for sales strategies.")

            st.info(f"**Insight on Benefit per Order:** The variable most strongly correlated with Benefit per Order is `{top_profit_corr}` with a correlation coefficient of {profit_correlation[top_profit_corr]:.2f}. This indicates that `{top_profit_corr}` plays a significant role in determining profitability, and efforts to optimize this variable could enhance profit margins.")

            # PPS Analysis
            st.header('2. Predictive Power Score (PPS) Analysis')
            st.markdown("### Top PPS Scores for Sales and Profit")

            top_sales_pps = pps_sales.iloc[0]['x']
            top_profit_pps = pps_profit.iloc[0]['x']

            # Visualizing Top PPS Scores
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            sns.barplot(x=pps_sales['ppscore'].head(5), y=pps_sales['x'].head(5), ax=axes[0], palette='Blues_d')
            axes[0].set_title('Top 5 PPS Scores for Sales per Customer')
            axes[0].set_xlabel('PPS Score')
            axes[0].set_ylabel('')

            sns.barplot(x=pps_profit['ppscore'].head(5), y=pps_profit['x'].head(5), ax=axes[1], palette='Reds_d')
            axes[1].set_title('Top 5 PPS Scores for Benefit per Order')
            axes[1].set_xlabel('PPS Score')
            axes[1].set_ylabel('')

            st.pyplot(fig)

            st.info(f"**Predictive Insight for Sales:** The variable with the highest PPS score for predicting Sales per Customer is `{top_sales_pps}` with a score of {pps_sales.iloc[0]['ppscore']:.2f}. This means `{top_sales_pps}` is a strong predictor of sales outcomes, suggesting it should be a key focus in predictive models and strategies.")

            st.info(f"**Predictive Insight for Profit:** The variable with the highest PPS score for predicting Benefit per Order is `{top_profit_pps}` with a score of {pps_profit.iloc[0]['ppscore']:.2f}. This highlights `{top_profit_pps}` as a crucial variable in predicting profitability and should be targeted for optimization efforts.")

            # Regression Analysis
            st.header('3. Regression Analysis')
            st.markdown("### Significant Predictors from Regression Analysis")

            sales_pvalues = model_sales.pvalues[1:]  # excluding the constant
            significant_sales_vars = sales_pvalues[sales_pvalues < 0.05].index.tolist()

            profit_pvalues = model_profit.pvalues[1:]  # excluding the constant
            significant_profit_vars = profit_pvalues[profit_pvalues < 0.05].index.tolist()

            if significant_sales_vars:
                st.subheader('Sales per Customer')
                st.markdown(f"The significant predictors of Sales per Customer in the regression model are: `{', '.join(significant_sales_vars)}`.")
                st.info(f"**Regression Insight for Sales:** The variables `{', '.join(significant_sales_vars)}` are statistically significant predictors of Sales per Customer with p-values less than 0.05. This suggests that these variables have a strong impact on sales and should be considered in sales strategies.")
            else:
                st.subheader('Sales per Customer')
                st.markdown('There are no statistically significant predictors of Sales per Customer in the regression model.')
                st.info('**Regression Insight for Sales:** No variables were found to be statistically significant predictors of Sales per Customer. This suggests that other factors, not included in this analysis, may influence sales.')

            if significant_profit_vars:
                st.subheader('Benefit per Order')
                st.markdown(f"The significant predictors of Benefit per Order in the regression model are: `{', '.join(significant_profit_vars)}`.")
                st.info(f"**Regression Insight for Profit:** The variables `{', '.join(significant_profit_vars)}` are statistically significant predictors of Benefit per Order with p-values less than 0.05. This indicates these variables have a strong influence on profitability and should be prioritized for improvement to enhance profits.")
            else:
                st.subheader('Benefit per Order')
                st.markdown('There are no statistically significant predictors of Benefit per Order in the regression model.')
                st.info('**Regression Insight for Profit:** No variables were found to be statistically significant predictors of Benefit per Order. This implies that other factors, not included in this analysis, may affect profitability.')
