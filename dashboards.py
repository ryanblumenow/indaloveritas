import streamlit as st
from streamlit_option_menu import option_menu
import base64
from email import header
from html.entities import html5
from importlib.resources import read_binary
import hydralit as hy
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
# matplotlib.use('TkAgg')
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

def dashboardcode():

    st.subheader("Tiger analytics")

    with st.spinner("Analyzing dataset and generating dataset statistics"):

        my_bar = st.progress(0)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)

        start_time = time.time()

    # Helper function for section headers
    def section_header(header_text, header_size=2):
        header_tag = f"h{header_size}"
        centered_style = 'text-align: center;'
        st.markdown(f"<{header_tag} style='{centered_style}'>{header_text}</{header_tag}>", unsafe_allow_html=True)

    # Apply CSS styling to create a rounded block format #3182A8
    image_block_style = """
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px;
        border-radius: 10px;
        background-color: #E8E8E8;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px; /* Add margin bottom to create space below the block */
    """

    def style_custom_metric_cards(
        background_color: str = "#E8E8E8",
        border_size_px: int = 1,
        border_color: str = "#CCC",
        border_radius_px: int = 5,
        border_left_color: str = "#207DCE",
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

    demogoverview = st.expander("Demographics overview", expanded=True)

    with demogoverview:

        # Initial dashboard EDA

        # Working but unused for now

        # # Subset the dataframe with the desired columns
        # df1 = df[['individualclientid', 'age', 'TenantValue_gender']]
        # g = sns.pairplot(df1, hue='TenantValue_gender')

        # # Save the pairplot to a buffer
        # buf = BytesIO()
        # plt.savefig(buf, format="png")
        # buf.seek(0)
        # # Convert the saved image to base64
        # data = base64.b64encode(buf.read()).decode("utf-8")
        # data = "data:image/png;base64," + data
        
        # df2 = df[['individualclientid', 'age', 'TenantValue_provinces']]
        # sns.pairplot(df2, hue='TenantValue_provinces')

        # df3 = df[['individualclientid', 'TenantValue_provinces', 'TenantValue_gender']]
        # sns.pairplot(df3, hue='TenantValue_gender')

        section_header("Demographics overview", header_size=3)

        # Subset the dataframe with the desired columns
        df1 = df[['individualclientid', 'age', 'age_group', 'TenantValue_gender', 'ProvinceCode', 'TenantValue_maritalstatus', 'TenantValue_maritalstatus_encoded']]
        # Cluster data based on demographic groups
        # Convert categorical variables to numerical values using one-hot encoding
        # Perform one-hot encoding with prefix to preserve original field
        df_encoded = pd.get_dummies(df['TenantValue_maritalstatus'], prefix='TenantValue_maritalstatus')
        # Concatenate the encoded columns with the original DataFrame
        df = pd.concat([df, df_encoded], axis=1)
        X = df[['age', 'TenantValue_maritalstatus_Single', 'TenantValue_maritalstatus_Married', 'TenantValue_maritalstatus_Separated', 'TenantValue_maritalstatus_Divorced']]
        # Create an imputer object with mean strategy
        imputer = SimpleImputer(strategy='mean')
        # Impute missing values in the X DataFrame
        X = imputer.fit_transform(X)
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=10)
        kmeans.fit(X)
        df['cluster'] = kmeans.labels_

        # Age and gender

        # Set the plot style
        sns.set_style('whitegrid')

        # Count of gender plot

        plt.figure(figsize=(5, 3))
        sns.countplot(data=df1, x='TenantValue_gender', palette='hls')
        plt.xlabel('Gender', fontsize=8)
        plt.ylabel('Count', fontsize=8)
        plt.title('Distribution of Gender', fontsize=21)
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        data = base64.b64encode(buffer.read()).decode('utf-8')
        data = 'data:image/png;base64,' + data

        # Count of ages plot

        plt.figure(figsize=(5, 3))
        sns.countplot(data=df1, x='age_group', palette='hls')
        plt.xlabel('Gender', fontsize=8)
        plt.xticks(rotation='vertical')
        plt.ylabel('Count', fontsize=8)
        plt.title('Distribution of Age Group', fontsize=21)
        buffer2 = BytesIO()
        plt.savefig(buffer2, format='png')
        buffer2.seek(0)
        data2 = base64.b64encode(buffer2.read()).decode('utf-8')
        data2 = 'data:image/png;base64,' + data2

        # Plot of genders by age

        # Compute the cross-tabulation of age group and gender
        cross_tab = pd.crosstab(df1['age_group'], df1['TenantValue_gender'])
        # Normalize the counts to get proportions
        normalized_cross_tab = cross_tab.div(cross_tab.sum(axis=1), axis=0)
        plt.figure(figsize=(8, 5))
        # Plot the stacked bar plot
        normalized_cross_tab.plot(kind='bar', stacked=True)
        plt.xlabel('Age Group', fontsize=8)
        plt.ylabel('Proportion', fontsize=8)
        plt.title('Distribution of Gender by Age Group', fontsize=21)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        # plt.yticks([0.4 * i for i in range(6)])  # Set y-axis ticks to 0.4 intervals
        # plt.ylim(0, 1)  # Set the upper limit of the y-axis to 1
        buffer3 = BytesIO()
        plt.savefig(buffer3, format='png')
        buffer3.seek(0)
        data3 = base64.b64encode(buffer3.read()).decode('utf-8')
        data3 = 'data:image/png;base64,' + data3
        
        # Age and provinces

        # # Count of observations by province
        # plt.figure(figsize=(8, 5))
        # sns.countplot(data=df, x='TenantValue_provinces', palette='hls')
        # plt.xlabel('Province', fontsize=12)
        # plt.ylabel('Count', fontsize=12)
        # plt.title('Distribution of Provinces', fontsize=14)
        # plt.xticks(fontsize=10, rotation=90)
        # plt.yticks(fontsize=10)
        # buffer4 = BytesIO()
        # plt.savefig(buffer4, format='png')
        # buffer4.seek(0)
        # data4 = base64.b64encode(buffer4.read()).decode('utf-8')
        # data4 = 'data:image/png;base64,' + data4

        # # Split of age groups per province
        # plt.figure(figsize=(12, 6))
        # sns.countplot(data=df, x='TenantValue_provinces', hue='age_group', palette='hls')
        # plt.xlabel('Province', fontsize=12)
        # plt.ylabel('Count', fontsize=12)
        # plt.title('Distribution of Age Groups by Province', fontsize=14)
        # plt.xticks(fontsize=10, rotation=90)
        # plt.yticks(fontsize=10)
        # plt.legend(title='Age Group', fontsize=10)
        # buffer5 = BytesIO()
        # plt.savefig(buffer5, format='png')
        # buffer5.seek(0)
        # data5 = base64.b64encode(buffer5.read()).decode('utf-8')
        # data5 = 'data:image/png;base64,' + data5

        # Filter out empty values in 'TenantValue_provinces' column
        df_filtered = df1[df1['ProvinceCode'].notna()]
        # Count of observations by province
        plt.figure(figsize=(5, 3))
        sns.countplot(data=df_filtered, x='ProvinceCode', palette='hls')
        plt.xlabel('Province', fontsize=8)
        plt.ylabel('Count', fontsize=8)
        plt.title('Distribution of Location', fontsize=21)
        plt.xticks(fontsize=8, rotation=90)
        plt.yticks(fontsize=8)
        buffer4 = BytesIO()
        plt.savefig(buffer4, format='png')
        buffer4.seek(0)
        data4 = base64.b64encode(buffer4.read()).decode('utf-8')
        data4 = 'data:image/png;base64,' + data4

        # Split of age groups per province
        plt.figure(figsize=(5, 4.75))
        sns.countplot(data=df_filtered, x='ProvinceCode', hue='age_group', palette='hls')
        plt.xlabel('Location', fontsize=8)
        plt.ylabel('Count', fontsize=8)
        plt.title('Distribution of Age Groups by Location', fontsize=21)
        plt.xticks(fontsize=8) # , rotation=90
        plt.yticks(fontsize=8)
        plt.legend(title='Age Group', fontsize=6, bbox_to_anchor=(0.9,0.14))
        buffer5 = BytesIO()
        plt.savefig(buffer5, format='png')
        buffer5.seek(0)
        data5 = base64.b64encode(buffer5.read()).decode('utf-8')
        data5 = 'data:image/png;base64,' + data5

        # Marital status and age

        # Visualize count of marital status using a bar chart
        marital_status_counts = df1['TenantValue_maritalstatus_encoded'].value_counts()
        # Create the bar plot
        plt.figure(figsize=(5, 3))
        sns.barplot(x=marital_status_counts.index, y=marital_status_counts.values)
        plt.xlabel('Marital Status', fontsize=8)
        plt.ylabel('Count', fontsize=8)
        plt.title('Count of Marital Status', fontsize=21)
        plt.xticks(rotation=45)
        buffer6 = BytesIO()
        plt.savefig(buffer6, format='png')
        buffer6.seek(0)
        data6 = base64.b64encode(buffer6.getvalue()).decode('utf-8')
        data6 = 'data:image/png;base64,' + data6

        # Visualize marital status per age group

        # # Group the data by age group and marital status
        # df['age_group'] = pd.cut(df['age'], bins=np.arange(0, df1['age'].max() + 5, 5))
        marital_status_age_group_counts = df1.groupby(['age_group', 'TenantValue_maritalstatus_encoded']).size().unstack()
        # Create the bar plot
        plt.figure(figsize=(5, 3))
        marital_status_age_group_counts.plot(kind='bar', stacked=True)
        plt.xlabel('Age Group', fontsize=8)
        plt.ylabel('Count', fontsize=8)
        plt.title('Marital Status by Age Group', fontsize=21)
        buffer7 = BytesIO()
        plt.savefig(buffer7, format='png')
        buffer7.seek(0)
        data7 = base64.b64encode(buffer7.getvalue()).decode('utf-8')
        data7 = 'data:image/png;base64,' + data7

        df2 = df[['age', 'age_group', 'marketvalue', 'totalholdingvalue', 'sum_asset_values', 'TenantValue_maritalstatus_Single', 'TenantValue_maritalstatus_Married', 'TenantValue_maritalstatus_Separated', 'TenantValue_maritalstatus_Divorced', 'cluster']]

        # def plot_holdings_by_cluster():
        sns.catplot(x="cluster", y="sum_asset_values", kind="bar", data=df2, height=3, aspect=1)
        sns.set(style="whitegrid")
        buffer8 = BytesIO()
        plt.savefig(buffer8, format='png')
        buffer8.seek(0)
        data8 = base64.b64encode(buffer8.getvalue()).decode('utf-8')
        data8 = 'data:image/png;base64,' + data8

        # def plot_holdings_by_age():
        sns.catplot(x="age", y="sum_asset_values", kind="bar", data=df2, height=3, aspect=2)
        plt.xticks(rotation='vertical')
        buffer9 = BytesIO()
        plt.savefig(buffer9, format='png')
        buffer9.seek(0)
        data9 = base64.b64encode(buffer9.getvalue()).decode('utf-8')
        data9 = 'data:image/png;base64,' + data9

        # Display the plots using Streamlit
        coln1, coln2, coln3 = st.columns(3)
        col1, col2, col3, col4 = st.columns(4)
        colm1, colm2 = st.columns(2)

        with col1:
            # Create a Markdown block with the desired styling and display the image
            st.markdown(
                f'<div style="{image_block_style}">'
                f'<img src=' + data + ' style="width:300px;height:173px;">'
                '</div>',
                unsafe_allow_html=True
            )

        with col2:
            # Create a Markdown block with the desired styling and display the image
            st.markdown(
                f'<div style="{image_block_style}">'
                f'<img src=' + data2 + ' style="width:300px;height:173px;">'
                '</div>',
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f'<div style="{image_block_style}">'
                f'<img src=' + data4 + ' style="width:300px;height:173px;">'
                '</div>',
                unsafe_allow_html=True
            )  

        with col4:
            st.markdown(
                f'<div style="{image_block_style}">'
                f'<img src=' + data6 + ' style="width:300px;height:173px;">'
                '</div>',
                unsafe_allow_html=True
            )  

        with coln1:
            # Create a Markdown block with the desired styling and display the image
            st.markdown(
                f'<div style="{image_block_style}">'
                f'<img src=' + data3 + ' style="width:500px;height:221px;">'
                '</div>',
                unsafe_allow_html=True
            )

        with coln2:
            # Create a Markdown block with the desired styling and display the image
            st.markdown(
                f'<div style="{image_block_style}">'
                f'<img src=' + data5 + ' style="width:500px;height:221px;">'
                '</div>',
                unsafe_allow_html=True
            )

        with coln3:
            # Create a Markdown block with the desired styling and display the image
            st.markdown(
                f'<div style="{image_block_style}">'
                f'<img src=' + data7 + ' style="width:500px;height:221px;">'
                '</div>',
                unsafe_allow_html=True
            )

        with colm1:
            # Create a Markdown block with the desired styling and display the image
            st.markdown(
                f'<div style="{image_block_style}">'
                f'<img src=' + data8 + '>'
                '</div>',
                unsafe_allow_html=True
            )

        with colm2:
            # Create a Markdown block with the desired styling and display the image
            st.markdown(
                f'<div style="{image_block_style}">'
                f'<img src=' + data9 + '>'
                '</div>',
                unsafe_allow_html=True
            )

    headlinemetrics = st.expander("Headline metrics", expanded=True)

    with headlinemetrics:
        
        # # Section 1: Count of Records

        section_header("Count of Usable Records", header_size=3)
        style_metric_cards(background_color='#E8E8E8', border_left_color='#207DCE', box_shadow=True)
        col1, col2, col3 = st.columns(3)
        record_count = len(df)
        formatted_count = f"{record_count:,.0f}"
        col2.metric(label="Count of Usable Records", value=formatted_count)

        # Section 2: Demographic Distribution

        section_header("Demographic Distribution", header_size=3)
        col1, col2, col3 = st.columns(3)
        gender_distribution_male = (df['TenantValue_gender'].value_counts()[0])
        gender_distribution_female = (df['TenantValue_gender'].value_counts()[1])
        total = gender_distribution_male + gender_distribution_female
        marital_status_distribution = df['TenantValue_maritalstatus_encoded'].value_counts()
        col1.metric(label="Gender Distribution %", value="Males: " + str(round(gender_distribution_male*100/(total), 2)) + "%", delta="-Females: " + str(round(gender_distribution_female*100/(total), 2)) + "%", delta_color = "normal")
        
        # Dataframe option
        # Create a copy of the column with the desired name
        # gender_distribution_with_name = gender_distribution.rename('Gender')
        # marital_status_distribution_with_name = marital_status_distribution.rename('Marital status')
        # col1.dataframe(gender_distribution_with_name)
        # col2.dataframe(marital_status_distribution_with_name)

        marital_status_distribution = df['TenantValue_maritalstatus_encoded'].value_counts()

        # Calculate the total count
        total = marital_status_distribution.sum()
        # Calculate the percentage distribution for each category
        percentage_distribution = marital_status_distribution * 100 / total
        # Get the top marital status and its percentage
        top_marital_status = marital_status_distribution.index[0]
        top_percentage = percentage_distribution[0]
        # Concatenate the other categories and their percentages
        other_statuses = marital_status_distribution.index[1:]
        other_percentages = percentage_distribution[1:]
        delta = " - ".join([f"{status.capitalize()}: {percentage:.2f}%" for status, percentage in zip(other_statuses, other_percentages)])
        # Display the metric for the top marital status and delta for other categories
        # Regular metric card
        # col3.metric(label='Marital status shares', value=f"{top_marital_status.capitalize()}: {top_percentage:.2f}%", delta=delta, delta_color="off")
        
        # Custom metric card with multi-line delta
        
        with col3:                        

            # Apply custom styles to custom metric cards
            style_custom_metric_cards()

            # Define the values for the custom card
            label = 'Marital status shares'
            value = f"{top_marital_status.capitalize()}: {top_percentage:.2f}%"
            delta = delta

            # Display the custom metric card with custom content
            st.markdown(
                f"""
                <div data-testid="custom-metric-container">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="custom-metric-delta">{delta}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Section 3: Age Distribution

        # section_header("Age Distribution", header_size=3)
        # col3.pyplot(df['age'].hist())
        # fig, ax = plt.subplots()
        # df['age'].hist(ax=ax)
        # fig.set_size_inches((1,1))
        # ax.set_xlabel('Age', fontsize=8)
        # ax.set_ylabel('Count', fontsize=8)
        # st.pyplot(fig)
        average_age = round(df['age'].mean(), 2)
        col2.metric(label='Average age', value=average_age)

        # Section 4: Nationality and Language

        section_header("Nationality and Language")
        col1, col2, col3 = st.columns(3)
        nationality_distribution = df['TenantValue_countries'].value_counts()
        nationalitypct1 = round(nationality_distribution[0]*100/nationality_distribution.sum(), 2)
        nationalitypct2 = round(nationality_distribution[1]*100/nationality_distribution.sum(), 2)
        province_distribution = df['ProvinceCode'].value_counts()                   
        language_distribution = df['TenantValue_language'].value_counts()
        langpct1 = round(language_distribution[0]*100/language_distribution.sum(), 2)
        langpct2 = round(language_distribution[1]*100/language_distribution.sum(), 2)
        col1.metric(label="Top Nationality", value=str(nationality_distribution.index[0]) + " " + str(nationalitypct1) + "%", delta=str(nationality_distribution.index[1]) + " " + str(nationalitypct2) + "%")
        col2.metric(label="Top Location", value=province_distribution.index[0])
        col3.metric(label="Top Language", value=str(language_distribution.index[0]) + " " + str(langpct1) + "%", delta=str(language_distribution.index[1]) + " " + str(langpct2) +"%")

        # Section 5: Financial Metrics

        section_header("Financial Metrics")
        col1, col2, col3 = st.columns(3)
        # Remove extreme outliers
        # Calculate the mean and standard deviation of the 'debtamount' column
        meandebt = float(df['debtamount'].mean())
        std_dev_debt = float(df['debtamount'].std())
        meaninc = float(df['amount_x'].mean())
        std_dev_inc = float(df['amount_x'].std())
        # Define the number of standard deviations for outlier threshold
        num_std_devs = 3  # adjust as needed
        # Calculate the upper and lower bounds for outlier removal
        lower_bound_debt = meandebt - num_std_devs * std_dev_debt
        upper_bound_debt = meandebt + num_std_devs * std_dev_debt
        lower_bound_inc = meaninc - num_std_devs * std_dev_inc
        upper_bound_inc = meaninc + num_std_devs * std_dev_inc
        # Filter out rows with values outside the bounds
        filtered_df_debt = df[(df['debtamount'] >= lower_bound_debt) & (df['debtamount'] <= upper_bound_debt)]
        filtered_df_inc = df[(df['amount_x'] >= lower_bound_inc) & (df['amount_x'] <= upper_bound_inc)]
        # total_holding_value = "{:,.2f}".format(df['totalholdingvalue'].sum())
        filtered_df_hldg = df.drop_duplicates(subset='holdingid').groupby('holdingid').agg({'totalholdingvalue': 'sum'})
        total_holding_value = "{:,.2f}".format(filtered_df_hldg['totalholdingvalue'].sum())
        # sum_asset_values = "{:,.2f}".format(df['sum_asset_values'].sum())
        filtered_df_assets = df.drop_duplicates(subset='individualassetid').groupby('individualassetid').agg({'sum_asset_values': 'sum'})
        sum_asset_values = "{:,.2f}".format(filtered_df_assets['sum_asset_values'].sum())
        # annual_income = df['amount_x'].mean()
        filtered_df_inc = filtered_df_inc.drop_duplicates(subset='individualincomeid').groupby('individualincomeid').agg({'amount_x': 'sum'})
        annual_income = filtered_df_inc['amount_x'].sum()
        # total_debt = df['debtamount'].mean()
        filtered_df_debt = filtered_df_debt.drop_duplicates(subset='individualliabilityid').groupby('individualliabilityid').agg({'debtamount': 'sum'})
        total_debt = filtered_df_debt['debtamount'].sum()
        print(annual_income)
        print(total_debt)
        col1.metric(label="Total Holding Value", value=total_holding_value)
        col2.metric(label="Total Asset Value", value=sum_asset_values)
        debt_to_holding_ratio = round(filtered_df_debt['debtamount'].sum() / filtered_df_hldg['totalholdingvalue'].sum() * 100, 2)
        col3.metric(label="Debt to Holding Ratio", value=debt_to_holding_ratio)

        # # Section 6: Geographical Analysis
        # section_header("Geographical Analysis")
        # country_distribution = df['TenantValue_countries'].value_counts()
        # province_distribution = df['ProvinceCode'].value_counts()
        # region_distribution = df['RegionCode'].value_counts()[0]
        # print(region_distribution)
        # col1.metric(label="Top Country", value=country_distribution.index[0])
        # col2.metric(label="Top Location", value=province_distribution.index[0])
        # col3.metric(label="Top Region", value=region_distribution.index[0])

        # Extract unique branch names
        branches = set()
        for paths in df['encoded_paths']:
            branches.update(paths)

        # Section 7: Relationship Metrics
        # section_header("Relationship Metrics")
        # Drop duplicates based on relevant columns
        df_unique = df.drop_duplicates(subset=['individualclientid'])
        # Convert 'isclient' column to string type
        df_unique['isclient'] = df_unique['isclient'].astype(str)
        # Clean the 'isclient' column values (if necessary)
        df_unique['isclient'] = df_unique['isclient'].str.strip().str.upper()
        # Calculate is client count
        isclient_count = (df_unique['isclient'] == "TRUE").sum()
        # Calculate active individuals count
        active_individuals_count = df_unique['Active'].sum()
        # Calculate active individuals percentage
        active_individuals_percentage = str(round(active_individuals_count / len(df_unique) * 100, 2)) + "%"
        # Display the metrics
        col1.metric(label="Is Client Count", value=isclient_count)
        col2.metric(label="Active Individuals Count", value=active_individuals_count)
        col3.metric(label="Active Individuals Percentage", value=active_individuals_percentage)

        # Section 8: Additional Metrics
        # section_header("Additional Metrics")
        # user_role_distribution = df['UserRole_ID'].value_counts()
        retirement_start_date_stats = df[df['retirementstartdate'].notnull()]['retirementstartdate']
        # print(user_role_distribution)
        # col1.metric(label="User Role Distribution", value=user_role_distribution.index[0])
        # col2.metric(label="Retirement Start Date Statistics", value=retirement_start_date_stats.describe())
        retirement_start_date_stats = df['retirementstartdate'].describe()
        top_retirement_start_date = retirement_start_date_stats['top']
        col1.metric("Top Retirement Start Date:", top_retirement_start_date)

        # # Section 9: Gender Ratio
        # section_header("Gender Ratio")
        # gender_ratio = df['TenantValue_gender'].value_counts(normalize=True) * 100
        # col1.metric(label="Gender Ratio", value=gender_ratio)

        # # Section 10: Marital Status Ratio
        # section_header("Marital Status Ratio")
        # marital_status_ratio = df['TenantValue_maritalstatus'].value_counts(normalize=True) * 100
        # col1.metric(label="Marital Status Ratio", value=marital_status_ratio)

        # # Section 11: Average Age
        # section_header("Average Age")
        # average_age = df['age'].mean()
        # col1.metric(label="Average Age", value=average_age)

        # Section 12: Average Market Value
        # section_header("Average Market Value")

        # Section 13: Average Annual Income
        # section_header("Average Annual Income")
        average_annual_income = "{:,.2f}".format(round(filtered_df_inc['amount_x'].mean(), 2))
        average_debt = "{:,.2f}".format(round(filtered_df_debt['debtamount'].mean(), 2))
        col1.metric(label="Average Annual Income", value=average_annual_income)
        col2.metric(label='Average Debt', value=average_debt)
        col3.metric(label="Average Income/Debt Ratio", value="{:,.2f}".format((annual_income/total_debt)))

        average_market_value = "{:,.2f}".format(round(df['marketvalue'].mean(), 2))
        col2.metric(label="Average Market Value", value=average_market_value)

        # Count observations per category in the "type" column
        type_counts = df['type'].value_counts()
        capital_count = str(type_counts['Capital'])
        income_count = str(type_counts['Income'])
        col3.metric(label='Count of needs by category', value='Capital needs: ' + capital_count, delta='Income needs: ' + income_count, delta_color='off')

        # Section 14: Total Debt to Total Holding Value Ratio
        # section_header("Total Debt to Total Holding Value Ratio")


        # # Section 15: Nationality Distribution
        # section_header("Nationality Distribution")
        # nationality_distribution = df['TenantValue_countries'].value_counts()
        # col1.metric(label="Nationality Distribution", value=nationality_distribution)

        # # Section 16: Language Distribution
        # # section_header("Language Distribution")
        # language_distribution = df['TenantValue_language'].value_counts()
        # col1.metric(label="Language Distribution", value=language_distribution)

        # Section 17: Total Holding Value by Nationality
        # section_header("Total Holding Value by Nationality")
        # Calculate total holding value by nationality

        total_asset_value_by_gender = df.groupby('TenantValue_gender')['sum_asset_values'].sum()
        total_asset_value_by_gender = total_asset_value_by_gender.reset_index()
        total_asset_value_by_gender = total_asset_value_by_gender.rename(columns={'TenantValue_gender': 'Gender', 'sum_asset_values': 'Total Asset Value'})
        col1.dataframe(data=total_asset_value_by_gender)

        total_asset_value_by_nationality = df.groupby('TenantValue_countries')['sum_asset_values'].sum()
        total_asset_value_by_nationality = total_asset_value_by_nationality.reset_index()
        total_asset_value_by_nationality = total_asset_value_by_nationality.rename(columns={'TenantValue_countries': 'Country', 'sum_asset_values': 'Total Asset Value'})
        col2.dataframe(data=total_asset_value_by_nationality)

        # # Section 18: Average Age by Province
        # section_header("Average Age by Province")
        # average_age_by_province = df.groupby('TenantValue_provinces')['age'].mean()
        # col1.metric(label="Average Age by Province", value=average_age_by_province)

        # # Section 19: Average Annual Income by Branch
        # section_header("Average Annual Income by Branch")
        # average_annual_income_by_branch = df.groupby('encoded_paths')['amount_x'].mean()
        # col1.metric(label="Average Annual Income by Branch", value=average_annual_income_by_branch)

        # # Section 20: Count of Individuals with Active Status by User Role
        # section_header("Count of Individuals with Active Status by User Role")
        # active_individuals_by_user_role = df[df['Active'] == 'Active']['UserRole_ID'].value_counts()
        # col1.metric(label="Count of Active Individuals by User Role", value=active_individuals_by_user_role)

        # # Section 21: Total Holding Value by Relationship to Client
        # section_header("Total Holding Value by Relationship to Client")
        # total_holding_value_by_relationship = df.groupby('relationshiptoclient')['totalholdingvalue'].sum()
        # col1.metric(label="Total Holding Value by Relationship to Client", value=total_holding_value_by_relationship)

        # Group by "savingfor" and calculate the sum of "amountneeded"
        amount_needed_by_category = df.groupby('savingfor')['amountneeded'].sum()
        amount_needed_by_category = amount_needed_by_category.reset_index()
        amount_needed_by_category = amount_needed_by_category.rename(columns={'amountneeded':'Amount Needed', 'savingfor':'Purpose'})
        col3.dataframe(data=amount_needed_by_category)

    interactivedashboard = st.expander("Interactive dashboard", expanded=True)

    with interactivedashboard:

        col1, col2, col3 = st.columns((1,3,1))

        df_age_sliders = df[['age', 'amount_x', 'debtamount']]

        with col2:

            @st.cache  # Optional: Use Streamlit's cache to improve performance
            def filter_data(min_age, max_age):
                return df_age_sliders[(df_age_sliders['age'] >= min_age) & (df_age_sliders['age'] <= max_age)]

            # Define the function to plot the scatterplot
            def plot_income_liabilities(filtered_df):
                fig, ax = plt.subplots(figsize=(4, 3), dpi=100)  # Adjust the figsize and dpi as needed
                marker_size = 6  # Adjust the marker size (s)
                font_size = 8 # * min(fig.get_size_inches())  # Calculate the font size based on the smaller dimension of the figsize
                ax.scatter(x='amount_x', y='debtamount', data=filtered_df, s=marker_size)
                ax.tick_params(axis='both', which='major', labelsize=6)
                ax.set_xlabel('Income', fontsize=font_size)
                ax.set_ylabel('Debt Amount', fontsize=font_size)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=False)

            # Create the age range sliders
            selected_age = st.slider('Please select age range', 0.0, 100.0, (25.0, 75.0))
            min_age = selected_age[0]
            max_age = selected_age[1]

            # Filter the data based on the selected age range
            filtered_data = filter_data(min_age, max_age)

            # Plot the scatterplot
            plot_income_liabilities(filtered_data)

    # Notes on DBs
    # Incomes vs expenses, assets vs liabilities, total asset/liabilities values per gender/age, avg assets/liabs per user (use isclient flag to exclude dependents), avg marital status, avg income/expenses, assets/liabs, avg total income + capital needs, avg income + capital need per age/gender (joined to goals)
    # Provinces/where are clients located
    # Asset types by type of client
    # Expense types by type of client
    # Avg needs (disability, retirement, etc) - for income and capital needs
    # Asset groups
    # Average term of liabilities + settling terms
    # Employee benefits - tbl_employee_benefits --> tbl_defnied_benefits/tbl_defined_contribution
    # Bequests
    # Avg retirement age per client type - tbl_individual_retirement (individualclientid)
    # tbl_lifestyle_information (lookups for qualification and occupation)

        df_avg_demographics = df[['Name', 'Surname', 'age', 'TenantValue_gender', 'TenantValue_maritalstatus']]
        df_avg_demographics['full_name'] = df_avg_demographics['Name'].str.cat(df_avg_demographics['Surname'], sep=' ')

        # Assign codes to categorical variables
        marital_status_codes = {'Married': 2, 'Single': 1, 'Divorced': 0, 'Widowed': -1, 'Married In Community of Property': 2, 'Married Out Community of Property with Accrual': 2, 'Married Out Community of Property without Accrual': 2, 'None': 3, 'Engaged': 4, 'Common-Law': 2, 'Separated': 0, 'Unknown': 3}
        df_avg_demographics['maritalstatus_assignedcode'] = df_avg_demographics['TenantValue_maritalstatus'].map(marital_status_codes)

        gender_codes = {'Male': 1, 'Female': 0}
        df_avg_demographics['gender_code'] = df_avg_demographics['TenantValue_gender'].map(gender_codes)

        # Define function to get average demographics of clients for a user
        def get_user_stats(user):
            user_clients = df_avg_demographics[df_avg_demographics['full_name'] == user]
            user_clients.dropna(inplace=True)
            
            avg_age = round(user_clients['age'].mean(), 3)
            
            mode_marital_status_code = user_clients['maritalstatus_assignedcode'].mode().values
            mode_marital_status = next((k for k, v in marital_status_codes.items() if v == mode_marital_status_code[0]), None) if mode_marital_status_code.size else None
            
            mode_gender_code = user_clients['gender_code'].mode().values
            mode_gender = next((k for k, v in gender_codes.items() if v == mode_gender_code[0]), None) if mode_gender_code.size else None
            
            return {'Average Age': avg_age, 'Mode Marital Status': mode_marital_status, 'Mode Gender': mode_gender}

        # Get list of unique user names
        df_avg_demographics = df_avg_demographics.sort_values("Surname")
        user_names = df_avg_demographics['full_name'].unique().tolist()

        # Create dropdown menu of user names
        user_dropdown = st.selectbox('Select a User', user_names)

        # Define function to display average demographics for selected user
        def display_user_stats(user):
            user_stats = get_user_stats(user)
            for stat, value in user_stats.items():
                st.write(f'{stat}: {value}')

        # Display user stats for selected user
        display_user_stats(user_dropdown)

    st.write("")

    needsanalysis = st.expander('Needs analysis', expanded=True)

    with needsanalysis:

        # Select the desired columns
        df_clients_holdings_assets_users = df[['Name', 'Surname', 'age', 'TenantValue_gender', 'TenantValue_maritalstatus', 'description', 'assettype']]

        # Assign codes to categorical variables
        marital_status_codes = {'Married': 2, 'Single': 1, 'Divorced': 0, 'Widowed': -1, 'Married In Community of Property': 2, 'Married Out Community of Property with Accrual': 2, 'Married Out Community of Property without Accrual': 2, 'None': 3, 'Engaged': 4, 'Common-Law': 2, 'Separated': 0, 'Unknown': 3}
        df_clients_holdings_assets_users['maritalstatus_code'] = df_clients_holdings_assets_users['TenantValue_maritalstatus'].map(marital_status_codes)

        gender_codes = {'Male': 1, 'Female': 0}
        df_clients_holdings_assets_users['gender_code'] = df_clients_holdings_assets_users['TenantValue_gender'].map(gender_codes)

        def get_mode_value(x):
            try:
                return x.mode().values[0]
            except IndexError:
                return None

        marital_gender_description = df_clients_holdings_assets_users.groupby(['TenantValue_maritalstatus', 'TenantValue_gender'], group_keys=True)['description'].apply(get_mode_value).reset_index()

        # Determine most common holding description for each demographic group
        # marital_gender_description = df_clients_holdings_assets_users.groupby(['TenantValue_maritalstatus', 'TenantValue_gender'])['description'].apply(lambda x: x.mode()[0] if not x.empty else None).reset_index()

        # Fill missing values in the 'description' column with a placeholder value
        df_clients_holdings_assets_users['description'].fillna('Missing', inplace=True)

        # Create a dictionary to map description codes to descriptions
        description_dict = {}
        for description in df_clients_holdings_assets_users['description'].unique():
            description_dict[description] = description
        
        # Create a list to store the table data
        table_data = []

        # Loop through the rows of marital_gender_description and append the formatted data to the table data list
        for index, row in marital_gender_description.iterrows():
            description_desc = description_dict.get(row['description'], 'No data available')
            if description_desc == 'No data available':
                table_data.append([row['TenantValue_maritalstatus'], row['TenantValue_gender'], description_desc])
            else:
                table_data.append([row['TenantValue_maritalstatus'], row['TenantValue_gender'], description_desc.split(':')[-1].strip()])

        # Print the formatted table using Streamlit's st.table
        # st.table(table_data)

        # Create a new column with the description for each row
        marital_gender_description['description_desc'] = marital_gender_description['description'].apply(lambda x: description_dict.get(x, 'No data available'))

        # Remove anything before the colon in the description column
        marital_gender_description['description_desc'] = marital_gender_description['description_desc'].apply(lambda x: x.split(":")[-1].strip())

        # Create a LabelEncoder to encode the description_desc column
        le = LabelEncoder()
        marital_gender_description['description_code'] = le.fit_transform(marital_gender_description['description_desc'])

        # Create a pivot table of the marital_gender_description dataframe
        pt = pd.pivot_table(marital_gender_description, values='description_code', index='TenantValue_maritalstatus', columns='TenantValue_gender', fill_value=0, aggfunc='max')

        # Create a heatmap of the pivot table using the description_code column as the values
        ax = sns.heatmap(pt, annot=True, cmap='Blues', fmt='g', annot_kws={'size': 14})

        # Set the x and y axis labels
        ax.set_xlabel('Gender')
        ax.set_ylabel('Marital Status')

        # Get the tick labels for the colorbar
        ticklabels = le.inverse_transform(range(len(le.classes_)))

        # Get the colorbar axes
        cbar_ax = ax.figure.add_axes([0.95, 0.1, 0.02, 0.8])  # Adjust the position and size of the colorbar

        # Add a colorbar with tick labels
        cbar = plt.colorbar(ax.collections[0], cax=cbar_ax, cmap='Blues')
        cbar.set_ticks(range(len(le.classes_)))
        cbar.set_ticklabels(ticklabels)

        # Set the label for the colorbar
        cbar.set_label('Description')

        # Show the plot using Streamlit's st.pyplot
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    st.subheader("Notes on dashboard analysis")

    # Spawn a new Quill editor
    st.subheader("Notes on dashboard data analysis")
    edacontent = st_quill(placeholder="Write your notes here", value=st.session_state.dbnotes, key="dbquill")

    st.session_state.edanotes = edacontent

    st.write("Exploratory data analysis took ", time.time() - start_time, "seconds to run")
