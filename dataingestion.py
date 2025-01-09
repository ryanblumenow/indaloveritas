import pandas as pd
import streamlit as st
import pyodbc
import numpy as np

@st.cache_data
def readdata(init): 

    init = 1

    # Load the data
    original_data = pd.ExcelFile("SiAGIA Baseline Report.xlsx")

    # Load individual sheets
    df_consolidated = pd.read_excel(original_data, 'Consolidated')
    df_combined = pd.read_excel(original_data, 'Combined')
    indalovator_data = pd.read_excel(original_data, 'Indalovator CH2')
    indalogrow_data = pd.read_excel(original_data, 'Indalogrow CH2')

    return df_consolidated, df_combined, indalovator_data, indalogrow_data