import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
from st_on_hover_tabs import on_hover_tabs
# from streamlit.state.session_state import SessionState
from PIL import Image
from functions import *
import keyboard
import os
import signal
import streamlit.components.v1 as components
import time
from indalohomeveritas import indalohome
import apps
from streamlit_option_menu import option_menu
from indalodashboardsveritas import indalodashboards
from indaloedaveritas import indaloeda
from indalovaluemapveritas import indalovaluemap
from apps.login_app_veritas import LoginApp
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit.runtime.scriptrunner import RerunData, RerunException
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")

# Run dtale-streamlit as a subprocess
command = [
    "dtale-streamlit",
    "run",
    __file__,  # Use the current file
    "--theme.primaryColor=#FFCC66",
    "--client.showErrorDetails=false"
]

st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)

# Hide white space at top

hide_streamlit_style = """
    <style>
        #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 1rem;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Customized footer

customizedfooter = """
            <style>
            footer {
	
	visibility: hidden;
	
	}
    footer:after {
        content:'Made by and (c) Ryan Blumenow';
        visibility: visible;
        display: block;
        position: relative;
        #background-color: maroon;
        padding: 5px;
        top: 2px;
        left: 630px;
    }</style>"""

st.markdown(customizedfooter, unsafe_allow_html=True)

with st.sidebar:
        st.image('indalologo.png')
        add_vertical_space(1)
        value = on_hover_tabs(tabName=['SPONSORS', 'ABOUT', 'CONTACT US', ''], 
                            iconName=['contacts', 'dashboard', 'account_tree', 'table', 'report', 'edit', 'update', 'pivot_table_chart', 'menu_book'],
                            styles = {'navtab': {'background-color': "maroon",
                                                'color': 'white',
                                                'font-size': '18px',
                                                'transition': '.3s',
                                                'white-space': 'nowrap',
                                                'text-transform': 'uppercase',
                                                'font-weight': 'bold'},
                                    'tabOptionsStyle': {':hover :hover': {'color': '#dcac54',
                                                                    'cursor': 'pointer'}},
                                    'iconStyle':{'position':'fixed',
                                                    'left':'7.5px',
                                                    'text-align': 'left'},
                                    'tabStyle' : {'list-style-type': 'none',
                                                    'margin-bottom': '30px',
                                                    'padding-left': '30px'}},
                            key="hoversidebar",
                            default_choice=3)
        
        if value == "SPONSORS":
            st.write("")
            st.header("")
            st.text_area("", "", height=240)

        if value == "ABOUT":
            st.write("")
            st.header("")
            st.text_area(
                "",
                "",
                height=321,
            )
        elif value == "CONTACT US":
            st.write("")
            # Custom HTML for Contact Page
            # Add Font Awesome

css = '''
<style>
    .stTabs [data-baseweb="tab-highlight"] {
        background-color:maroon;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

# Initialize Session State for Navigation and Click Handling
if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "Indalo home"
if "last_clicked_bar" not in st.session_state:
    st.session_state["last_clicked_bar"] = None


# Custom Navbar using streamlit-option-menu
def custom_navbar():

    selected = option_menu(
        menu_title=None,  # required
        options=["Indalo home", "Indalo EDA", "Indalo M&E", "Indalo value map", "Log out"],  # required
        icons=["house", "bar-chart", "clipboard", "map", "cross"],  # optional
        menu_icon="cast",  # optional
        default_index=["Indalo home", "Indalo EDA", "Indalo M&E", "Indalo value map", "Log out"].index(
            st.session_state["selected_page"]
        ),  # Sync default index with the session state
        orientation="horizontal",
        styles={
            "container": {
                "padding": "0 10px",
                "background-color": "maroon", # "#800000",
                "border-radius": "10px",
                "display": "flex",
                "justify-content": "flex-start",  # Align items to the left
                "margin-left": "10px",  # Shift the navbar to the left
                "margin-right": "20px",
                "width": "100%",  # Optional: Adjust the navbar width
                "box-sizing": "border-box",  # Ensure padding and border are included in the total width
            },
            "icon": {
                "font-size": "18px",
            },
            "nav-link": {
                "font-size": "14px",
                "text-align": "center",
                "margin": "0 5px",  # Reduced margin to make it compact
                "color": "white",
                "padding": "10px",
                "border-radius": "10px",
                "--hover-color": "#DB5C45"
            },
            "nav-link-selected": {
                "background-color": "white",
                "color": "maroon",
                "border-radius": "8px",
            },
        },
    )

    # Inject custom CSS for selected icon styling
    st.markdown("""
        <style>
        .nav-link-selected svg {
            fill: maroon !important;  /* Change icon color to red for selected tab */
        }
        </style>
    """, unsafe_allow_html=True)

    if selected != st.session_state["selected_page"]:
        st.session_state["selected_page"] = selected
        st.rerun()

# Main Page Rendering
def main():

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["username"] = None
        st.session_state["selected_page"] = "Indalo home"  # Default page

    if not st.session_state["logged_in"]:
        app = LoginApp(title="Indalo Login")
        app.run()
    else:
        # Render Navbar
        custom_navbar()

        # Render Content Based on Selected Page
        if st.session_state["selected_page"] == "Indalo home":
            indalohome()
        elif st.session_state["selected_page"] == "Indalo EDA":
            indaloeda()
        elif st.session_state["selected_page"] == "Indalo M&E":
            indalodashboards()
        elif st.session_state["selected_page"] == "Indalo value map":
            indalovaluemap()
        elif st.session_state["selected_page"] == "Log out":
            # Log out the user
            st.session_state['logged_in'] = False
            st.session_state['username'] = None
            st.session_state["selected_page"] = "Indalo home"  # Reset to default page
            st.rerun()

# Run the App
if __name__ == "__main__":
    main()


