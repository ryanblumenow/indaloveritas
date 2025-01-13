import os
os.system("pip install hydralit")
os.system("pip install streamlit-folium")

from hydralit import HydraApp
import streamlit as st
# from streamlit.state.session_state import SessionState
from PIL import Image
from functions import *
import keyboard
import os
import signal
import streamlit.components.v1 as components
import hydralit_components as hl
import hydralit_components as hc
import time
from indalohome import indalohome
import apps
from streamlit_option_menu import option_menu
from indalodashboards import indalodashboardshome
from indaloeda import indaloeda
from indalovaluemap import indalovaluemap
from apps.login_app import LoginApp
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit.runtime.scriptrunner import RerunData, RerunException

# Monkey-patch st.experimental_rerun
def experimental_rerun():
    raise RerunException(RerunData())

st.experimental_rerun = experimental_rerun

if __name__ == '__main__':

    if 'pagechoice' not in st.session_state:
        st.session_state['pagechoice'] = 'home'

    #this is the host application, we add children to it and that's it!
    app = HydraApp(title='indalo Analytics', favicon="indalologo.png", hide_streamlit_markers=False, layout='wide') # navbar_sticky=True), navbar_mode='sticky', use_navbar=True)

    # from st_on_hover_tabs import on_hover_tabs

    # st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)

    # with st.sidebar:
    #         st.image('indalologo.jpg')
    #         add_vertical_space(1)
    #         # st.markdown(":grey[Please chooose domain]")
    #         # add_vertical_space(1)
    #         value = on_hover_tabs(tabName=['Supplier', 'Item'], 
    #                             iconName=['contacts', 'dashboard'],
    #                             styles = {'navtab': {'background-color':'#6d0606',
    #                                                 'color': 'white',
    #                                                 'font-size': '18px',
    #                                                 'transition': '.3s',
    #                                                 'white-space': 'nowrap',
    #                                                 'text-transform': 'uppercase'},
    #                                     'tabOptionsStyle': {':hover :hover': {'color': 'red',
    #                                                                     'cursor': 'pointer'}},
    #                                     'iconStyle':{'position':'fixed',
    #                                                     'left':'7.5px',
    #                                                     'text-align': 'left'},
    #                                     'tabStyle' : {'list-style-type': 'none',
    #                                                     'margin-bottom': '30px',
    #                                                     'padding-left': '30px'}},
    #                             key="hoversidebar1",
    #                             default_choice=0)

    # css = '''
    # <style>
    #     .stTabs [data-baseweb="tab-highlight"] {
    #         background-color:blue;
    #     }
    # </style>
    # '''

    # st.markdown(css, unsafe_allow_html=True)

    # if value == 'Supplier':
    #     print('1')

    # To eliminate space at top of page

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
        #background-color: red;
        padding: 5px;
        top: 2px;
        left: 630px;
    }</style>"""

    st.markdown(customizedfooter, unsafe_allow_html=True)

    # header1 = st.container()

    # with header1:
    #     clm1, clm2, clm3, clm4, clm5 = st.columns([1.45,1,1,1,1])
    #     with clm1:
    #         pass
    #     with clm2:
    #         pass
    #     with clm3:
    #         # image1 = Functions.set_svg_image('indalologo.jpg')
    #         # image2 = Image.open(image1)
    #         # st.image(image2, width=283)
    #         # st.image("indalologo.jpg", use_column_width=False, width=180, caption="")
    #         pass
    #     with clm4:
    #         pass
    #     with clm5:
    #         quitapp = st.button("Exit", key="multipgexit")
    
    # if quitapp==True:
    #     keyboard.press_and_release('ctrl+w')
    #     os.kill(os.getpid(), signal.SIGTERM)

    #add all your application classes here
    # app.add_app(title="Tiger login",icon="analytics.ico", app=tigerlogin())
    # app.add_app(title="Dashboards2", icon="analytics.ico", app=dashboards2())
    # app.add_app(title="Dashboards3", icon="analytics.ico", app=dashboards3())
    # app.add_app(title="Graphs", icon="analytics.ico", app=graphs())
    # app.add_app(title="Advanced analytics", icon="analytics.ico", app=advanalytics())
    # app.add_app(title="Make a prediction", icon="analytics.ico", app=predictions())
    # app.add_app(title="Overview", icon="analytics.ico", app=tigerdashboardshome())
    # app.add_app(title="Profiling", icon="analytics.ico", app=profiling())
    app.add_app(title="Indalo home", icon="./gui/svg_icons/home.png", app=indalohome())
    app.add_app(title="Indalo EDA", icon="dartico.jpg", app=indaloeda())
    app.add_app(title="Indalo M&E", icon="analytics.ico", app=indalodashboardshome())
    app.add_app(title="Indalo value map", icon="analytics.ico", app=indalovaluemap())
    app.add_loader_app(apps.MyLoadingApp(delay=0))
    app.add_app("Log out", apps.LoginApp(title='Login'),is_login=True)

    # app.add_app("Log out", apps.LoginApp(title='Login'),is_login=True)

    complex_nav = {
            # 'Tiger login': ['Tiger login'],
            # 'Analytics overview': ['Graphs'],
            # 'Overview   ': ['Overview'],
            'Indalo home': ['Indalo home'],
            'Indalo EDA': ['Indalo EDA'],
            'Indalo M&E': ['Indalo M&E'],
            'Indalo value map': ['Indalo value map']
            # 'Analytics dashboards': ['Dashboards', 'Dashboards2', 'Dashboards3'],
            # 'Advanced analytics': ['Advanced analytics'],
            # 'Make a prediction': ['Make a prediction'],
            # 'Profiling': ['Profiling']
    }

    #run the whole lot
    app.run(complex_nav)
