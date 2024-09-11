import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from upload_data import upload_data
from Categorical_Analysis import categorical_data_analysis
from Continuous_Analysis import continuous_data_analysis
from Regression_Analysis import  regression_analysis
import warnings

def statistical_analysis1():
# Use query parameters to track the active tab
    query_params = st.experimental_get_query_params()
    active_tab = query_params.get("tab", ["categorical"])[0]  # Default tab is categorical

    # Display the tabs
    tab1, tab2, tab3 = st.tabs(["Categorical Data Analysis", "Continuous Data Analysis", "Regression Analysis"])

   # Categorical Data Analysis tab
    with tab1:
        st.write("## Categorical Data Analysis")
        categorical_data_analysis()

    # Continuous Data Analysis tab
    with tab2:
        st.write("## Continuous Data Analysis")
        continuous_data_analysis()

    # Regression Analysis tab
    with tab3:
        st.write("## Regression Analysis")
        regression_analysis()


if __name__ == "__main__":
    if 'active_tab' not in st.session_state:    
        st.session_state.active_tab = 'categorical'
    statistical_analysis1()