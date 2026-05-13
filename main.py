import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from upload_data import upload_data
from src.components.Categorical_Analysis import categorical_data_analysis
from src.components.Continuous_Analysis import continuous_data_analysis
from src.components.Regression_Analysis import  regression_analysis
def main_updated():
    
    

    # Sidebar for primary task selection
    primary_task = st.sidebar.radio(
    "Choose a primary task:",
    ["Data Upload", "Categorical data analysis", "Continuous data analysis", 
     "Regression Analysis", "Extensive Data Analysis","Time Series Analysis","Causality Analysis", "Decision Tree Analysis", "Save"]
    )

    if primary_task == "Data Upload":
        upload_data()
    elif primary_task=="Categorical data analysis":
        categorical_data_analysis()
    elif primary_task=="Continuous data analysis":
        continuous_data_analysis()
    elif primary_task == "Regression Analysis":
        regression_analysis()

def missing_value_analysis(df):
    st.subheader("Missing Value Analysis")

    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100

    missing_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_percent
    })

    st.dataframe(missing_df)

    fig, ax = plt.subplots()
    missing.sort_values(ascending=False).plot(kind='bar', ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)


if __name__ == "__main__":
    main_updated()

        
        
        
    
