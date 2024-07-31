import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px

def check(df):
    l=[]
    columns=df.columns
    for col in columns:
        dtypes=df[col].dtypes
        nunique=df[col].nunique()
        duplicated=df.duplicated().sum()
        sum_null=df[col].isnull().sum()
        l.append([col,dtypes,nunique,duplicated,sum_null])
    df_check=pd.DataFrame(l)
    df_check.columns=['columns','Data Types','No of Unique Values','No of Duplicated Rows','No of Null Values']
    return df_check 

def interactive_data_cleaning():
    st.header("Interactive Data Cleaning")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)

        # Data overview
        st.subheader("Data Overview")
        st.write(f"Shape: {df.shape}")
        st.write(f"Columns: {', '.join(df.columns)}")

        # Column selection for cleaning
        column = st.selectbox("Select a column to clean", df.columns)

        # Data cleaning options
        st.subheader(f"Cleaning options for {column}")

        # Handle missing values
        if df[column].isnull().sum() > 0:
            missing_action = st.radio(
                f"Handle missing values in {column}",
                ("Drop", "Fill with mean", "Fill with median", "Fill with mode", "Fill with custom value")
            )
            if missing_action == "Drop":
                df = df.dropna(subset=[column])
            elif missing_action == "Fill with mean":
                df[column] = df[column].fillna(df[column].mean())
            elif missing_action == "Fill with median":
                df[column] = df[column].fillna(df[column].median())
            elif missing_action == "Fill with mode":
                df[column] = df[column].fillna(df[column].mode()[0])
            elif missing_action == "Fill with custom value":
                custom_value = st.text_input(f"Enter custom value for {column}")
                if custom_value:
                    df[column] = df[column].fillna(custom_value)

        # Handle outliers (for numeric columns)
        if pd.api.types.is_numeric_dtype(df[column]):
            st.subheader(f"Outlier detection for {column}")
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            
            st.write(f"Number of outliers detected: {len(outliers)}")
            if len(outliers) > 0:
                outlier_action = st.radio(
                    f"Handle outliers in {column}",
                    ("Keep", "Remove", "Cap")
                )
                if outlier_action == "Remove":
                    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
                elif outlier_action == "Cap":
                    df[column] = df[column].clip(lower_bound, upper_bound)

        # Data transformation
        st.subheader(f"Transform {column}")
        transform_action = st.selectbox(
            f"Apply transformation to {column}",
            ("None", "Log", "Square root", "Min-Max scaling")
        )
        if transform_action == "Log":
            df[f"{column}_log"] = np.log1p(df[column])
        elif transform_action == "Square root":
            df[f"{column}_sqrt"] = np.sqrt(df[column])
        elif transform_action == "Min-Max scaling":
            df[f"{column}_scaled"] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

        # Display updated dataframe
        st.subheader("Updated Data")
        st.write(df)

        # Data visualization
        st.subheader("Data Visualization")
        if pd.api.types.is_numeric_dtype(df[column]):
            fig = px.histogram(df, x=column)
            st.plotly_chart(fig)

        # Generate pandas profiling report
        if st.button("Generate Detailed Report"):
            pr = ProfileReport(df, explorative=True)
            st_profile_report(pr)

        # Option to download cleaned data
        st.download_button(
            label="Download cleaned data as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='cleaned_data.csv',
            mime='text/csv',
        )
