import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from langchain_groq import ChatGroq
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM initialization
def initialize_llm():
    if 'llm' not in st.session_state or st.session_state.llm is None:
        if st.session_state.get('groq_api_key'):
            try:
                st.session_state.llm = ChatGroq(
                    groq_api_key=st.session_state['groq_api_key'], 
                    model_name=st.session_state['model_name'],
                    temperature=st.session_state['temperature'],
                    top_p=st.session_state['top_p']
                )
                logger.info("LLM initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing LLM: {str(e)}")
                st.sidebar.error(f"Error initializing model: {str(e)}")
                st.session_state.llm = None
        else:
            logger.warning("Groq API key not found in session state")
            st.session_state.llm = None
    return st.session_state.llm

# LLM response function with error handling and retry
def get_llm_visualization_response(llm, explanation, max_retries=3):
    if llm is None:
        logger.warning("LLM is not initialized. Skipping explanation.")
        return None

    system_message_prompt = """
    You are VisualBot, an expert in data visualization and analysis. 
    Explain the results of a data visualization in simple English.
    """
    
    human_message_prompt = f"""
    Visualization:
    {explanation}
    
    Based on this visualization, explain the key insights in a concise manner. Do not overdo it.
    """
    
    for attempt in range(max_retries):
        try:
            response = llm.invoke(human_message_prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error getting LLM response (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                st.error(f"Failed to get LLM response after {max_retries} attempts")
                return None

def visualization():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Dataset Visualization App</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Explore various visualizations of your dataset using interactive Plotly charts.</p>", unsafe_allow_html=True)
    st.divider()

    # Initialize or get the LLM
    llm = initialize_llm()

    if 'df' in st.session_state and st.session_state.df is not None:
        sub_page = st.selectbox(
            "Choose a visualization type:",
            ["Histogram", "Scatter Plot", "Line Plot", "Box Plot", "Bar Plot", "Heatmap", "Pie Chart", "Violin Plot"]
        )
        
        if sub_page == "Histogram":
            display_histogram(llm)
        elif sub_page == "Scatter Plot":
            display_scatter_plot(llm)
        elif sub_page == "Line Plot":
            display_line_plot(llm)
        elif sub_page == "Box Plot":
            display_box_plot(llm)
        elif sub_page == "Bar Plot":
            display_bar_plot(llm)
        elif sub_page == "Heatmap":
            display_heatmap(llm)
        elif sub_page == "Pie Chart":
            display_pie_chart(llm)
        elif sub_page == "Violin Plot":
            display_violin_plot(llm)
    else:
        st.warning("Please upload data to proceed.")

def display_histogram(llm):
    column = st.selectbox("Select a column for the histogram:", st.session_state.df.columns)
    if st.button("Generate Histogram"):
        fig = px.histogram(st.session_state.df, x=column, marginal="box", nbins=30, title=f'Histogram of {column}')
        fig.update_layout(width=1000, height=600)
        st.plotly_chart(fig)

        explanation = f"Histogram of {column}"
        display_llm_explanation(llm, explanation)

def display_scatter_plot(llm):
    column1 = st.selectbox("Select X-axis column:", st.session_state.df.columns)
    column2 = st.selectbox("Select Y-axis column:", st.session_state.df.columns)
    if st.button("Generate Scatter Plot"):
        fig = px.scatter(st.session_state.df, x=column1, y=column2, title=f'Scatter Plot of {column2} vs {column1}')
        fig.update_layout(width=1000, height=600)
        st.plotly_chart(fig)

        explanation = f"Scatter Plot of {column2} vs {column1}"
        display_llm_explanation(llm, explanation)

def display_line_plot(llm):
    column1 = st.selectbox("Select X-axis column:", st.session_state.df.columns)
    column2 = st.selectbox("Select Y-axis column:", st.session_state.df.columns)
    if st.button("Generate Line Plot"):
        fig = px.line(st.session_state.df, x=column1, y=column2, title=f'Line Plot of {column2} vs {column1}')
        fig.update_layout(width=1000, height=600)
        st.plotly_chart(fig)

        explanation = f"Line Plot of {column2} vs {column1}"
        display_llm_explanation(llm, explanation)

def display_box_plot(llm):
    column = st.selectbox("Select a column for the box plot:", st.session_state.df.columns)
    if st.button("Generate Box Plot"):
        fig = px.box(st.session_state.df, y=column, title=f'Box Plot of {column}')
        fig.update_layout(width=1000, height=600)
        st.plotly_chart(fig)

        explanation = f"Box Plot of {column}"
        display_llm_explanation(llm, explanation)

def display_bar_plot(llm):
    column1 = st.selectbox("Select X-axis column:", st.session_state.df.columns)
    column2 = st.selectbox("Select Y-axis column:", st.session_state.df.columns)
    if st.button("Generate Bar Plot"):
        fig = px.bar(st.session_state.df, x=column1, y=column2, title=f'Bar Plot of {column2} by {column1}')
        fig.update_layout(width=1000, height=600)
        st.plotly_chart(fig)

        explanation = f"Bar Plot of {column2} by {column1}"
        display_llm_explanation(llm, explanation)

def display_heatmap(llm):
    if st.button("Generate Heatmap"):
        fig = px.imshow(st.session_state.df.corr(), title='Correlation Heatmap')
        fig.update_layout(width=1000, height=600)
        st.plotly_chart(fig)

        explanation = "Correlation Heatmap of the dataset"
        display_llm_explanation(llm, explanation)

def display_pie_chart(llm):
    column = st.selectbox("Select a column for the pie chart:", st.session_state.df.columns)
    if st.button("Generate Pie Chart"):
        fig = px.pie(st.session_state.df, names=column, title=f'Pie Chart of {column}')
        fig.update_layout(width=1000, height=600)
        st.plotly_chart(fig)

        explanation = f"Pie Chart of {column}"
        display_llm_explanation(llm, explanation)

def display_violin_plot(llm):
    column1 = st.selectbox("Select X-axis column:", st.session_state.df.columns)
    column2 = st.selectbox("Select Y-axis column:", st.session_state.df.columns)
    if st.button("Generate Violin Plot"):
        fig = px.violin(st.session_state.df, x=column1, y=column2, box=True, points="all", title=f'Violin Plot of {column2} by {column1}')
        fig.update_layout(width=1000, height=600)
        st.plotly_chart(fig)

        explanation = f"Violin Plot of {column2} by {column1}"
        display_llm_explanation(llm, explanation)

def display_llm_explanation(llm, explanation):
    if llm is not None:
        response = get_llm_visualization_response(llm, explanation)
        if response:
            st.divider()
            st.write("## Explanation from LLM:")
            st.markdown(response)
            st.divider()
    else:
        st.warning("LLM is not initialized. Please check your API key and model settings.")
        logger.warning("LLM explanation skipped due to uninitialized LLM")

# # Main function to run the Streamlit app
# def main():
#     # Your main Streamlit app code here
#     # Make sure to call visualization() function when needed
#     visualization()

# if __name__ == "__main__":
#     main()