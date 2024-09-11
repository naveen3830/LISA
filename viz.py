import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from langchain_groq import ChatGroq

llm = None
if st.session_state.get('groq_api_key'):
    try:
        llm = ChatGroq(
            groq_api_key=st.session_state['groq_api_key'], 
            model_name=st.session_state['model_name'],
            temperature=st.session_state['temperature'],
            top_p=st.session_state['top_p']
        )
    except Exception as e:
        st.sidebar.error(f"Error initializing model: {str(e)}")

# LLM response function
def get_llm_visualization_response(llm, explanation):
    system_message_prompt = """
    You are VisualBot, an expert in data visualization and analysis. 
    Explain the results of a data visualization in simple English.
    """
    
    human_message_prompt = f"""
    Visualization:
    {explanation}
    
    Based on this visualization, explain the key insights in a concise manner. Do not overdo it.
    """
    
    response = llm.invoke(human_message_prompt)
    return response.content

def visualization():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Dataset Visualization App</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Explore various visualizations of your dataset using interactive Plotly charts.</p>", unsafe_allow_html=True)
    st.divider()
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

        # LLM Explanation
        explanation = f"Histogram of {column}"
        if llm:
            response = get_llm_visualization_response(llm, explanation)
            st.divider()
            st.write("## Explanation from LLM:")
            st.markdown(response)
            st.divider()

def display_scatter_plot(llm):
    column1 = st.selectbox("Select X-axis column:", st.session_state.df.columns)
    column2 = st.selectbox("Select Y-axis column:", st.session_state.df.columns)
    if st.button("Generate Scatter Plot"):
        fig = px.scatter(st.session_state.df, x=column1, y=column2, title=f'Scatter Plot of {column2} vs {column1}')
        fig.update_layout(width=1000, height=600)
        st.plotly_chart(fig)

        # LLM Explanation
        explanation = f"Scatter Plot of {column2} vs {column1}"
        if llm:
            response = get_llm_visualization_response(llm, explanation)
            st.divider()
            st.write("## Explanation from LLM:")
            st.markdown(response)
            st.divider()

def display_line_plot(llm):
    column1 = st.selectbox("Select X-axis column:", st.session_state.df.columns)
    column2 = st.selectbox("Select Y-axis column:", st.session_state.df.columns)
    if st.button("Generate Line Plot"):
        fig = px.line(st.session_state.df, x=column1, y=column2, title=f'Line Plot of {column2} vs {column1}')
        fig.update_layout(width=1000, height=600)
        st.plotly_chart(fig)

        # LLM Explanation
        explanation = f"Line Plot of {column2} vs {column1}"
        if llm:
            response = get_llm_visualization_response(llm, explanation)
            st.divider()
            st.write("## Explanation from LLM:")
            st.markdown(response)
            st.divider()

def display_box_plot(llm):
    column = st.selectbox("Select a column for the box plot:", st.session_state.df.columns)
    if st.button("Generate Box Plot"):
        fig = px.box(st.session_state.df, y=column, title=f'Box Plot of {column}')
        fig.update_layout(width=1000, height=600)
        st.plotly_chart(fig)

        # LLM Explanation
        explanation = f"Box Plot of {column}"
        if llm:
            response = get_llm_visualization_response(llm, explanation)
            st.divider()
            st.write("## Explanation from LLM:")
            st.markdown(response)
            st.divider()

def display_bar_plot(llm):
    column1 = st.selectbox("Select X-axis column:", st.session_state.df.columns)
    column2 = st.selectbox("Select Y-axis column:", st.session_state.df.columns)
    if st.button("Generate Bar Plot"):
        fig = px.bar(st.session_state.df, x=column1, y=column2, title=f'Bar Plot of {column2} by {column1}')
        fig.update_layout(width=1000, height=600)
        st.plotly_chart(fig)

        # LLM Explanation
        explanation = f"Bar Plot of {column2} by {column1}"
        if llm:
            response = get_llm_visualization_response(llm, explanation)
            st.divider()
            st.write("## Explanation from LLM:")
            st.markdown(response)
            st.divider()

def display_heatmap(llm):
    if st.button("Generate Heatmap"):
        fig = px.imshow(st.session_state.df.corr(), title='Correlation Heatmap')
        fig.update_layout(width=1000, height=600)
        st.plotly_chart(fig)

        # LLM Explanation
        explanation = "Correlation Heatmap of the dataset"
        if llm:
            response = get_llm_visualization_response(llm, explanation)
            st.divider()
            st.write("## Explanation from LLM:")
            st.markdown(response)
            st.divider()

def display_pie_chart(llm):
    column = st.selectbox("Select a column for the pie chart:", st.session_state.df.columns)
    if st.button("Generate Pie Chart"):
        fig = px.pie(st.session_state.df, names=column, title=f'Pie Chart of {column}')
        fig.update_layout(width=1000, height=600)
        st.plotly_chart(fig)

        # LLM Explanation
        explanation = f"Pie Chart of {column}"
        if llm:
            response = get_llm_visualization_response(llm, explanation)
            st.divider()
            st.write("## Explanation from LLM:")
            st.markdown(response)
            st.divider()

def display_violin_plot(llm):
    column1 = st.selectbox("Select X-axis column:", st.session_state.df.columns)
    column2 = st.selectbox("Select Y-axis column:", st.session_state.df.columns)
    if st.button("Generate Violin Plot"):
        fig = px.violin(st.session_state.df, x=column1, y=column2, box=True, points="all", title=f'Violin Plot of {column2} by {column1}')
        fig.update_layout(width=1000, height=600)
        st.plotly_chart(fig)

        # LLM Explanation
        explanation = f"Violin Plot of {column2} by {column1}"
        if llm:
            response = get_llm_visualization_response(llm, explanation)
            st.divider()
            st.write("## Explanation from LLM:")
            st.markdown(response)
            st.divider()