import pandas as pd
import streamlit as st
from scipy import stats
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

# LLM response function
def get_llm_categorical_response(llm, explanation):
    if llm is None:
        logger.warning("LLM is not initialized. Skipping explanation.")
        return None

    system_message_prompt = """
    You are StatBot, an expert statistical analyst. 
    Explain the results of a categorical data analysis in simple English.
    """
    
    human_message_prompt = f"""
    Results:
    {explanation}
    
    Based on these results, explain the significance in a crisp manner. Do not overdo it.
    """
    
    try:
        response = llm.invoke(human_message_prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error getting LLM response: {str(e)}")
        return None

def categorical_data_analysis():
    # Initialize or get the LLM
    llm = initialize_llm()
            
    if 'df' in st.session_state and st.session_state.df is not None:
        sub_page = st.selectbox(
            "Choose a task for categorical data analysis:",
            ["Frequency Table", "Pareto Chart", "Mode", 
            "Chi Square test of Independence"]
        )
        
        if sub_page == "Frequency Table":
            display_frequency_table(llm)
        elif sub_page == "Pareto Chart":
            display_pareto_chart(llm)
        elif sub_page == "Mode":
            display_mode(llm)
        elif sub_page == "Chi Square test of Independence":
            display_chi_square_test_of_independence(llm)
    else:
        st.warning("Please upload data to proceed.")

def display_frequency_table(llm):
    feature_to_plot = st.selectbox("Select a feature for the frequency table:", st.session_state.df.columns)
    if st.button("Generate Frequency Table"):
        freq_table = st.session_state.df[feature_to_plot].value_counts()
        fig = px.bar(freq_table, x=freq_table.index, y=freq_table.values, 
                    labels={'x': feature_to_plot, 'y': 'Frequency'},
                    title=f'Frequency Distribution of {feature_to_plot}')
        fig.update_layout(width=1000, height=600)
        st.plotly_chart(fig)

        # LLM Explanation
        explanation = f"Feature: {feature_to_plot}\nFrequency Table:\n{freq_table.to_string()}"
        display_llm_explanation(llm, explanation)

def display_pareto_chart(llm):
    feature_to_plot = st.selectbox("Select a feature for the Pareto chart:", st.session_state.df.columns)
    if st.button("Generate Pareto Chart"):
        freq = st.session_state.df[feature_to_plot].value_counts().reset_index()
        freq.columns = [feature_to_plot, 'Frequency']
        freq = freq.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
        freq['Cumulative Percentage'] = freq['Frequency'].cumsum() / freq['Frequency'].sum() * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(x=freq[feature_to_plot], y=freq['Frequency'], name='Frequency', marker_color='blue'))
        fig.add_trace(go.Scatter(x=freq[feature_to_plot], y=freq['Cumulative Percentage'], 
                                name='Cumulative %', yaxis='y2', mode='lines+markers', marker_color='red'))
        fig.update_layout(
            title=f'Pareto Chart for {feature_to_plot}',
            yaxis=dict(title='Frequency'),
            yaxis2=dict(title='Cumulative Percentage', overlaying='y', side='right', range=[0, 100]),
            xaxis=dict(title=feature_to_plot),
            width=1000, height=600,
            showlegend=False
        )
        st.plotly_chart(fig)

        # LLM Explanation
        explanation = f"Pareto Chart for feature {feature_to_plot} shows cumulative frequency."
        display_llm_explanation(llm, explanation)

def display_mode(llm):
    feature_to_plot = st.selectbox("Select a feature for Mode analysis:", st.session_state.df.columns)
    if st.button("Calculate Mode"):
        freq = st.session_state.df[feature_to_plot].value_counts().reset_index()
        st.write(freq)

        # LLM Explanation
        explanation = f"Feature: {feature_to_plot}\nMode analysis results:\n{freq.to_string()}"
        display_llm_explanation(llm, explanation)

def display_chi_square_test_of_independence(llm):
    st.header("Chi Square Test of Independence")
    option1 = st.selectbox("Select Category 1:", st.session_state.df.columns)
    option2 = st.selectbox("Select Category 2:", st.session_state.df.columns)
    
    if st.button("Run Test"):
        crosstab = pd.crosstab(st.session_state.df[option1], st.session_state.df[option2])
        chi2_stat, p_val, dof, expected = stats.chi2_contingency(crosstab)
        
        st.write(f"Chi-Square Statistic: {chi2_stat}")
        st.write(f"P-value: {p_val}")
        st.write(f"Degrees of Freedom: {dof}")

        # LLM Explanation
        explanation = f"Chi-Square Test between {option1} and {option2}:\nChi2 Statistic: {chi2_stat}\nP-value: {p_val}\nDOF: {dof}. Explain it in a very detailed manner"
        display_llm_explanation(llm, explanation)

def display_llm_explanation(llm, explanation):
    if llm is not None:
        response = get_llm_categorical_response(llm, explanation)
        if response:
            st.divider()
            st.write("## Explanation from LLM:")
            st.markdown(response)
            st.divider()
    else:
        st.warning("LLM is not initialized. Please check your API key and model settings.")
        logger.warning("LLM explanation skipped due to uninitialized LLM")

# Main function to run the Streamlit app
# def main():
#     st.title("Categorical Data Analysis")
#     categorical_data_analysis()

# if __name__ == "__main__":
#     main()