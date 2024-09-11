import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage
from langchain.schema import AIMessage
from langchain_core.output_parsers import StrOutputParser
from pandasql import sqldf
from functions import check
from dotenv import load_dotenv
from pathlib import Path
import os
from streamlit_option_menu import option_menu

def get_llm_response(llm, prompt_template, data):
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        "You are StatBot, an expert statistical analyst. "
        "Explain the output in simple English. Straight away start with your explanations.")
    human_message_prompt = HumanMessagePromptTemplate.from_template(prompt_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    formatted_chat_prompt = chat_prompt.format_messages(**data)
    response = llm.invoke(formatted_chat_prompt)
    return response.content

def groq_infer(llm, prompt):
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    print(response.content)
    return response.content
template = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. Don't add \n characters.

You must output the SQL query that answers the question in a single line.

### Input:
`{question}`

### Context:
`{context}`

### Response:
"""
prompt = PromptTemplate.from_template(template=template)


def load_sidebar():
    # Load sidebar elements for API key and model parameters
    with st.sidebar:
        st.divider()
        with st.sidebar.expander("Get Your API Key Here"):
            st.markdown("## How to use\n"
            "1. Enter your [Groq API key](https://console.groq.com/keys) below🔑\n" 
            "2. Upload a CSV file📄\n"
            "3. Let LISA do its work!!!💬\n")
        
        st.session_state['groq_api_key'] = st.text_input("Enter your Groq API key:", type="password",
                                                        placeholder="Paste your Groq API key here (gsk_...)",
                                                        value=st.session_state.get('groq_api_key', ''))
        
        st.session_state['model_name'] = st.selectbox("Select Model:", 
                                                    ["llama-3.1-70b-versatile","llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"], 
                                                    index=st.session_state.get('model_name_index', 0))
        
        st.session_state['temperature'] = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=st.session_state.get('temperature', 0.5), step=0.1)
        st.session_state['top_p'] = st.slider("Top-p:", min_value=0.0, max_value=1.0, value=st.session_state.get('top_p', 1.0), step=0.25)

def Home():
    load_sidebar()
    st.divider()

    llm = None
    if st.session_state['groq_api_key']:
        try:
            llm = ChatGroq(
                    groq_api_key=st.session_state['groq_api_key'], 
                    model_name=st.session_state['model_name'],
                    temperature=st.session_state['temperature'],
                    top_p=st.session_state['top_p']
                )
        except Exception as e:
            st.sidebar.error(f"Error initializing model: {str(e)}")
                
    tab1, tab2, tab3= st.tabs(["Home", "ChatBot","LLM Model Card"])
    
    with tab1:
        st.header("Welcome to LISA: LLM Informed Statistical Analysis 🎈")
        st.markdown("LISA is an innovative platform designed to automate your data analysis process using advanced Large Language Models (LLM) for insightful inferences. Whether you're a data enthusiast, researcher, or business analyst, LISA simplifies complex data tasks, providing clear and comprehensible explanations for your data.")
        st.markdown("LISA combines the efficiency of automated data processing with the intelligence of modern language models to deliver a seamless and insightful data analysis experience. Empower your data with LISA!")
        st.divider()
        
        if 'df' not in st.session_state:
            st.session_state["df"] = None
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        
        if uploaded_file is not None:
            st.session_state["df"] = pd.read_csv(uploaded_file)
            st.write("Uploaded data preview:")
            st.write(st.session_state["df"])
            option = st.selectbox("Select an option:", ["Show dataset dimensions", "Display data description", "Verify data integrity", "Summarize numerical data statistics", "Summarize categorical data", "Ask a question about the data"])
            
            if not st.session_state['groq_api_key']:
                st.warning("Please enter your Groq API key in the sidebar to use the analysis features.")
            elif llm is None:
                st.error("Failed to initialize the model. Please check your API key.")
            else:
                if option == "Show dataset dimensions":
                    shape_of_the_data = st.session_state.df.shape
                    response = get_llm_response(llm, 'The shape of the dataset is: {shape}', {'shape': shape_of_the_data})
                    st.write(response)
                    
                elif option == "Display data description":
                    column_description = st.session_state.df.columns.tolist()
                    response = get_llm_response(llm, 'The columns in the dataset are: {columns}', {'columns': column_description})
                    st.write(response)
                    
                elif option == "Verify data integrity":
                    df_check = check(st.session_state.df)
                    st.dataframe(df_check)
                    st.divider()
                    response = get_llm_response(llm, 'The data integrity check results are: {df_check}', {'df_check': df_check})
                    st.write(response)
                    
                elif option == "Summarize numerical data statistics":
                    describe_numerical = st.session_state.df.describe().T
                    st.dataframe(describe_numerical)
                    st.divider()    
                    response = get_llm_response(llm, 'The numerical data statistics are: {stats}', {'stats': describe_numerical})
                    st.write(response)
                    
                elif option == "Summarize categorical data":
                    categorical_df = st.session_state.df.select_dtypes(include=['object'])
                    if categorical_df.empty:
                        st.write("No categorical columns found.")
                        response = get_llm_response(llm, 'There are no categorical columns in this dataset.', {})
                    else:
                        describe_categorical = categorical_df.describe()
                        st.dataframe(describe_categorical)
                        st.divider()
                        response = get_llm_response(llm, 'The categorical data summary is: {summary}', {'summary': describe_categorical})
                    st.write(response)
                
                elif option == "Ask a question about the data":
                    question = st.text_input("Write a question about the data", key="question")
                    if question:
                        attempt = 0
                        max_attempts = 5
                        while attempt < max_attempts:
                            try:
                                # Use Pandas DataFrame as context for SQL query generation
                                context = pd.io.sql.get_schema(st.session_state.df.reset_index(), "df").replace('"', "")
                                input_data = {"context": context, "question": question}
                                formatted_prompt = prompt.format(**input_data)
                                response = groq_infer(llm, formatted_prompt)
                                final = response.replace("`", "").replace("sql", "").strip()
                                
                                # st.write("Generated SQL Query:")
                                # st.code(final)
                                
                                result = sqldf(final, {'df': st.session_state.df})
                                st.write("Answer:")
                                st.dataframe(result)
                                
                                explanation_prompt = f"""
                                Given the context of the dataset from {uploaded_file}, explain the following answer in simple English: Also do not explain the sql query.

                                {result.to_string()}
                                """
                                explanation_response = groq_infer(llm, explanation_prompt)
                                st.write("Explanation:")
                                st.write(explanation_response)
                                break
                            except Exception as e:
                                attempt += 1
                                st.error(f"Attempt {attempt}/{max_attempts} failed. Error: {str(e)}")
                                if attempt == max_attempts:
                                    st.error("Unable to get the correct query after 5 attempts. Please try again or refine your question.")
                                continue
                    else:
                        st.warning("Please enter a question before clicking 'Get Answer'.")
        elif st.session_state["df"] is not None:
            st.write("Uploaded data preview:")
            st.write(st.session_state["df"])
            option = st.selectbox("Select an option:", ["Show dataset dimensions", "Display data description", "Verify data integrity", "Summarize numerical data statistics", "Summarize categorical data", "Ask a question about the data"])
            
            if not st.session_state['groq_api_key']:
                st.warning("Please enter your Groq API key in the sidebar to use the analysis features.")
            elif llm is None:
                st.error("Failed to initialize the model. Please check your API key.")
            else:
                if option == "Show dataset dimensions":
                    shape_of_the_data = st.session_state.df.shape
                    response = get_llm_response(llm, 'The shape of the dataset is: {shape}', {'shape': shape_of_the_data})
                    st.write(response)
                    
                elif option == "Display data description":
                    column_description = st.session_state.df.columns.tolist()
                    response = get_llm_response(llm, 'The columns in the dataset are: {columns}', {'columns': column_description})
                    st.write(response)
                    
                elif option == "Verify data integrity":
                    df_check = check(st.session_state.df)
                    st.dataframe(df_check)
                    st.divider()
                    response = get_llm_response(llm, 'The data integrity check results are: {df_check}', {'df_check': df_check})
                    st.write(response)
                    
                elif option == "Summarize numerical data statistics":
                    describe_numerical = st.session_state.df.describe().T
                    st.dataframe(describe_numerical)
                    st.divider()    
                    response = get_llm_response(llm, 'The numerical data statistics are: {stats}', {'stats': describe_numerical})
                    st.write(response)
                    
                elif option == "Summarize categorical data":
                    categorical_df = st.session_state.df.select_dtypes(include=['object'])
                    if categorical_df.empty:
                        st.write("No categorical columns found.")
                        response = get_llm_response(llm, 'There are no categorical columns in this dataset.', {})
                    else:
                        describe_categorical = categorical_df.describe()
                        st.dataframe(describe_categorical)
                        st.divider()
                        response = get_llm_response(llm, 'The categorical data summary is: {summary}', {'summary': describe_categorical})
                    st.write(response)
                
                elif option == "Ask a question about the data":
                    question = st.text_input("Write a question about the data", key="question")
                    if question:
                        attempt = 0
                        max_attempts = 5
                        while attempt < max_attempts:
                            try:
                                # Use Pandas DataFrame as context for SQL query generation
                                context = pd.io.sql.get_schema(st.session_state.df.reset_index(), "df").replace('"', "")
                                input_data = {"context": context, "question": question}
                                formatted_prompt = prompt.format(**input_data)
                                response = groq_infer(llm, formatted_prompt)
                                final = response.replace("`", "").replace("sql", "").strip()
                                
                                # st.write("Generated SQL Query:")
                                # st.code(final)
                                
                                result = sqldf(final, {'df': st.session_state.df})
                                st.write("Answer:")
                                st.dataframe(result)
                                
                                explanation_prompt = f"""
                                Given the context of the dataset from {uploaded_file}, explain the following answer in simple English: Also do not explain the sql query.

                                {result.to_string()}
                                """
                                explanation_response = groq_infer(llm, explanation_prompt)
                                st.write("Explanation:")
                                st.write(explanation_response)
                                break
                            except Exception as e:
                                attempt += 1
                                st.error(f"Attempt {attempt}/{max_attempts} failed. Error: {str(e)}")
                                if attempt == max_attempts:
                                    st.error("Unable to get the correct query after 5 attempts. Please try again or refine your question.")
                                continue
                    else:
                        st.warning("Please enter a question before clicking 'Get Answer'.")

    with tab2:
        st.markdown("""Our integrated chatbot is available to assist you, providing real-time answers to your data-related queries and enhancing your overall experience with personalized support.""")
        st.markdown("""---""")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        def get_response(query, chat_history, df):
            template = """
            You are a knowledgeable data assistant. Answer the user's question based on the provided dataset and the conversation history. If the data isn't directly related to the question, guide the user on how they might extract relevant insights.

            Dataset information (limit to first 50 rows):
            {df}

            Chat history:
            {chat_history}

            User question: {user_question}

            Make sure to use the dataset to provide specific answers related to the data where appropriate. If the dataset is not relevant, ask for clarification. 
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | llm | StrOutputParser()

            # Ensure that df is passed correctly to the chain
            return chain.stream({
                "chat_history": chat_history,
                "user_question": query,
                "df": df.head(50).to_string() if df is not None else "No data uploaded yet."
            })



        # Display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(message.content)
            else:
                with st.chat_message("AI"):
                    st.markdown(message.content)
                    
        if not st.session_state['groq_api_key']:
            st.warning("Please enter your Groq API key in the sidebar to use the chatbot.")
        elif llm is None:
            st.error("Failed to initialize the model. Please check your API key.")
        else:
            st.write("") 
            user_query = st.chat_input("Type your message here")
            
            if user_query is not None and user_query != "":
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                
                with st.chat_message("Human"):
                    st.markdown(user_query)
                    
                with st.chat_message("AI"):
                    full_response = ""
                    message_placeholder = st.empty()
                    try:
                        for chunk in get_response(user_query, st.session_state.chat_history, st.session_state.get('df')):
                            full_response += chunk
                            message_placeholder.markdown(full_response + "")
                        message_placeholder.markdown(full_response)
                    except Exception as e:
                        error_message = f"An error occurred: {str(e)}. Please make sure you've uploaded a dataset."
                        message_placeholder.error(error_message)
                        full_response = error_message
                
                st.session_state.chat_history.append(AIMessage(content=full_response))
                
    with tab3:
        st.header("LLM Model Card")
        
        st.markdown("In our innovative project LISA (LLM Informed Statistical Analysis), we are harnessing the power of Groq-hosted large language models (LLMs) to revolutionize the way statistical analysis is performed and interpreted. Groq’s platform plays a pivotal role in enabling LISA to deliver accurate, fast, and insightful data analysis by providing access to highly optimized, open-source LLMs that are tailored for complex data processing tasks.")
        
        st.markdown("Groq is the AI infrastructure company that delivers fast AI inference.The LPU™ Inference Engine by Groq is a hardware and software platform that delivers exceptional compute speed, quality, and energy efficiency.")
        
        st.markdown("The table below provides comparision of the performance of different LLM models across various NLP (Natural Language Processing) benchmarks")
        
        data_folder_path = "Data"
        dataframes = {}
        for file_name in os.listdir(data_folder_path):
            if file_name.endswith(".csv"):
                data_file_path = os.path.join(data_folder_path, file_name)
                df = pd.read_csv(data_file_path)
                dataframes[file_name] = df

        model_card = dataframes.get('modelcard.csv')
        if model_card is not None:
            st.dataframe(model_card, hide_index=True)
        else:
            st.error("Model card CSV not found.")
            
        st.markdown("""
    <style>
    ul {
        list-style-type: disc;
        margin-left: 20px;
    }
    </style>
    <ul>
        Here’s what these benchmarks mean:
        <li><b>MMLU (Massive Multitask Language Understanding):</b> A benchmark designed to understand how well a language model can multitask. The model’s performance is assessed across a range of subjects, such as math, computer science, and law.</li>
        <li><b>GPQA (Graduate-Level Google-Proof Q&A):</b> Assesses a model’s ability to answer questions that are challenging for search engines to solve directly. This benchmark evaluates whether the AI can handle questions that usually require human-level research skills.</li>
        <li><b>HumanEval:</b> Assesses how well the model can write code by asking it to perform programming tasks.</li>
        <li><b>GSM-8K:</b> Evaluates the model’s ability to solve math word problems.</li>
        <li><b>MATH:</b> Tests the model’s ability to solve middle school and high school math problems.</li>
    </ul>
    """, unsafe_allow_html=True)
        
        st.info("We've observed that the Gemma2-9B-IT model excels in querying data, while the Llama variants are particularly effective for inferring results.", icon="ℹ️")