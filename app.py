import streamlit as st
import pandas as pd
from st_aggrid import AgGrid
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage
from langchain.schema import AIMessage
from langchain_core.output_parsers import StrOutputParser
from pandasql import sqldf
from functions import check
from dotenv import load_dotenv


st.set_page_config(page_title="LISA : LLM Informed Statistical Analysis", page_icon=":books:", layout="wide")

# Initialize session state to store the dataframe
if 'df' not in st.session_state:
    st.session_state.df = None

tab1, tab2, tab3 = st.tabs(["Home", "ChatBot","Data Preparation"])

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

with st.sidebar:
    with st.sidebar.expander("Get Your API Key Here"):
        st.markdown("## How to use\n"
            "1. Enter your [Groq API key](https://console.groq.com/keys) below🔑\n" 
            "2. Upload a CSV file📄\n"
            "3. Let LISA do its work!!!💬\n")
    
    groq_api_key = st.text_input("Enter your Groq API key:", type="password",
            placeholder="Paste your Groq API key here (gsk_...)",
            help="You can get your API key from https://console.groq.com/keys")
    
    with st.sidebar.expander("Model Parameters"):
        model_name = st.selectbox("Select Model:", ["llama-3.1-70b-versatile","llama-3.1-8b-instant","llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it", "gemma2-9b-it"])
        temperature = st.slider("Temperature: It determines whether the output is more random, creative or more predictable.", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        top_p = st.slider("Top-p: It determines the cumulative probability distribution used for sampling the next token in the generated response", min_value=0.0, max_value=1.0, value=1.0, step=0.25)

    st.divider()

    # Initialize LLM only if API key is provided
    llm = None
    if groq_api_key:
        try:
            llm = ChatGroq(
                groq_api_key=groq_api_key, 
                model_name=model_name,
                temperature=temperature,
                top_p=top_p
            )
        except Exception as e:
            st.sidebar.error(f"Error initializing model: {str(e)}")

# Define the prompt template
template = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. Don't add \n characters.

You must output the SQL query that answers the question in a single line.

### Input:
`{question}`

### Context:
`{context}`

### Response:
"""

prompt = PromptTemplate.from_template(template=template)

with tab1:
    st.header("Welcome to LISA: LLM Informed Statistical Analysis 🎈", divider='rainbow')
    st.markdown("LISA is an innovative platform designed to automate your data analysis process using advanced Large Language Models (LLM) for insightful inferences. Whether you're a data enthusiast, researcher, or business analyst, LISA simplifies complex data tasks, providing clear and comprehensible explanations for your data.")
    st.markdown("LISA combines the efficiency of automated data processing with the intelligence of modern language models to deliver a seamless and insightful data analysis experience. Empower your data with LISA!")
    st.divider()
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file, encoding="latin1")
        st.session_state.df.columns = st.session_state.df.columns.str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
        st.dataframe(st.session_state.df)
        st.divider()

        option = st.selectbox("Select an option:", ["Show dataset dimensions", "Display data description", "Verify data integrity", "Summarize numerical data statistics", "Summarize categorical data", "Ask a question about the data"])
        
        if not groq_api_key:
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
                response = get_llm_response(llm, 'The data integrity check results are: {df_check}', {'df_check': df_check})
                st.write(response)
                
            elif option == "Summarize numerical data statistics":
                describe_numerical = st.session_state.df.describe().T
                st.dataframe(describe_numerical)
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
                            
                            # Execute the SQL query on the DataFrame
                            result = sqldf(final, {'df': st.session_state.df})
                            st.write("Answer:")
                            st.dataframe(result)
                            
                            # Pass the output dataframe and question to the LLM for explanation
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
                    
                    
# Assuming you've already set up your tabs and this is inside the tab2 section
with tab2:
    st.markdown("""Our integrated chatbot is available to assist you, providing real-time answers to your data-related queries and enhancing your overall experience with personalized support.""")
    st.markdown("""---""")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def get_response(query, chat_history, df):
        template = """
        You are a helpful assistant. Answer the following question based on the provided dataset:

        Dataset information:
        {df}

        Chat history:
        {chat_history}

        User question: {user_question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        return chain.stream({
            "chat_history": chat_history,
            "user_question": query,
            "df": df.head().to_string()  # Using head() to limit the data shown. Adjust as needed.
        })

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message("AI"):
                st.markdown(message.content)
                
    if not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar to use the chatbot.")
    elif llm is None:
        st.error("Failed to initialize the model. Please check your API key.")
    else:
        user_query = st.chat_input("Type your message here")
        
        if user_query is not None and user_query != "":
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            
            with st.chat_message("Human"):
                st.markdown(user_query)
                
            with st.chat_message("AI"):
                full_response = ""
                message_placeholder = st.empty()
                for chunk in get_response(user_query, st.session_state.chat_history, st.session_state.df):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            
            st.session_state.chat_history.append(AIMessage(content=full_response))