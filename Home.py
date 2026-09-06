import streamlit as st
import pandas as pd
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pandasql import sqldf
from functions import check

load_dotenv()

@st.cache_data(ttl=600, show_spinner=False)
def fetch_available_models(api_key: str):
    """Fetch models accessible by this specific Groq API key."""
    if not api_key:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        models_res = client.models.list()
        chat_models = [
            m.id for m in models_res.data
            if "whisper" not in m.id.lower()
        ]
        return chat_models if chat_models else None
    except Exception:
        return None

def get_llm_response(llm, prompt_template, data):
    try:
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            "You are StatBot, an expert statistical analyst. "
            "Explain the output in simple English. Straight away start with your explanations."
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template(prompt_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        formatted_chat_prompt = chat_prompt.format_messages(**data)
        response = llm.invoke(formatted_chat_prompt)
        return response.content
    except Exception as e:
        err_str = str(e)
        if "model_not_found" in err_str or "404" in err_str:
            model_id = getattr(llm, 'model_name', 'selected model')
            return (
                f"**Model Access Error**: The model `{model_id}` is not enabled or accessible on your current Groq plan.\n\n"
                "**Action needed**: Please select another model from the sidebar dropdown (such as `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, or `llama-3.1-8b-instant`)."
            )
        return f"Error generating response: {err_str}"

def groq_infer(llm, prompt):
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return response.content

def clean_sql_query(raw_response: str) -> str:
    """Extract clean SQL from LLM response, handling reasoning tags and markdown code blocks."""
    if not raw_response:
        return ""
    # Strip reasoning tags (e.g. DeepSeek R1)
    cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
    # Extract from markdown code fence if present
    match = re.search(r"```(?:sql)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
    else:
        cleaned = cleaned.replace("`", "").strip()
    # Remove leading SQL: or Query: prefixes if generated
    cleaned = re.sub(r"^(?:SQL|Query|SQLite):\s*", "", cleaned, flags=re.IGNORECASE).strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    return cleaned

def build_schema_context(df: pd.DataFrame) -> str:
    """Build a rich schema description including column types and real sample values."""
    col_details = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_samples = [str(x) for x in df[col].dropna().unique()[:3]]
        samples_str = ", ".join(f"'{s}'" for s in unique_samples) if unique_samples else "None"
        col_details.append(f"- Column: [{col}] | Type: {dtype} | Sample values: [{samples_str}]")
    
    schema_text = "\n".join(col_details)
    preview = df.head(3).to_string(index=False)
    return (
        "Database Engine: SQLite (In-memory table named 'df')\n\n"
        f"Columns & Sample Values:\n{schema_text}\n\n"
        f"First 3 Rows of Data:\n{preview}"
    )

def generate_sql_query(llm, question: str, df: pd.DataFrame, previous_error: str = None, previous_sql: str = None) -> str:
    """Generate or self-correct an SQLite query using schema context and previous error feedback."""
    schema_context = build_schema_context(df)
    
    system_prompt = (
        "You are an expert SQLite data analyst. Your job is to convert natural language questions into single, executable SQLite queries for a table named 'df'.\n\n"
        "STRICT SQLITE RULES:\n"
        "1. The table name is always 'df'.\n"
        "2. Wrap all column names in square brackets, e.g. [Column Name], to avoid errors with spaces or SQLite keywords.\n"
        "3. For text comparisons, use case-insensitive checks where appropriate: LOWER([Column]) = LOWER('value') or [Column] LIKE '%value%'.\n"
        "4. Use only standard SQLite functions (COUNT, SUM, AVG, MIN, MAX, ROUND, strftime, SUBSTR, COALESCE, etc.). Never use PostgreSQL or MySQL-specific functions (no ILIKE, no DATE_TRUNC, no CONCAT).\n"
        "5. Output ONLY the raw SQL query. Do not include markdown explanations, reasoning text, or conversational commentary."
    )
    
    if previous_error and previous_sql:
        user_prompt = (
            f"User Question: {question}\n\n"
            f"{schema_context}\n\n"
            "ATTENTION - PREVIOUS QUERY FAILED:\n"
            f"Failed SQL: {previous_sql}\n"
            f"SQLite Error Message: {previous_error}\n\n"
            "Analyze why the previous query failed, check the available columns and sample values, and output a corrected SQLite query."
        )
    else:
        user_prompt = (
            f"User Question: {question}\n\n"
            f"{schema_context}\n\n"
            "Output only the SQLite query to answer the user question."
        )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    response = llm.invoke(messages)
    return clean_sql_query(response.content)

def load_sidebar():
    # Load sidebar elements for API key and model parameters
    with st.sidebar:
        st.divider()
        with st.sidebar.expander("Get Your API Key Here"):
            st.markdown("## How to use\n"
            "1. Enter your [Groq API key](https://console.groq.com/keys) below\n" 
            "2. Upload a CSV file\n"
            "3. Let LISA do its work\n")
        
        default_api_key = st.session_state.get('groq_api_key') or os.getenv('GROQ_API_KEY', '')
        st.session_state['groq_api_key'] = st.text_input(
            "Enter your Groq API key:",
            type="password",
            placeholder="Paste your Groq API key here (gsk_...)",
            value=default_api_key
        )
        
        api_key = st.session_state['groq_api_key']
        fetched_models = fetch_available_models(api_key) if api_key else None
        
        # Models available on Developer plans / standard tiers
        fallback_models = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
        ]
        
        available_models = fetched_models if fetched_models else fallback_models
        
        current_model = st.session_state.get('model_name', available_models[0])
        model_index = available_models.index(current_model) if current_model in available_models else 0
        
        st.session_state['model_name'] = st.selectbox(
            "Select Model:", 
            available_models, 
            index=model_index
        )
        
        st.session_state['temperature'] = st.slider(
            "Temperature:",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('temperature', 0.5),
            step=0.1
        )
        st.session_state['top_p'] = st.slider(
            "Top-p:",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('top_p', 1.0),
            step=0.25
        )

def get_chatgroq_llm():
    api_key = st.session_state.get('groq_api_key', '')
    if not api_key:
        return None
    try:
        return ChatGroq(
            groq_api_key=api_key,
            model=st.session_state.get('model_name', 'openai/gpt-oss-120b'),
            temperature=st.session_state.get('temperature', 0.5),
            model_kwargs={"top_p": st.session_state.get('top_p', 1.0)}
        )
    except Exception as e:
        st.sidebar.error(f"Error initializing model: {str(e)}")
        return None

def Home():
    llm = get_chatgroq_llm()
    st.divider()
                
    tab1, tab2, tab3 = st.tabs(["Home", "ChatBot", "LLM Model Card"])
    
    with tab1:
        st.header("Welcome to LISA: LLM Informed Statistical Analysis")
        st.markdown("LISA is an innovative platform designed to automate your data analysis process using advanced Large Language Models (LLM) for insightful inferences. Whether you're a data enthusiast, researcher, or business analyst, LISA simplifies complex data tasks, providing clear and comprehensible explanations for your data.")
        st.markdown("LISA combines the efficiency of automated data processing with the intelligence of modern language models to deliver a seamless and insightful data analysis experience. Empower your data with LISA!")
        st.divider()
        
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        if uploaded_file is not None:
            st.session_state["df"] = pd.read_csv(uploaded_file)
            st.session_state["filename"] = uploaded_file.name
        
        if st.session_state.get("df") is not None:
            df = st.session_state["df"]
            filename = st.session_state.get("filename", "uploaded dataset")
            
            st.write("Uploaded data preview:")
            st.dataframe(df.head(10))
            
            option = st.selectbox(
                "Select an option:",
                [
                    "Show dataset dimensions",
                    "Display data description",
                    "Verify data integrity",
                    "Summarize numerical data statistics",
                    "Summarize categorical data",
                    "Ask a question about the data"
                ]
            )
            
            if not st.session_state.get('groq_api_key'):
                st.warning("Please enter your Groq API key in the sidebar to use the analysis features.")
            elif llm is None:
                st.error("Failed to initialize the model. Please check your API key in the sidebar.")
            else:
                if option == "Show dataset dimensions":
                    shape_of_the_data = df.shape
                    with st.spinner("Analyzing dimensions..."):
                        response = get_llm_response(llm, 'The shape of the dataset is: {shape}', {'shape': str(shape_of_the_data)})
                    if response.startswith("**Model Access Error**"):
                        st.error(response)
                    else:
                        st.write(response)
                    
                elif option == "Display data description":
                    column_description = df.columns.tolist()
                    with st.spinner("Analyzing columns..."):
                        response = get_llm_response(llm, 'The columns in the dataset are: {columns}', {'columns': str(column_description)})
                    if response.startswith("**Model Access Error**"):
                        st.error(response)
                    else:
                        st.write(response)
                    
                elif option == "Verify data integrity":
                    df_check = check(df)
                    st.dataframe(df_check)
                    st.divider()
                    with st.spinner("Verifying integrity..."):
                        response = get_llm_response(llm, 'The data integrity check results are:\n{df_check}', {'df_check': df_check.to_string()})
                    if response.startswith("**Model Access Error**"):
                        st.error(response)
                    else:
                        st.write(response)
                    
                elif option == "Summarize numerical data statistics":
                    describe_numerical = df.describe().T
                    st.dataframe(describe_numerical)
                    st.divider()    
                    with st.spinner("Summarizing statistics..."):
                        response = get_llm_response(llm, 'The numerical data statistics are:\n{stats}', {'stats': describe_numerical.to_string()})
                    if response.startswith("**Model Access Error**"):
                        st.error(response)
                    else:
                        st.write(response)
                    
                elif option == "Summarize categorical data":
                    categorical_df = df.select_dtypes(include=['object', 'category'])
                    if categorical_df.empty:
                        st.info("No categorical columns found in this dataset.")
                    else:
                        describe_categorical = categorical_df.describe()
                        st.dataframe(describe_categorical)
                        st.divider()
                        with st.spinner("Summarizing categorical data..."):
                            response = get_llm_response(llm, 'The categorical data summary is:\n{summary}', {'summary': describe_categorical.to_string()})
                        if response.startswith("**Model Access Error**"):
                            st.error(response)
                        else:
                            st.write(response)
                
                elif option == "Ask a question about the data":
                    st.markdown("### Query Your Data with Natural Language")
                    st.write("Ask questions about this dataset in plain English. LISA will generate an optimized SQLite query, execute it, and explain the findings.")
                    
                    question = st.text_input("Ask a question about the data:", key="home_question_input", placeholder="e.g., How many records are there where age > 30?")
                    
                    if question:
                        max_attempts = 4
                        attempt = 0
                        success = False
                        previous_error = None
                        previous_sql = None
                        final_sql = ""
                        result = None
                        
                        with st.spinner("Generating and executing SQLite query..."):
                            while attempt < max_attempts and not success:
                                attempt += 1
                                try:
                                    final_sql = generate_sql_query(
                                        llm=llm,
                                        question=question,
                                        df=df,
                                        previous_error=previous_error,
                                        previous_sql=previous_sql
                                    )
                                    result = sqldf(final_sql, {'df': df})
                                    success = True
                                except Exception as e:
                                    err_str = str(e)
                                    if "model_not_found" in err_str or "404" in err_str:
                                        model_id = getattr(llm, 'model_name', 'selected model')
                                        st.error(f"**Model Access Error**: The model `{model_id}` is not accessible on your Groq plan.")
                                        st.info("Please switch to an accessible model from the sidebar (such as `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, or `llama-3.1-8b-instant`).")
                                        break
                                    previous_error = err_str
                                    previous_sql = final_sql
                        
                        if success and result is not None:
                            # Interactive SQL Inspector & Editor
                            with st.expander("View & Edit Generated SQL Query", expanded=False):
                                st.caption("LISA converted your question into the following SQLite query:")
                                user_sql = st.text_area("SQL Query:", value=final_sql, height=80, key=f"user_sql_editor_{attempt}")
                                if st.button("Run Modified SQL", key="rerun_sql_btn"):
                                    try:
                                        result = sqldf(user_sql, {'df': df})
                                        st.success("Custom query executed successfully!")
                                    except Exception as sql_err:
                                        st.error(f"SQL Execution Error: {str(sql_err)}")
                            
                            st.subheader("Results:")
                            if result.empty:
                                st.info("The query executed successfully but returned 0 rows. (Tip: check your filter conditions or casing).")
                            else:
                                st.dataframe(result)
                                st.caption(f"Returned {len(result)} row(s)")
                                
                                # Explanation with safe row slicing
                                with st.spinner("Generating explanation..."):
                                    display_slice = result.head(20).to_string()
                                    row_info = f"Showing first 20 of {len(result)} rows" if len(result) > 20 else f"Total {len(result)} rows"
                                    explanation_prompt = f"""
                                    Given the context of the dataset '{filename}', explain the following query result in simple English:

                                    Question: {question}
                                    {row_info}:
                                    {display_slice}

                                    Summarize the main takeaways clearly and concisely without technical SQL syntax.
                                    """
                                    explanation_response = groq_infer(llm, explanation_prompt)
                                    st.markdown("### Explanation:")
                                    st.write(explanation_response)
                        elif not success and previous_error:
                            st.error(f"Unable to execute query after {max_attempts} attempts.")
                            with st.expander("View Diagnostic Details"):
                                st.code(f"Last Attempted SQL:\n{previous_sql}\n\nSQLite Error:\n{previous_error}", language="sql")
                            st.info("Tip: Try mentioning exact column names from the dataset preview above.")
                    else:
                        st.info("Please enter a question above to get an answer.")

    with tab2:
        st.markdown("Our integrated chatbot is available to assist you, providing real-time answers to your data-related queries and enhancing your overall experience with personalized support.")
        st.divider()

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.markdown(message.content)
                    
        if not st.session_state.get('groq_api_key'):
            st.warning("Please enter your Groq API key in the sidebar to use the chatbot.")
        elif llm is None:
            st.error("Failed to initialize the model. Please check your API key in the sidebar.")
        else:
            user_query = st.chat_input("Type your message here")
            if user_query:
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                with st.chat_message("Human"):
                    st.markdown(user_query)
                    
                with st.chat_message("AI"):
                    df = st.session_state.get('df')
                    df_context = df.head(50).to_string() if df is not None else "No dataset uploaded yet."
                    system_prompt = (
                        "You are a knowledgeable data assistant. Answer the user's question based on the provided dataset and conversation history. "
                        "If the data isn't directly related to the question, guide the user on how they might extract relevant insights.\n\n"
                        f"Dataset information (limit to first 50 rows):\n{df_context}"
                    )
                    messages = [SystemMessage(content=system_prompt)] + st.session_state.chat_history
                    
                    try:
                        stream = llm.stream(messages)
                        full_response = st.write_stream(stream)
                        st.session_state.chat_history.append(AIMessage(content=full_response))
                    except Exception as e:
                        err = str(e)
                        if "model_not_found" in err or "404" in err:
                            model_id = getattr(llm, 'model_name', 'selected model')
                            err = f"Model `{model_id}` is not accessible on your Groq plan. Please select a supported model in the sidebar (such as `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, or `llama-3.1-8b-instant`)."
                        else:
                            err = f"An error occurred: {err}"
                        st.error(err)
                        st.session_state.chat_history.append(AIMessage(content=err))

    with tab3:
        st.header("LLM Model Card")
        
        st.markdown("In our innovative project LISA (LLM Informed Statistical Analysis), we are harnessing the power of Groq-hosted large language models (LLMs) to revolutionize the way statistical analysis is performed and interpreted. Groq’s platform plays a pivotal role in enabling LISA to deliver accurate, fast, and insightful data analysis by providing access to highly optimized, open-source LLMs that are tailored for complex data processing tasks.")
        
        st.markdown("Groq is the AI infrastructure company that delivers fast AI inference. The LPU™ Inference Engine by Groq is a hardware and software platform that delivers exceptional compute speed, quality, and energy efficiency.")
        
        st.markdown("The table below provides comparison of the performance of different LLM models across various NLP (Natural Language Processing) benchmarks:")
        
        data_folder_path = "Data"
        model_card_path = os.path.join(data_folder_path, "modelcard.csv")
        if os.path.exists(model_card_path):
            model_card = pd.read_csv(model_card_path)
            st.dataframe(model_card, hide_index=True)
        else:
            st.error("Model card CSV not found.")
            
        st.markdown("""
        <ul>
            <b>Here’s what these benchmarks mean:</b>
            <li><b>MMLU (Massive Multitask Language Understanding):</b> A benchmark designed to understand how well a language model can multitask. The model’s performance is assessed across a range of subjects, such as math, computer science, and law.</li>
            <li><b>GPQA (Graduate-Level Google-Proof Q&A):</b> Assesses a model’s ability to answer questions that are challenging for search engines to solve directly. This benchmark evaluates whether the AI can handle questions that usually require human-level research skills.</li>
            <li><b>HumanEval:</b> Assesses how well the model can write code by asking it to perform programming tasks.</li>
            <li><b>GSM-8K:</b> Evaluates the model’s ability to solve math word problems.</li>
            <li><b>MATH:</b> Tests the model’s ability to solve middle school and high school math problems.</li>
        </ul>
        """, unsafe_allow_html=True)
        
        st.info("We recommend GPT OSS 120B and Llama 3.3 70B for deep reasoning and statistical analysis, and GPT OSS 20B or Llama 3.1 8B for fast response times.")
