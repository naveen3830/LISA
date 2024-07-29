import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from st_aggrid import AgGrid
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pandasql import sqldf
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from dotenv import load_dotenv

st.set_page_config(page_title="LISA : LLM Informed Statistical Analysis", page_icon=":books:", layout="wide")

# Initialize session state to store the dataframe
if 'df' not in st.session_state:
    st.session_state.df = None

tab1, tab2, tab3, tab4 = st.tabs(["Home", "Data Analysis", "ChatBot", "Data Preparation"])

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
        temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        top_p = st.slider("Top-p:", min_value=0.0, max_value=1.0, value=1.0, step=0.25)

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

    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file, encoding="latin1")
        st.session_state.df.columns = st.session_state.df.columns.str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
        st.sidebar.success("File uploaded successfully!")

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
    
    if st.session_state.df is not None:
        st.dataframe(st.session_state.df)
    else:
        st.info("Please upload a CSV file in the sidebar to get started.")

with tab2:
    if st.session_state.df is not None:
        st.header("Data Analysis")
        
        option = st.selectbox("Select an option:", ["Show dataset dimensions", "Display data description", "Verify data integrity", "Summarize numerical data statistics", "Summarize categorical data", "Visualize data distribution", "Correlation analysis", "Ask a question about the data"])
        
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
                missing_values = st.session_state.df.isnull().sum()
                duplicate_rows = st.session_state.df.duplicated().sum()
                st.write(f"Missing values:\n{missing_values}")
                st.write(f"Number of duplicate rows: {duplicate_rows}")
                response = get_llm_response(llm, 'The data integrity check results are: Missing values: {missing_values}, Duplicate rows: {duplicate_rows}', {'missing_values': missing_values, 'duplicate_rows': duplicate_rows})
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
            
            elif option == "Visualize data distribution":
                numeric_columns = st.session_state.df.select_dtypes(include=[np.number]).columns
                selected_column = st.selectbox("Select a numeric column to visualize:", numeric_columns)
                fig = px.histogram(st.session_state.df, x=selected_column, title=f"Distribution of {selected_column}")
                st.plotly_chart(fig)
                
            elif option == "Correlation analysis":
                numeric_df = st.session_state.df.select_dtypes(include=[np.number])
                corr_matrix = numeric_df.corr()
                fig = px.imshow(corr_matrix, title="Correlation Heatmap")
                st.plotly_chart(fig)
                response = get_llm_response(llm, 'Analyze the correlation matrix and provide insights: {corr_matrix}', {'corr_matrix': corr_matrix})
                st.write(response)
            
            elif option == "Ask a question about the data":
                question = st.text_input("Write a question about the data", key="question")
                if question:
                    attempt = 0
                    max_attempts = 5
                    while attempt < max_attempts:
                        try:
                            context = pd.io.sql.get_schema(st.session_state.df.reset_index(), "df").replace('"', "")
                            input_data = {"context": context, "question": question}
                            formatted_prompt = prompt.format(**input_data)
                            response = groq_infer(llm, formatted_prompt)
                            final = response.replace("`", "").replace("sql", "").strip()
                            
                            st.write("Generated SQL Query:")
                            st.code(final)
                            
                            result = sqldf(final, {'df': st.session_state.df})
                            st.write("Answer:")
                            st.dataframe(result)
                            
                            explanation_prompt = f"""
                            Given the context of the dataset, explain the following answer in simple English: Also do not explain the sql query.

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
    else:
        st.info("Please upload a CSV file in the sidebar to perform data analysis.")

with tab3:
    st.header("ChatBot")
    st.markdown("""Our integrated chatbot is available to assist you, providing real-time answers to your data-related queries and enhancing your overall experience with personalized support.""")
    st.markdown("""---""")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def get_response(query, chat_history):
        template = """
        You are a helpful assistant. Answer the following the user asks:
        
        Chat history:{chat_history}
        user question:{user_question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        return chain.stream({
            "chat_history": chat_history,"user_question": query})

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
            st.session_state.chat_history.append(HumanMessage(user_query))
            
            with st.chat_message("Human"):
                st.markdown(user_query)
                
            with st.chat_message("AI"):
                ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history))
                
            st.session_state.chat_history.append(ai_response)

with tab4:
    st.header("Data Preparation")
    
    if st.session_state.df is not None:
        st.subheader("Current Data Overview")
        st.write(f"Shape: {st.session_state.df.shape}")
        st.write(f"Columns: {', '.join(st.session_state.df.columns)}")

        # Column selection for cleaning
        column = st.selectbox("Select a column to clean", st.session_state.df.columns)

        # Data cleaning options
        st.subheader(f"Cleaning options for {column}")

        # Handle missing values
        if st.session_state.df[column].isnull().sum() > 0:
            missing_action = st.radio(
                f"Handle missing values in {column}",
                ("Keep", "Drop", "Fill with mean", "Fill with median", "Fill with mode", "Fill with custom value")
            )
            if missing_action == "Drop":
                st.session_state.df = st.session_state.df.dropna(subset=[column])
            elif missing_action == "Fill with mean":
                st.session_state.df[column] = st.session_state.df[column].fillna(st.session_state.df[column].mean())
            elif missing_action == "Fill with median":
                st.session_state.df[column] = st.session_state.df[column].fillna(st.session_state.df[column].median())
            elif missing_action == "Fill with mode":
                st.session_state.df[column] = st.session_state.df[column].fillna(st.session_state.df[column].mode()[0])
            elif missing_action == "Fill with custom value":
                custom_value = st.text_input(f"Enter custom value for {column}")
                if custom_value:
                    st.session_state.df[column] = st.session_state.df[column].fillna(custom_value)

        # Handle outliers (for numeric columns)
        if pd.api.types.is_numeric_dtype(st.session_state.df[column]):
            st.subheader(f"Outlier detection for {column}")
            q1 = st.session_state.df[column].quantile(0.25)
            q3 = st.session_state.df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            outliers = st.session_state.df[(st.session_state.df[column] < lower_bound) | (st.session_state.df[column] > upper_bound)]
            
            st.write(f"Number of outliers detected: {len(outliers)}")
            if len(outliers) > 0:
                outlier_action = st.radio(
                    f"Handle outliers in {column}",
                    ("Keep", "Remove", "Cap")
                )
                if outlier_action == "Remove":
                    st.session_state.df = st.session_state.df[(st.session_state.df[column] >= lower_bound) & (st.session_state.df[column] <= upper_bound)]
                elif outlier_action == "Cap":
                    st.session_state.df[column] = st.session_state.df[column].clip(lower_bound, upper_bound)

        # Data transformation
        st.subheader(f"Transform {column}")
        transform_action = st.selectbox(
            f"Apply transformation to {column}",
            ("None", "Log", "Square root", "Min-Max scaling", "Z-score normalization")
        )
        if transform_action == "Log":
            st.session_state.df[f"{column}_log"] = np.log1p(st.session_state.df[column])
        elif transform_action == "Square root":
            st.session_state.df[f"{column}_sqrt"] = np.sqrt(st.session_state.df[column])
        elif transform_action == "Min-Max scaling":
            st.session_state.df[f"{column}_scaled"] = (st.session_state.df[column] - st.session_state.df[column].min()) / (st.session_state.df[column].max() - st.session_state.df[column].min())
        elif transform_action == "Z-score normalization":
            st.session_state.df[f"{column}_normalized"] = (st.session_state.df[column] - st.session_state.df[column].mean()) / st.session_state.df[column].std()

        # Feature engineering
        st.subheader("Feature Engineering")
        if st.checkbox("Create interaction features"):
            numeric_columns = st.session_state.df.select_dtypes(include=[np.number]).columns
            col1, col2 = st.multiselect("Select two columns for interaction", numeric_columns, max_selections=2)
            if len(col1) == 2:
                st.session_state.df[f"{col1[0]}_{col1[1]}_interaction"] = st.session_state.df[col1[0]] * st.session_state.df[col1[1]]
                st.success(f"Created interaction feature: {col1[0]}_{col1[1]}_interaction")

        if st.checkbox("Create polynomial features"):
            numeric_columns = st.session_state.df.select_dtypes(include=[np.number]).columns
            poly_column = st.selectbox("Select a column for polynomial features", numeric_columns)
            poly_degree = st.slider("Select polynomial degree", min_value=2, max_value=5, value=2)
            for degree in range(2, poly_degree + 1):
                st.session_state.df[f"{poly_column}_poly_{degree}"] = st.session_state.df[poly_column] ** degree
            st.success(f"Created polynomial features for {poly_column} up to degree {poly_degree}")

        # Display updated dataframe
        st.subheader("Updated Data")
        st.write(st.session_state.df)

        # Data visualization
        st.subheader("Data Visualization")
        if pd.api.types.is_numeric_dtype(st.session_state.df[column]):
            fig = px.histogram(st.session_state.df, x=column)
            st.plotly_chart(fig)

        # Generate pandas profiling report
        if st.button("Generate Detailed Report"):
            pr = ProfileReport(st.session_state.df, explorative=True)
            st_profile_report(pr)

        # Option to download cleaned data
        st.download_button(
            label="Download cleaned data as CSV",
            data=st.session_state.df.to_csv(index=False).encode('utf-8'),
            file_name='cleaned_data.csv',
            mime='text/csv',
        )
    else:
        st.info("Please upload a CSV file in the sidebar to prepare your data.")

if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.markdown("Created with ❤️ by Your Name")
    st.sidebar.markdown("[GitHub Repository](https://github.com/yourusername/lisa)")