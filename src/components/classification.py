import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import plotly.graph_objects as go
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

def get_llm_response(llm, accuracy=None, confusion_matrix=None, classification_report=None):
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        "You are StatBot, an expert statistical analyst. "
        "Explain the output in simple English."
    )
    human_message_prompt_template = """
    {accuracy_section}
    {confusion_matrix_section}
    {classification_report_section}
    
    Based on these results, explain the performance of the model in a crisp manner do not overdo it. Discuss any relevant insights as well.
    """

    accuracy_section = f"The accuracy of the model is {accuracy:.3f}." if accuracy is not None else ""
    confusion_matrix_section = f"The confusion matrix is as follows:\n{confusion_matrix}" if confusion_matrix is not None else ""
    classification_report_section = f"The classification report is as follows:\n{classification_report}" if classification_report is not None else ""
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_message_prompt_template
    )
    
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    formatted_chat_prompt = chat_prompt.format_messages(
        accuracy_section=accuracy_section,
        confusion_matrix_section=confusion_matrix_section,
        classification_report_section=classification_report_section
    )
    response = llm.invoke(formatted_chat_prompt)

    return response.content


# Function to perform cross-validation
def perform_cross_validation(X, y, model, cv=5):
    cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42))
    return cv_scores.mean(), cv_scores.std()

# Function to plot class distribution
def plot_class_distribution(y, title):
    class_counts = y.value_counts().reset_index()
    class_counts.columns = ['Class', 'Count']
    fig = px.bar(class_counts, x='Class', y='Count', title=title)
    return fig

def create_preprocessing_pipeline(X):
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor, categorical_features, numeric_features

# Main function for classification
def classify():
    # Page title and introduction
    st.header('🔍 Classification Model Builder',divider='grey')
    st.write("""
    Welcome to the Classification Model Builder! This app allows you to build and evaluate various classification models on your dataset.
    You can select the features, choose from a variety of algorithms, handle class imbalances, and evaluate the model's performance.
    """)
    
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

    if 'df' not in st.session_state:
        st.write("Please upload a CSV file in the Data Upload tab to begin.")
        return

    data = st.session_state.df
    st.divider()
    st.subheader("Dataset Overview")
    st.write("Here are the first few rows of your dataset:")
    st.dataframe(data.head())
    
    # # Display categorical variables before encoding
    # st.subheader("📊 Categorical Variables Before Encoding")
    # categorical_features = data.select_dtypes(include=['object', 'category']).columns
    # st.write(categorical_features.tolist())

    st.divider()

    # Feature selection with columns
    col1, col2 = st.columns(2)
    
    with col1:
        dependent_feature = st.selectbox("Select Dependent Variable", data.columns)
    with col2:
        df1 = data.drop(dependent_feature, axis=1)
        independent_features = st.multiselect("Select Independent Variables", df1.columns)

    st.divider()
    # Model selection with algorithm-specific parameters
    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_algorithm = st.selectbox(
            'Select Classification Algorithm',
            ('Logistic Regression', 'SVM', 'KNN', 'AdaBoost', 'Random Forest', 'XGBoost')
        )

    # Hyperparameter tuning based on selected algorithm
    if selected_algorithm == 'Logistic Regression':
        with col2:
            C = st.slider('C (Inverse of regularization strength)', 0.01, 10.0, 1.0)
            max_iter = st.slider('Max Iterations', 100, 500, 200)
    elif selected_algorithm == 'SVM':
        with col2:
            C = st.slider('C (Regularization parameter)', 0.01, 10.0, 1.0)
            kernel = st.selectbox('Kernel', ('linear', 'rbf', 'poly', 'sigmoid'))
            gamma = st.slider('Gamma (Kernel coefficient)', 0.0001, 1.0, 0.1)
    elif selected_algorithm == 'KNN':
        with col2:
            n_neighbors = st.slider('Number of Neighbors (K)', 1, 20, 5)
            algorithm = st.selectbox('Algorithm', ('auto', 'ball_tree', 'kd_tree', 'brute'))
    elif selected_algorithm == 'AdaBoost':
        with col2:
            n_estimators = st.slider('Number of Estimators', 50, 200, 50)
            learning_rate = st.slider('Learning Rate', 0.01, 2.0, 1.0)
    elif selected_algorithm == 'Random Forest':
        with col2:
            n_estimators = st.slider('Number of Estimators', 50, 200, 100)
            max_depth = st.slider('Max Depth', 1, 50, 10)
    else:  # XGBoost
        with col2:
            n_estimators = st.slider('Number of Estimators', 50, 200, 100)
            learning_rate = st.slider('Learning Rate', 0.01, 0.5, 0.1)
            max_depth = st.slider('Max Depth', 1, 15, 6)

    st.divider()
    # Imbalance handling
    st.subheader("Imbalance Handling")
    imbalance_method = st.selectbox(
        'Select Imbalance Handling Method',
        ('None', 'Random Oversampling', 'Random Undersampling', 'SMOTE', 'Class Weight')
    )

    st.divider()

    if st.button("🚀 Train and Evaluate Model"):
        if independent_features and dependent_feature:
            X = data[independent_features]
            y = data[dependent_feature]

            st.write('**Dataset Shape:**', X.shape)
            st.write('**Number of Classes:**', len(np.unique(y)))

            # Create preprocessing pipeline
            preprocessor, _, _ = create_preprocessing_pipeline(X)

            # Class distribution plots before imbalance handling
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_class_distribution(y, "Class Distribution (Before Imbalance Handling)"))

            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            X_train_preprocessed = preprocessor.fit_transform(X_train)
            X_test_preprocessed = preprocessor.transform(X_test)

            # Handle class imbalance
            if imbalance_method == 'Random Oversampling':
                ros = RandomOverSampler(random_state=42)
                X_train_resampled, y_train_resampled = ros.fit_resample(X_train_preprocessed, y_train)
            elif imbalance_method == 'Random Undersampling':
                rus = RandomUnderSampler(random_state=42)
                X_train_resampled, y_train_resampled = rus.fit_resample(X_train_preprocessed, y_train)
            elif imbalance_method == 'SMOTE':
                smote = SMOTE(random_state=42)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)
            else:
                X_train_resampled, y_train_resampled = X_train_preprocessed, y_train

            if imbalance_method != 'None':
                with col2:
                    st.plotly_chart(plot_class_distribution(pd.Series(y_train_resampled), "Class Distribution (After Imbalance Handling)"))

            # Initialize and train model
            if selected_algorithm == 'Logistic Regression':
                model = LogisticRegression(C=C, max_iter=max_iter, class_weight='balanced' if imbalance_method == 'Class Weight' else None)
            elif selected_algorithm == 'SVM':
                model = SVC(C=C, kernel=kernel, gamma=gamma, class_weight='balanced' if imbalance_method == 'Class Weight' else None)
            elif selected_algorithm == 'KNN':
                model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
            elif selected_algorithm == 'AdaBoost':
                model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
            elif selected_algorithm == 'Random Forest':
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced' if imbalance_method == 'Class Weight' else None, random_state=42)
            else:  # XGBoost
                model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)

            model.fit(X_train_resampled, y_train_resampled)
            y_pred = model.predict(X_test_preprocessed)
            
            # Display Model Performance
            st.header("Model Performance",divider='grey')
            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            st.info(f"Test Accuracy: {accuracy:.3f}")
            
            if llm:
                response_accuracy = get_llm_response(llm, accuracy=accuracy, confusion_matrix=None, classification_report=None)
                st.markdown("## Explanation from LLM for Accuracy:")
                st.markdown(response_accuracy)

            # # Cross-Validation
            # st.write("### Cross-Validation Performance")
            # mean_cv_score, std_cv_score = perform_cross_validation(X_train_resampled, y_train_resampled, model)
            # st.write(f"Mean CV Score: {mean_cv_score:.3f} ± {std_cv_score:.3f}")

            # # LLM explanation for cross-validation
            # if llm:
            #     response_cv = get_llm_response(llm, accuracy=None, confusion_matrix=None, classification_report=None, mean_cv_score=mean_cv_score, std_cv_score=std_cv_score)
            #     st.markdown("**Explanation from LLM for Cross-Validation:**")
            #     st.markdown(response_cv)

            st.header("Confusion Matrix",divider='grey')
            conf_matrix = confusion_matrix(y_test, y_pred)
            labels = np.arange(conf_matrix.shape[0])

            fig = go.Figure(data=go.Heatmap(
                z=conf_matrix,
                x=labels, 
                y=labels,  
                colorscale='Blues',
                zmin=0,
                zmax=conf_matrix.max(),
                colorbar=dict(title='Count'),
                text=conf_matrix, 
                texttemplate="%{text}", 
                textfont=dict(size=18, color='black')  
            ))

            fig.update_layout(
                xaxis_title='Predicted Labels',
                yaxis_title='True Labels'
            )

            st.plotly_chart(fig)

            if llm:
                response_cm = get_llm_response(llm, confusion_matrix=conf_matrix, classification_report=None)
                st.divider()
                st.markdown("## Explanation from LLM for Confusion Matrix:")
                st.markdown(response_cm)
            
            def display_classification_report_as_df(y_true, y_pred):
                report = classification_report(y_true, y_pred, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                df_report = df_report[['precision', 'recall', 'f1-score', 'support']]
                df_report.index.name = 'Class Label'
                
                return df_report

            st.header("Classification Report",divider='grey')
            df_report = display_classification_report_as_df(y_test, y_pred)
            st.dataframe(df_report, use_container_width=True)

            if llm:
                response_cr = get_llm_response(llm, classification_report=df_report)
                st.divider()
                st.markdown("## Explanation from LLM for Classification Report:")
                st.markdown(response_cr)
                st.divider()
    