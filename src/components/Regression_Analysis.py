import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
from statsmodels.stats.stattools import durbin_watson
import plotly.graph_objs as go
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Initialize session state variables
if 'llm' not in st.session_state:
    st.session_state.llm = None

def initialize_llm():
    if st.session_state.get('groq_api_key') and not st.session_state.llm:
        try:
            st.session_state.llm = ChatGroq(
                groq_api_key=st.session_state['groq_api_key'], 
                model_name=st.session_state['model_name'],
                temperature=st.session_state['temperature'],
                top_p=st.session_state['top_p']
            )
        except Exception as e:
            st.sidebar.error(f"Error initializing model: {str(e)}")

def get_llm_response(r2_score=None, mae=None, rmse=None, mse=None, summary=None):
    if not st.session_state.llm:
        initialize_llm()
    
    if not st.session_state.llm:
        return "LLM not initialized. Please check your API key and settings."
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        "You are StatBot, an expert statistical analyst. "
        "Explain the output in simple English."
    )
    
    human_message_prompt_template = """
    {r2_score_section}
    {mae_section}
    {rmse_section}
    {mse_section}
    {summary_section}
    
    Based on these results, explain the performance of the model in a crisp manner. Do not overdo it. Discuss any relevant insights as well.
    """

    r2_score_section = f"The r2 score of the model is {r2_score:.3f}." if r2_score is not None else ""
    mae_section = f"The Mean Absolute Error (MAE) of the model is {mae:.3f}." if mae is not None else ""
    rmse_section = f"The Root Mean Squared Error (RMSE) of the model is {rmse:.3f}." if rmse is not None else ""
    mse_section = f"The Mean Squared Error (MSE) of the model is {mse:.3f}." if mse is not None else ""
    summary_section = f"Key model summary insights:\n{summary}" if summary is not None else ""

    human_message_prompt = HumanMessagePromptTemplate.from_template(human_message_prompt_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    
    formatted_chat_prompt = chat_prompt.format_messages(
        r2_score_section=r2_score_section,
        mae_section=mae_section,
        rmse_section=rmse_section,
        mse_section=mse_section,
        summary_section=summary_section
    )

    response = st.session_state.llm.invoke(formatted_chat_prompt)
    return response.content

def regression_analysis():
    if 'df' in st.session_state and st.session_state.df is not None:
        analysis()

def preprocess_data(data, independent_vars):
    st.subheader("🔧 Data Preprocessing", divider='gray')
    
    categorical_vars = data[independent_vars].select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_vars = data[independent_vars].select_dtypes(exclude=['object', 'category']).columns.tolist()
    data_encoded = pd.get_dummies(data, columns=categorical_vars)

    new_independent_vars = numeric_vars + list(data_encoded.columns[len(data.columns) - len(categorical_vars):])
    
    st.info(f"""
    Preprocessing Results:
    - Categorical Variables: {', '.join(categorical_vars) or 'None'}
    - Numeric Variables: {', '.join(numeric_vars) or 'None'}
    - Total Features after Encoding: {len(new_independent_vars)}
    """)
    
    return data_encoded, categorical_vars, new_independent_vars

def analysis():
    initialize_llm()  # Initialize LLM at the start of analysis
    st.write("""
    Welcome to the Regression Model Builder! This app allows you to build and evaluate various regression models on your dataset.
    You can select the features, choose from a variety of algorithms, handle class imbalances, and evaluate the model's performance.
    """)
            
    st.subheader("📊 Dataset Overview",divider='gray')
    st.write("Here are the first few rows of your dataset:")
    st.write(st.session_state.df.head())
    st.divider()

    st.subheader("Feature Selection")
    dependent_features = st.selectbox("Select a feature for the Dependent variable:", st.session_state.df.columns)
    df1 = st.session_state.df.drop(dependent_features, axis=1)
    independent_features = st.multiselect("Select features for the Independent variable:", df1.columns)

    st.divider()
    st.subheader("🤖 Model Selection")
    model_options = ['Linear_Regression', 'GLM-Binomial', 'GLM-Gaussian', 'GLM-Poisson', 'GLM-Negative_Binomial']
    model_type = st.selectbox("Please select Model: ", model_options)

    if st.button("🚀 Train and Evaluate Model"):
        st.session_state.ind = independent_features
        st.session_state.model_type = model_type
        if independent_features and dependent_features:
            data_encoded, categorical_vars, new_independent_vars = preprocess_data(st.session_state.df, independent_features)
            st.session_state.new_independent_vars1 = new_independent_vars
            st.session_state.dep1 = dependent_features
            
            X = data_encoded[new_independent_vars]
            y = data_encoded[dependent_features]

            X_numeric = X.select_dtypes(include=[np.number])
            X_categorical = X.drop(columns=X_numeric.columns)

            xtrain_numeric, xtest_numeric, ytrain, ytest = train_test_split(X_numeric, y, test_size=0.2, random_state=42)
            xtrain_categorical, xtest_categorical = train_test_split(X_categorical, test_size=0.2, random_state=42)

            st.session_state.scaler = StandardScaler()
            xtrain_numeric = st.session_state.scaler.fit_transform(xtrain_numeric)
            xtest_numeric = st.session_state.scaler.transform(xtest_numeric)

            xtrain = np.hstack([xtrain_numeric, xtrain_categorical])
            xtest = np.hstack([xtest_numeric, xtest_categorical])

            f = check_linear_reg_validity(xtrain, ytrain)

            if f == 1:
                st.success('✅ Linear Regression Assumption Satisfied!')
                model = LinearRegression()
                model.fit(xtrain, ytrain)
                test_prediction = model.predict(xtest)

                err = MAE(test_prediction, ytest)
                err2 = MSE(test_prediction, ytest)
                err3 = np.sqrt(err2)
                r2_score_value = model.score(xtest, ytest)
                
                st.divider()
                st.subheader("Model Performance Metrics")
                st.info(f"""
                Model Metrics:
                MAE: {err:.3f}, 
                MSE: {err3:.3f},
                R^2 Score: {r2_score_value:.3f}
                """)
                
                if st.session_state.llm:
                    response_metrics = get_llm_response(r2_score=r2_score_value, mae=err, mse=err3)
                    st.write("## Explanation from LLM for Metrics:")
                    st.markdown(response_metrics)
                else:
                    st.warning("LLM not initialized. Explanations will not be provided.")
                
                if r2_score_value > 0.2:
                    residuals = ytest - test_prediction
                    dw_stat = durbin_watson(residuals)
                    residuals_mean = np.mean(residuals)
                    
                    st.divider()
                    st.subheader("Residual Analysis")
                    st.info(f"""
                    Residual Analysis:
                    Mean of Residuals: {residuals_mean:.3f},
                    Durbin-Watson Statistic: {dw_stat:.3f}
                    """)
                    plot_fitted_vs_predicted_streamlit(ytest, test_prediction, 'Linear Regression Model')
                    st.divider()
                    
                else:
                    st.warning('⚠️ Despite the fact that Linear Regression assumptions are satisfied, it would not be a good fit! There might be some error in the data. Please check the Data and Retry!')

                if abs(residuals_mean) < 10:
                    st.subheader("🔄 Alternative Models", divider='gray')
                    mse_r, mae_r, model_r = ridge_regression(xtrain, xtest, ytrain, ytest, 10, st.session_state.llm)
                    mse_l, mae_l, model_l = lasso_regression(xtrain, xtest, ytrain, ytest, 10, st.session_state.llm)
                    if mae_r < mae_l:
                        st.session_state.model = model_r
                    else:
                        st.session_state.model = model_l
                else:
                    st.session_state.model = model
            else:
                st.warning('⚠️ The assumptions for Linear Regression are not satisfied. We recommend considering a Generalized Linear Model (GLM) instead.')
                if isinstance(ytrain, int):
                    st.info('Since your target variable is discrete, we recommend using either a Negative Binomial or Poisson GLM model for better results.')
                else:
                    st.info('Given that your target variable is continuous, a Gaussian GLM model would be a more suitable choice.')
                st.session_state.model = fit_glm(xtrain, xtest, ytrain, ytest, st.session_state.model_type)
        else:
            st.error("Please select both dependent and independent features before training the model.")

def ridge_regression(X_train, X_test, y_train, y_test, alpha, llm=None):
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    err = MAE(y_pred, y_test)
    err2 = MSE(y_pred, y_test)
    err3 = np.sqrt(err2)
    r2_score_value = ridge.score(X_test, y_test)
    
    if r2_score_value > 0.2:
        st.subheader("1) Ridge Regression Results:")
        st.info(f"""
                Model Metrics:
                MAE: {err:.3f}, 
                MSE: {err3:.3f},
                R^2 Score: {r2_score_value:.3f}
            """)

        if llm:
            response_metrics = get_llm_response(r2_score=r2_score_value, mae=err, mse=err3)
            st.write("## Explanation from LLM for Ridge Regression Metrics:")
            st.markdown(response_metrics)

        residuals = y_test - y_pred
        dw_stat = durbin_watson(residuals)
        residuals_mean = np.mean(residuals)
        st.info(f'Mean of Residuals: {residuals_mean}, Durbin-Watson Statistic: {dw_stat}')        
        st.divider()
        plot_fitted_vs_predicted_streamlit(y_test, y_pred, 'Ridge Model')
        st.divider()        
    
    return err, err2, ridge

def lasso_regression(X_train, X_test, y_train, y_test, alpha, llm=None):
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    err = MAE(y_pred, y_test)
    err2 = MSE(y_pred, y_test)
    err3 = np.sqrt(err2)
    r2_score_value = lasso.score(X_test, y_test)
    
    st.subheader("2) Lasso Regression Result:")
    st.info(f"""
                Model Metrics:
                MAE: {err:.3f}, 
                MSE: {err3:.3f},
                R^2 Score: {r2_score_value:.3f}
            """)

    if llm:
        response_metrics = get_llm_response(r2_score=r2_score_value, mae=err, mse=err3)
        st.markdown("## Explanation from LLM for Lasso Regression Metrics:")
        st.markdown(response_metrics)

    residuals = y_test - y_pred
    dw_stat = durbin_watson(residuals)
    residuals_mean = np.mean(residuals)
    st.info(f'Mean of Residuals: {residuals_mean}, Durbin-Watson Statistic: {dw_stat}')
    st.divider()
    plot_fitted_vs_predicted_streamlit(y_test, y_pred, 'Lasso Model')
    st.divider()    
    return err, err2, lasso

def check_linear_reg_validity(X, y):
    f = 0
    stat, p_value = shapiro(y)

    xtrain = sm.add_constant(X)
    
    ols_model = sm.OLS(y,xtrain).fit()
    
    lm_stat, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(ols_model.resid, xtrain)
    
    alpha = 0.05
    if ((p_value > alpha) and (lm_pvalue>alpha)):
        f = 1
    return f

def fit_glm(xtrain, xtest, ytrain, ytest, model_type):
    st.write(f"You selected: {model_type}")

    # Convert training data to DataFrames
    xtrain_df = pd.DataFrame(xtrain, columns=st.session_state.new_independent_vars1)
    ytrain_df = pd.DataFrame(ytrain, columns=[st.session_state.dep1])

    # Align indices between xtrain and ytrain
    xtrain_df.index = ytrain_df.index

    # Combine xtrain and ytrain into a single DataFrame
    train_dataset = pd.concat([xtrain_df, ytrain_df], axis=1)

    # Select the appropriate GLM family
    if model_type == 'GLM-Gaussian':
        family_class = sm.families.Gaussian()
    elif model_type == 'GLM-Binomial':
        family_class = sm.families.Binomial()
    elif model_type == 'GLM-Poisson':
        family_class = sm.families.Poisson()
    elif model_type == 'GLM-Negative_Binomial':
        family_class = sm.families.NegativeBinomial()
    else:
        st.error("Invalid model type selected.")
        return

    # Define the formula for the GLM model
    independent_vars = ' + '.join(xtrain_df.columns)
    formula1 = f"{st.session_state.dep1} ~ {independent_vars}"

    # Fit the GLM model
    model = smf.glm(formula=formula1, data=train_dataset, family=family_class).fit()
    
    st.divider()
    st.write("## Model Summary")
    st.write(model.summary())

    # Extract key summary stats
    summary_dict = {
        "coefficients": model.params.to_dict(),
        "p_values": model.pvalues.to_dict(),
        "aic": model.aic,
        "bic": model.bic
    }

    summary_str = f"""
    **Coefficients:** {summary_dict['coefficients']}
    **P-values:** {summary_dict['p_values']}
    **AIC:** {summary_dict['aic']}
    **BIC:** {summary_dict['bic']}
    """

    # Predict on the test set
    xtest_df = pd.DataFrame(xtest, columns=st.session_state.new_independent_vars1)
    ypred = model.predict(xtest_df)

    # Calculate evaluation metrics
    err = MAE(ytest, ypred)
    err2 = MSE(ytest, ypred)
    err3 = np.sqrt(err2)
    r2 = r2_score(ytest, ypred)
    
    st.divider()
    st.write("## Model Metrics")
    st.info(f"""
           **MAE:** {err:.3f},
           **MSE:** {err3:.3f},
           **R^2 Score:** {r2:.3f}
    """)

    if st.session_state.llm:
        response_metrics = get_llm_response(r2_score=r2, mae=err, mse=err3, summary=summary_str)
        st.write("## Explanation from LLM for GLM Metrics")
        st.markdown(response_metrics)
    
    st.divider()
    # Plot the fitted vs predicted values
    st.write("## Fitted vs Predicted")
    plot_fitted_vs_predicted_streamlit(ytest, ypred, title=f'Model: {model_type}')

    return model

def plot_fitted_vs_predicted_streamlit(y_fitted, y_pred, title="Fitted vs Predicted Values", xlabel="Index", ylabel="Values"):
    trace_fitted = go.Scatter(
        x=list(range(len(y_fitted))),
        y=y_fitted,
        mode='lines+markers',
        name='True Values',
        line=dict(color='green', dash='dash')
    )
    
    trace_pred = go.Scatter(
        x=list(range(len(y_pred))),
        y=y_pred,
        mode='lines+markers',
        name='Predicted Values',
        line=dict(color='red')
    )
    
    layout = go.Layout(
        title=title,
        xaxis=dict(title=xlabel),
        yaxis=dict(title=ylabel),
        showlegend=True
    )
    
    fig = go.Figure(data=[trace_fitted, trace_pred], layout=layout)
    fig.update_layout(width=1000, height=600)
    # Use st.plotly_chart to display the plot in Streamlit
    st.plotly_chart(fig)

def predict():
    if 'model' not in st.session_state or st.session_state.model is None:
        st.write("No trained model found. Please perform analysis first.")
        return

    model = st.session_state.model
    if model is None:
        st.write("No model available for prediction.")
        return
    st.write("Enter the features for prediction:")

    # Separate numeric and categorical features
    numeric_features = []
    categorical_features = []
    for feature in st.session_state.ind:
        if pd.api.types.is_numeric_dtype(st.session_state.df[feature]):
            numeric_features.append(feature)
        else:
            categorical_features.append(feature)
    
    # Input for numeric features
    numeric_values = []
    for feature in numeric_features:
        value = st.number_input(f"Enter value for {feature}:")
        numeric_values.append(value)
        
    # Input for categorical features
    categorical_values = []
    if len(categorical_features) > 0:
        for feature in categorical_features:
            options = st.session_state.df[feature].unique().tolist()
            value = st.selectbox(f"Select value for {feature}:", options)
            categorical_values.append(value)

    if st.button("Submit"):
        # Convert numeric inputs to NumPy array
        numeric_array = np.array(numeric_values).reshape(1, -1)
        
        # Apply the same scaling that was used during training
        numeric_scaled = st.session_state.scaler.transform(numeric_array)
        
        # Create DataFrames for numeric and categorical inputs
        numeric_df = pd.DataFrame(numeric_scaled, columns=numeric_features)
        
        if len(categorical_features) > 0:
            # One-hot encode categorical inputs
            categorical_df = pd.DataFrame([categorical_values], columns=categorical_features)
            categorical_encoded = pd.get_dummies(categorical_df)
            
            # Ensure the columns match the trained model's input
            missing_cols = set(st.session_state.new_independent_vars1) - set(numeric_features) - set(categorical_encoded.columns)
            for col in missing_cols:
                categorical_encoded[col] = 0
            categorical_encoded = categorical_encoded[st.session_state.new_independent_vars1[len(numeric_features):]]
            
            # Combine numeric and categorical data
            combined_df = pd.concat([numeric_df, categorical_encoded], axis=1)
        else:
            combined_df = numeric_df
        
        # Make a prediction
        prediction = model.predict(combined_df)

        # Display the prediction
        st.write(f"The predicted value is: {prediction[0]}")



