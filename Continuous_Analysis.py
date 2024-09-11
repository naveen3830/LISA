import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
from scipy.stats import f_oneway, wilcoxon, mannwhitneyu, kruskal, friedmanchisquare
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.weightstats import ztest
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

# Call this function at the beginning of your app
initialize_llm()

def get_llm_response_for_test(test_statistic=None, p_value=None, test_type=None):
    if not st.session_state.llm:
        return "LLM not initialized. Please check your API key and model settings."

    system_message_prompt = SystemMessagePromptTemplate.from_template(
        "You are StatBot, an expert statistical analyst. "
        "Explain the output of the {test_type} in simple English."
    )
    human_message_prompt_template = """
    {test_statistic_section}
    {p_value_section}
    
    Based on these results, explain the statistical test outcome in a concise and clear manner. Discuss any relevant insights as well.
    """

    test_statistic_section = f"The test statistic is {test_statistic:.3f}." if test_statistic is not None else ""
    p_value_section = f"The p-value is {p_value:.3f}." if p_value is not None else ""

    human_message_prompt = HumanMessagePromptTemplate.from_template(human_message_prompt_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    
    formatted_chat_prompt = chat_prompt.format_messages(
        test_statistic_section=test_statistic_section,
        p_value_section=p_value_section,
        test_type=test_type
    )
    response = st.session_state.llm.invoke(formatted_chat_prompt)
    return response.content

def continuous_data_analysis():
    if 'df' in st.session_state and st.session_state.df is not None:
        sub_page = st.selectbox(
            "Choose a task for continuous data analysis:",
            ["One Sample T Test", "Two Sample T Test",
            "Paired T Test", "One Way ANOVA", "Repeated Measure ANOVA",
            "Welch Test", "Mann Whitney U Test", "Wilcoxon Signed-Rank Test",
            "Kruskal-Wallis H Test", "One Sample Z Test"]
        )
        if sub_page == "One Sample T Test":
            one_sample_t_test()
        elif sub_page == "Two Sample T Test":
            two_sample_t_test()
        elif sub_page == "Paired T Test":
            paired_sample_t_test()
        elif sub_page == "One Way ANOVA":
            one_way_anova()
        elif sub_page == "Repeated Measure ANOVA":
            repeat_measure_anova()
        elif sub_page == "Welch Test":
            welch_test()
        elif sub_page == "Mann Whitney U Test":
            mann_whitney_u_test()
        elif sub_page == "Wilcoxon Signed-Rank Test":
            wilcoxon_signed_rank_test()
        elif sub_page == "Kruskal-Wallis H Test":
            kruskal_wallis_h_test()
        elif sub_page == "One Sample Z Test":
            one_sample_z_test()
    else:
        st.warning("Please upload data first.")

def one_sample_t_test():
    st.header("One Sample T Test")
    if 'hypothesized_mean' not in st.session_state:
        st.session_state.hypothesized_mean = 0.0
    if 'alpha' not in st.session_state:
        st.session_state.alpha = 0.05
    
    data_for_test = st.selectbox("Select a feature for One Sample T Test", st.session_state.df.columns)
    st.session_state.hypothesized_mean = st.number_input("Hypothesized Population Mean", value=st.session_state.hypothesized_mean)
    st.session_state.alpha = st.number_input("Significance Level", value=st.session_state.alpha)
    
    if st.button("Perform Test", key="one_sample_t_test"):
        t_statistic, p_value = stats.ttest_1samp(st.session_state.df[data_for_test], st.session_state.hypothesized_mean)
        st.write(f"T-statistic: {t_statistic}")
        st.write(f"P-value: {p_value}")
        
        if p_value < st.session_state.alpha:
            st.write("Reject the null hypothesis. The sample mean is significantly different from the population mean.")
        else:
            st.write("Fail to reject the null hypothesis. The sample mean is not significantly different from the population mean.")
            
        if st.session_state.llm:
            response = get_llm_response_for_test(test_statistic=t_statistic, p_value=p_value, test_type="One Sample T Test")
            st.divider()
            st.write("## Explanation from LLM:")
            st.markdown(response)
            st.divider()

def two_sample_t_test():
    st.header("Two Sample T Test")
    if 'alpha' not in st.session_state:
        st.session_state.alpha = 0.05
    
    data1_for_test = st.selectbox("Select first feature for Two Sample T Test", st.session_state.df.columns)
    a = st.session_state.df[data1_for_test].dropna()
    data2_for_test = st.selectbox("Select second feature for Two Sample T Test", st.session_state.df.columns)
    b = st.session_state.df[data2_for_test].dropna()
    
    st.session_state.alpha = st.number_input("Significance Level", value=st.session_state.alpha)
    
    if st.button("Perform Test", key="two_sample_t_test"):
        t_statistic, p_value = stats.ttest_ind(a, b, equal_var=True)
        st.write(f'T-statistic: {t_statistic:.4f}')
        st.write(f'P-value: {p_value:.4f}')
        
        if p_value < st.session_state.alpha:
            st.write("Reject the null hypothesis.")
        else:
            st.write("Fail to reject the null hypothesis.")
            
        if st.session_state.llm:
            response = get_llm_response_for_test(test_statistic=t_statistic, p_value=p_value, test_type="Two Sample T Test")
            st.divider()
            st.write("## Explanation from LLM:")
            st.markdown(response)
            st.divider()
            
def paired_sample_t_test():
    st.header("Paired Sample T Test")
    if 'alpha' not in st.session_state:
        st.session_state.alpha = 0.05
    
    data1_for_test = st.selectbox("Select first feature for Paired Sample T Test", st.session_state.df.columns)
    a = st.session_state.df[data1_for_test]
    data2_for_test = st.selectbox("Select second feature for Paired Sample T Test", st.session_state.df.columns)
    b = st.session_state.df[data2_for_test]
    
    st.session_state.alpha = st.number_input("Significance Level", value=st.session_state.alpha)
    
    if st.button("Perform Test", key="paired_sample_t_test"):
        t_statistic, p_value = stats.ttest_rel(a, b)
        st.write(f'T-statistic: {t_statistic}')
        st.write(f'P-value: {p_value}')
        
        if p_value < st.session_state.alpha:
            st.write("Reject the null hypothesis.")
        else:
            st.write("Fail to reject the null hypothesis.")
            
        if st.session_state.llm:
            response = get_llm_response_for_test(test_statistic=t_statistic, p_value=p_value, test_type="Paired Sample T Test")
            st.divider()
            st.write("## Explanation from LLM:")
            st.markdown(response)
            st.divider()

def one_way_anova():
    st.header("One Way ANOVA")
    if 'alpha' not in st.session_state:
        st.session_state.alpha = 0.05
    
    options = st.multiselect("Select Data for performing ANOVA", st.session_state.df.columns)
    selected_data = [st.session_state.df[col] for col in options]
    st.session_state.alpha = st.number_input("Significance Level", value=st.session_state.alpha)
    
    if st.button("Perform Test", key="one_way_anova"):
        anova_statistic, p_value = f_oneway(*selected_data)
        st.write(f'F-statistic: {anova_statistic}')
        st.write(f'P-value: {p_value}')
        
        if p_value < st.session_state.alpha:
            st.write("Reject the null hypothesis.")
        else:
            st.write("Fail to reject the null hypothesis.")

        if st.session_state.llm:
            response = get_llm_response_for_test(test_statistic=anova_statistic, p_value=p_value, test_type="One Way ANOVA Test")
            st.divider()
            st.write("## Explanation from LLM:")
            st.markdown(response)
            st.divider()

def repeat_measure_anova():
    st.header("Repeated Measure ANOVA")
    if 'alpha' not in st.session_state:
        st.session_state.alpha = 0.05
        
    dependent_variable = st.selectbox("Select dependent variable for Test", st.session_state.df.columns)
    subject = st.selectbox("Select subject for Test", st.session_state.df.columns)
    within_subject_factor = st.selectbox("Select within-subject factor for Test", st.session_state.df.columns)

    st.session_state.alpha = st.number_input("Significance Level", value=st.session_state.alpha)
    
    if st.button("Perform Test", key="repeat_measure_anova"):
        anova_results = AnovaRM(data=st.session_state.df, depvar=dependent_variable, 
                        subject=subject, within=[within_subject_factor]).fit()
        
        p_value = anova_results.anova_table['Pr > F'][within_subject_factor]
        F_value = anova_results.anova_table['F Value'][within_subject_factor]
        num_value = anova_results.anova_table['Num DF'][within_subject_factor]
        denom_value = anova_results.anova_table['Den DF'][within_subject_factor]
        
        st.write(f"F Value = {F_value}")
        st.write(f"P Value = {p_value}")
        st.write(f"Numerator degrees of freedom = {num_value}")
        st.write(f"Denominator degrees of freedom = {denom_value}")
        
        if p_value < st.session_state.alpha:
            st.write("Reject the null hypothesis.")
        else:
            st.write("Fail to reject the null hypothesis.")
        
        if st.session_state.llm:
            response = get_llm_response_for_test(test_statistic=F_value, p_value=p_value, test_type="Repeated Measures ANOVA")
            st.divider()
            st.write("## Explanation from LLM:")
            st.markdown(response)
            st.divider()

def welch_test():
    st.header("Welch Test")
    if 'alpha' not in st.session_state:
        st.session_state.alpha = 0.05
    
    data1_for_test = st.selectbox("Select first feature for Welch Test", st.session_state.df.columns)
    a = st.session_state.df[data1_for_test].dropna()
    data2_for_test = st.selectbox("Select second feature for Welch Test", st.session_state.df.columns)
    b = st.session_state.df[data2_for_test].dropna()
    
    st.session_state.alpha = st.number_input("Significance Level", value=st.session_state.alpha)
    
    if st.button("Perform Test", key="welch_test"):
        t_statistic, p_value = stats.ttest_ind(a, b, equal_var=False)

        st.write(f'T-statistic: {t_statistic}')
        st.write(f'P-value: {p_value}')
        
        if p_value < st.session_state.alpha:
            st.write("Reject the null hypothesis.")
        else:
            st.write("Fail to reject the null hypothesis.")
            
        if st.session_state.llm:
            response = get_llm_response_for_test(test_statistic=t_statistic, p_value=p_value, test_type="Welch Test")
            st.divider()
            st.write("## Explanation from LLM:")
            st.markdown(response)
            st.divider()

def mann_whitney_u_test():
    st.header("Mann-Whitney U Test")
    if 'alpha' not in st.session_state:
        st.session_state.alpha = 0.05
    
    data1_for_test = st.selectbox("Select first feature for Test", st.session_state.df.columns)
    a = st.session_state.df[data1_for_test].dropna()
    data2_for_test = st.selectbox("Select second feature for Test", st.session_state.df.columns)
    b = st.session_state.df[data2_for_test].dropna()

    st.session_state.alpha = st.number_input("Significance Level", value=st.session_state.alpha)
    
    if st.button("Perform Test", key="mann_whitney_u_test"):
        statistic, p_value = mannwhitneyu(a, b)

        st.write(f'Statistic: {statistic}')
        st.write(f'P-value: {p_value}')
        
        if p_value < st.session_state.alpha:
            st.write("Reject the null hypothesis.")
        else:
            st.write("Fail to reject the null hypothesis.")
        
        if st.session_state.llm:
            response = get_llm_response_for_test(test_statistic=statistic, p_value=p_value, test_type="Mann-Whitney U Test")
            st.divider()
            st.write("## Explanation from LLM:")
            st.markdown(response)
            st.divider()

def wilcoxon_signed_rank_test():
    st.header("Wilcoxon Signed-Rank Test")
    if 'alpha' not in st.session_state:
        st.session_state.alpha = 0.05
    
    data1_for_test = st.selectbox("Select first feature for Test", st.session_state.df.columns)
    a = st.session_state.df[data1_for_test].dropna()
    data2_for_test = st.selectbox("Select second feature for Test", st.session_state.df.columns)
    b = st.session_state.df[data2_for_test].dropna()

    st.session_state.alpha = st.number_input("Significance Level", value=st.session_state.alpha)
    
    if st.button("Perform Test", key="wilcoxon_signed_rank_test"):
        statistic, p_value = wilcoxon(a, b)

        st.write(f'Statistic: {statistic}')
        st.write(f'P-value: {p_value}')
        
        if p_value < st.session_state.alpha:
            st.write("Reject the null hypothesis.")
        else:
            st.write("Fail to reject the null hypothesis.")
        
        if st.session_state.llm:
            response = get_llm_response_for_test(test_statistic=statistic, p_value=p_value, test_type="Wilcoxon Signed-Rank Test")
            st.divider()
            st.write("## Explanation from LLM:")
            st.markdown(response)
            st.divider()

def kruskal_wallis_h_test():
    st.header("Kruskal-Wallis H Test")
    if 'alpha' not in st.session_state:
        st.session_state.alpha = 0.05
    
    options = st.multiselect("Select Data for performing Kruskal-Wallis H test", st.session_state.df.columns)
    selected_data = [st.session_state.df[col] for col in options]
    
    st.session_state.alpha = st.number_input("Significance Level", value=st.session_state.alpha)
    
    if st.button("Perform Test", key="kruskal_wallis_h_test"):
        statistic, p_value = kruskal(*selected_data)
        
        st.write(f'Statistic: {statistic}')
        st.write(f'P-value: {p_value}')
        
        if p_value < st.session_state.alpha:
            st.write("Reject the null hypothesis.")
        else:
            st.write("Fail to reject the null hypothesis.")
        
        if st.session_state.llm:
            response = get_llm_response_for_test(test_statistic=statistic, p_value=p_value, test_type="Kruskal-Wallis H Test")
            st.divider()
            st.write("## Explanation from LLM:")
            st.markdown(response)
            st.divider()

def one_sample_z_test():
    st.header("One Sample Z Test")
    
    if 'hypothesized_mean' not in st.session_state:
        st.session_state.hypothesized_mean = 0.0
    if 'alpha' not in st.session_state:
        st.session_state.alpha = 0.05
    
    data_for_test = st.selectbox("Select a feature for One Sample Z test", st.session_state.df.columns)
    st.session_state.hypothesized_mean = st.number_input("Hypothesized Population Mean", value=st.session_state.hypothesized_mean)
    st.session_state.alpha = st.number_input("Significance Level", value=st.session_state.alpha)
    
    if st.button("Perform Test", key="one_sample_z_test"):
        statistic, p_value = ztest(st.session_state.df[data_for_test], value=st.session_state.hypothesized_mean)
        
        st.write(f"Z-statistic: {statistic}")
        st.write(f"P-value: {p_value}")
        
        if p_value < st.session_state.alpha:
            st.write("Reject the null hypothesis.")
        else:
            st.write("Fail to reject the null hypothesis.")
        
        if st.session_state.llm:
            response = get_llm_response_for_test(test_statistic=statistic, p_value=p_value, test_type="One Sample Z Test")
            st.divider()
            st.write("## Explanation from LLM:")
            st.markdown(response)
            st.divider()

