import streamlit as st
import pandas as pd
def upload_data():
    # Dataset options for dropdown
    dataset_options = {
        'Select a Dataset': None,
        'Sample Dataset for chi square goodness of test': 'data_chisquared_goodness_of_fit.csv',
        'Sample Dataset for chi square test for independence': 'chi_square_test_for_independence.csv',
        'Sample Dataset for Two Sample T Test':'dataset_for_two_sample_t_test.csv',
        'Sample Dataset for Paired T Test':'dataset_for_paired_t_test.csv',
        'Sample Dataset for one way Anova':'dataset_for_one_way_anova.csv',
        'Sample Dataset for repeated measure anova':'dataset_for_repeated_measure_anova.csv',
        'Sample Dataset for Welch Test':'dataset_for_welch_test.csv',
        'Sample Dataset for Mann Whitney U Test':'dataset_for_mann_whitney_u_test.csv',
        'Sample Dataset for Wilcoxon signed rank test':'dataset_for_wilcoxon_signed_rank_test.csv',
        'Sample Dataset for Kruskal Test':'dataset_for_kruskal_test.csv',
        'Sample Dataset for One Sample Z Test':'dataset_for_one_sample_z_test.csv',
        'Sample Dataset for regression': '50_Startups.csv'

    }

    # Dropdown for dataset selection
    selected_dataset = st.selectbox('Want to explore?  We have these datasets ready to preload!', list(dataset_options.keys()))

    if selected_dataset != 'Select a Dataset':
        dataset_path = dataset_options[selected_dataset]
        st.session_state.data = pd.read_csv(dataset_path)
        st.write(f"{selected_dataset.split('(')[0].strip()} loaded:")
        st.write(st.session_state.data.head())

    # Option to clear the preloaded or uploaded file
    if st.button('Clear Data'):
        st.session_state.data = None
        st.write("Data cleared. You can now upload your own dataset or choose another from the dropdown.")

    # Step 1: Upload CSV
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.write(st.session_state.data.head())


  
    