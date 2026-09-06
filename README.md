# LISA: LLM Informed Statistical Analysis

**LISA** (**L**LM **I**nformed **S**tatistical **A**nalysis) is an interactive data science and statistical analysis web application powered by Streamlit and accelerated by Large Language Models (LLMs) via Groq. The platform bridges the gap between raw data and actionable intelligence, combining automated statistical workflows, machine learning models, and natural language interfaces to make complex data analysis intuitive and explainable.

## Live Application
Access the deployed application: [LISA on Streamlit](https://lisatool.streamlit.app/)

## Key Features

### Natural Language Data Querying (Text-to-SQL)
- Query datasets using plain English questions.
- Automatically generates optimized, schema-aware SQLite queries.
- Includes a self-healing error correction loop that automatically repairs syntax or schema issues.
- Interactive SQL viewer and editor allowing direct inspection, manual adjustments, and re-execution.
- Generates natural language summaries and insights from query results.

### Integrated Data Chatbot
- Conversational data assistant powered by Groq LLMs.
- Context-aware dialogue based on dataset rows and conversational history.
- Real-time response streaming for fast interactive exploration.

### Exploratory Data Analysis & Integrity Checks
- Quick inspection of dataset dimensions, column distributions, and data types.
- Automated data integrity reports checking for duplicates, missing values, and anomalies.
- Comprehensive descriptive statistics for both numeric and categorical variables.

### Interactive Visualizations
- Rich Plotly charts including histograms, scatter plots, line plots, box plots, bar charts, heatmaps, pie charts, and violin plots.
- AI-generated visual interpretations explaining key patterns and anomalies in plain English.

### Statistical Analysis Suite
- **Categorical Data Analysis**: Frequency tables, Pareto charts, mode calculations, and Chi-Square tests of independence.
- **Continuous Data Analysis**: Parametric and non-parametric hypothesis tests (One-way ANOVA, Repeated Measures ANOVA, Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis, Friedman test, and Z-tests).
- **Regression Analysis**: Ordinary Least Squares (OLS), Ridge, and Lasso regression with diagnostic tests (Breusch-Pagan for heteroscedasticity, Shapiro-Wilk for normality, Durbin-Watson for autocorrelation) and performance metrics (R², MAE, MSE, RMSE).

### Classification Model Builder
- Train, evaluate, and compare multiple classification algorithms:
  - Logistic Regression
  - Support Vector Classifiers (SVC)
  - K-Nearest Neighbors (KNN)
  - Random Forest & AdaBoost
  - XGBoost
- Automated preprocessing pipeline (imputation, one-hot encoding, feature scaling).
- Imbalanced dataset handling with SMOTE, Random Over-Sampling, and Random Under-Sampling.
- Stratified K-Fold cross-validation, confusion matrices, and AI performance summaries.

### High-Speed LLM Inference
- Powered by Groq's LPU hardware for low-latency inferences.
- Dynamic model discovery fetching models available on your specific Groq API key.
- Supports leading open models including GPT OSS 120B, GPT OSS 20B, Llama 3.3 70B, Llama 3.1 8B, and DeepSeek R1 Distill 70B.

## Getting Started

### Prerequisites
- Python 3.12 or 3.13
- A [Groq API Key](https://console.groq.com/keys)

### Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/naveen3830/lisa.git
   cd lisa
   ```

2. **Set up environment**:
   Using `uv` (recommended):
   ```sh
   uv sync
   ```
   Or using standard `pip`:
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure API Key**:
   Create a `.env` file in the root directory (optional, or enter it directly in the sidebar):
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Launch the Application**:
   ```sh
   streamlit run app.py
   ```
   Or with `uv`:
   ```sh
   uv run streamlit run app.py
   ```

## Project Structure
```
LISA/
├── app.py                     # Main application entry point and navigation
├── Home.py                    # Home page, text-to-SQL, data overview, chatbot, model card
├── functions.py               # Utility functions and data integrity checks
├── upload_data.py             # CSV upload component
├── src/
│   └── components/
│       ├── viz.py             # Plotly visualization module
│       ├── classification.py  # Machine learning classification pipeline
│       ├── statistical_analysis.py # Statistical analysis tab container
│       ├── Categorical_Analysis.py # Categorical statistical tests
│       ├── Continuous_Analysis.py  # Hypothesis testing and ANOVA
│       └── Regression_Analysis.py  # Linear, Ridge, and Lasso regression
├── Data/
│   └── modelcard.csv          # LLM benchmark comparison data
├── pyproject.toml             # Project configuration and dependency specifications
├── requirements.txt           # Pinned dependencies with secure version baselines
└── uv.lock                    # Fully resolved dependency lockfile
```

## Contributing
Contributions and feedback are welcome. Please feel free to open an issue or submit a pull request.

## License
This project is licensed under the terms of the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
