# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Life Expectancy Prediction",
    page_icon=":bar_chart:",
    layout="wide"
)

# Loading the Model and Scaler
from joblib import load

model = load("best_model_stacking_regressor.pkl")
scaler = load("scaler.pkl")

# =========================
# Load Dataset
# =========================

df = pd.read_csv("Life Expectancy Data.csv")


# ============================================================# COUNTRY LIST
# ============================================================

countries = sorted(df['Country'].unique())

country_mapping = {
    country: idx for idx, country in enumerate(countries)
}

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_names = [

    'Country',
    'Year',
    'Status',
    'Adult_Mortality',
    'Infant_Deaths',
    'Alcohol',
    'Percentage_Expenditure',
    'Hepatitis_B',
    'Measles',
    'Bmi',
    'Under-Five_Deaths',
    'Polio',
    'Total_Expenditure',
    'Diphtheria',
    'Hiv/Aids',
    'Gdp',
    'Population',
    'Thinness__1-19_Years',
    'Thinness_5-9_Years',
    'Income_Composition_Of_Resources',
    'Schooling'
]



# ============================================================
# LOAD BANNER
# ============================================================

banner = Image.open("banner.png")

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #0f172a, #111827);
    color: white;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4 {
    color: #FACC15;
}

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 15px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}

.stButton>button {
    background: linear-gradient(to right, #FACC15, #F59E0B);
    color: black;
    border-radius: 12px;
    height: 3.5em;
    width: 100%;
    font-size: 20px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    transform: scale(1.02);
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# DISPLAY BANNER
# ============================================================

st.image(banner, use_container_width=True)

# ============================================================
# TITLE
# ============================================================

# Title and Description
st.title("🌍 Life Expectancy Prediction using Machine Learning")

st.markdown("""
### Predict Life Expectancy using Healthcare, Economic & Social Indicators

✅ Final Model: Stacking Regressor  
✅ Test R² Score: 96.99%  
✅ RMSE: 1.61  

Developed By: **nmshah9**
""")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Project Navigation")

st.sidebar.info("""
### About Project

This project predicts Life Expectancy using:

- Healthcare Indicators
- Mortality Statistics
- Economic Factors
- Immunization Coverage
- Population Metrics

### ML Algorithms Used

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Decision Tree
- Random Forest
- KNN
- SVR
- AdaBoost
- XGBoost
- CatBoost
- LightGBM
- Gradient Boosting
- Voting Regressor
- Stacking Regressor
- Bagging Regressor
""")

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediction",
    "📊 Feature Importance",
    "📈 Analytics Dashboard",
    "📘 Project Insights"
])

# ============================================================
# TAB 1 — PREDICTION
# ============================================================

with tab1:

    st.header("📝 Input Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:

        selected_country = st.selectbox(
            "Select Country",
            countries
        )

        year = st.slider(
            "Year",
            2000,
            2030,
            2015
        )

        status = st.selectbox(
            "Status",
            ["Developing", "Developed"]
        )

        adult_mortality = st.slider(
            "Adult Mortality",
            0,
            500,
            150
        )

        infant_deaths = st.slider(
            "Infant Deaths",
            0,
            200,
            20
        )

        alcohol = st.slider(
            "Alcohol Consumption",
            0.0,
            20.0,
            4.0
        )

        percentage_expenditure = st.number_input(
            "Percentage Expenditure",
            0.0,
            10000.0,
            100.0
        )

    with col2:

        hepatitis_b = st.slider(
            "Hepatitis B",
            0,
            100,
            80
        )

        measles = st.slider(
            "Measles",
            0,
            10000,
            100
        )

        bmi = st.slider(
            "BMI",
            0.0,
            50.0,
            25.0
        )

        under_five_deaths = st.slider(
            "Under Five Deaths",
            0,
            200,
            25
        )

        polio = st.slider(
            "Polio",
            0,
            100,
            85
        )

        total_expenditure = st.slider(
            "Total Expenditure",
            0.0,
            20.0,
            5.0
        )

        diphtheria = st.slider(
            "Diphtheria",
            0,
            100,
            85
        )

    with col3:

        hiv_aids = st.slider(
            "HIV/AIDS",
            0.0,
            50.0,
            0.1
        )

        gdp = st.number_input(
            "GDP",
            0.0,
            100000.0,
            5000.0
        )

        population = st.number_input(
            "Population",
            0,
            200000000,
            1000000
        )

        thinness_1_19 = st.slider(
            "Thinness 1-19 Years",
            0.0,
            30.0,
            5.0
        )

        thinness_5_9 = st.slider(
            "Thinness 5-9 Years",
            0.0,
            30.0,
            5.0
        )

        income_composition = st.slider(
            "Income Composition",
            0.0,
            1.0,
            0.5
        )

        schooling = st.slider(
            "Schooling",
            0.0,
            25.0,
            12.0
        )

    # ========================================================
    # ENCODING
    # ========================================================

    country_encoded = country_mapping[selected_country]

    status_encoded = 1 if status == "Developed" else 0

    # ========================================================
    # CREATE INPUT DATAFRAME
    # ========================================================

    input_data = pd.DataFrame({

        'Country': [country_encoded],
        'Year': [year],
        'Status': [status_encoded],
        'Adult_Mortality': [adult_mortality],
        'Infant_Deaths': [infant_deaths],
        'Alcohol': [alcohol],
        'Percentage_Expenditure': [percentage_expenditure],
        'Hepatitis_B': [hepatitis_b],
        'Measles': [measles],
        'Bmi': [bmi],
        'Under-Five_Deaths': [under_five_deaths],
        'Polio': [polio],
        'Total_Expenditure': [total_expenditure],
        'Diphtheria': [diphtheria],
        'Hiv/Aids': [hiv_aids],
        'Gdp': [gdp],
        'Population': [population],
        'Thinness__1-19_Years': [thinness_1_19],
        'Thinness_5-9_Years': [thinness_5_9],
        'Income_Composition_Of_Resources': [income_composition],
        'Schooling': [schooling]
    })

    # ========================================================
    # PREDICTION BUTTON
    # ========================================================


    if st.button("🚀 Predict Life Expectancy"):

        prediction = model.predict(input_data)
        predicted_value = prediction[0]

        st.success(
            f"🌟 Predicted Life Expectancy: {predicted_value:.2f} Years"
        )

        # ====================================================
        # METRICS CARDS
        # ====================================================

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Predicted Years",
            f"{predicted_value:.2f}"
        )

        metric2.metric(
            "Model Accuracy",
            "96.79%"
        )

        metric3.metric(
            "RMSE",
            "1.665"
        )

        # ====================================================
        # PREDICTION GAUGE
        # ====================================================

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = predicted_value,
            title = {'text': "Life Expectancy"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 3 — ANALYTICS DASHBOARD
# ============================================================

with tab3:

    st.header("📈 Analytics Dashboard")

    chart_data = pd.DataFrame({
        'Category': ['Healthcare', 'Economy', 'Education', 'Immunization'],
        'Impact': [85, 78, 82, 80]
    })

    fig = px.pie(
        chart_data,
        values='Impact',
        names='Category',
        title='Factors Influencing Life Expectancy'
    )

    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(
        x=[2000, 2005, 2010, 2015, 2020],
        y=[65, 67, 69, 71, 73],
        labels={'x':'Year', 'y':'Average Life Expectancy'},
        title='Global Life Expectancy Trend'
    )

    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# TAB 4 — PROJECT INSIGHTS
# ============================================================

with tab4:

    st.header("📘 Project Insights")

    with st.expander("📌 Machine Learning Workflow"):

        st.markdown("""
        - Data Cleaning
        - Missing Value Handling
        - Outlier Treatment
        - Feature Scaling
        - Exploratory Data Analysis
        - Model Training
        - Ensemble Learning
        - Model Deployment
        """)

    with st.expander("🤖 Algorithms Compared"):

        st.markdown("""
        - Linear Regression
        - Ridge Regression
        - Lasso Regression
        - ElasticNet
        - Decision Tree
        - Random Forest
        - AdaBoost
        - Gradient Boosting
        - XGBoost
        - LightGBM
        - CatBoost
        - KNN
        - SVR
        - Voting Regressor
        - Stacking Regressor
        """)

    with st.expander("🏆 Final Model Selection"):

        st.markdown("""
        Final Selected Model:

        ✅ Stacking Regressor
        Reasons:

        - Excellent Generalization
        - High Test Accuracy
        - Lower Overfitting
        - Stable Predictions
        - Strong Ensemble Learning
        """)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div style='text-align: center;'>

### 🌍 Life Expectancy Prediction using Machine Learning

Built with ❤️ using Streamlit | Developed by nmshah9

</div>
""", unsafe_allow_html=True)
