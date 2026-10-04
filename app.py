import streamlit as st
import joblib
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Cost Prediction App",
    page_icon="💰",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 25px;
        background-color: black;
        border: 1px solid #cfe3ff;
    }

    .result-value {
        font-size: 32px;
        font-weight: 700;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f7f7f7;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "cost_prediction_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(
        "❌ Model file not found. "
        "Please place 'cost_prediction_model.pkl' in the same folder as app.py."
    )
    st.stop()

model = joblib.load(MODEL_PATH)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="title">💰 Cost Prediction App</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning based Cost Prediction</div>',
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Input
# -----------------------------
st.subheader("🔢 Enter Task Points")

points = st.number_input(
    "Points",
    min_value=0.0,
    max_value=100.0,
    value=15.0,
    step=0.1
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🚀 Predict Cost", use_container_width=True):

    prediction = model.predict([[points]])[0]

    if prediction < 0:
        prediction = 0

    st.markdown(
        f"""
        <div class="result-box">
            <div>Predicted Cost</div>
            <div class="result-value">{prediction:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success("Prediction completed successfully!")

# -----------------------------
# About Model
# -----------------------------
st.divider()

st.subheader("📊 About the Model")

st.markdown("""
This application uses a **Machine Learning Regression Model** to predict
the cost of a task based on its points.

**Input Feature:** Points  
**Target Variable:** Cost  
**Algorithm:** Linear Regression
""")

st.markdown(
    """
    <div class="info-box">
    💡 The prediction is based on the relationship identified during
    Exploratory Data Analysis between points and cost.
    </div>
    """,
    unsafe_allow_html=True
)
