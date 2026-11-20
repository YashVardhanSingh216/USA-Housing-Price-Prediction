
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# 1. Page Configuration
# -----------------------------

st.set_page_config(
    page_title="USA Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# 2. Load Trained Model
# -----------------------------

@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

pipe = load_model()

# -----------------------------
# 3. Website Heading
# -----------------------------

st.title("🏠 USA Housing Price Predictor")

st.write(
    "Estimate house prices in the USA using "
    "a Machine Learning model."
)

st.divider()

# -----------------------------
# 4. User Input
# -----------------------------

st.subheader("Enter House Details")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input(
            "Average Area Income",
            min_value=0.0,
            value=65000.0
        )

        house_age = st.number_input(
            "Average House Age",
            min_value=0.0,
            value=5.0
        )

        rooms = st.number_input(
            "Average Number of Rooms",
            min_value=0.0,
            value=6.0
        )

    with col2:
        bedrooms = st.number_input(
            "Average Number of Bedrooms",
            min_value=0.0,
            value=3.0
        )

        population = st.number_input(
            "Area Population",
            min_value=0.0,
            value=35000.0
        )

    submit = st.form_submit_button(
        "Predict House Price",
        use_container_width=True
    )

# -----------------------------
# 5. Prediction
# -----------------------------

if submit:

    input_data = pd.DataFrame([{
        "Avg. Area Income": income,
        "Avg. Area House Age": house_age,
        "Avg. Area Number of Rooms": rooms,
        "Avg. Area Number of Bedrooms": bedrooms,
        "Area Population": population
    }])

    prediction = pipe.predict(input_data)[0]

    st.divider()

    st.subheader("Predicted House Price")

    st.success(f"${prediction:,.2f}")
