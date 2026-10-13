
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("Bank Churn Prediction")
st.write(
    "Enter the customer's details below to predict "
    "whether they are likely to leave the bank."
)



@st.cache_resource
def load_artifacts():
    with open("StandardScaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("OneHotEncoding.pkl", "rb") as f:
        ohe = pickle.load(f)

    with open("LabelEncoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    model = load_model("model.h5", compile=False)

    return scaler, ohe, label_encoder, model


try:
    scaler, ohe, label_encoder, model = load_artifacts()
except Exception as e:
    st.error(f"Could not load the model or preprocessing files: {e}")
    st.stop()


# This order must match the columns used to fit StandardScaler.
FEATURE_COLUMNS = [
    "CreditScore",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Geography_France",
    "Geography_Germany",
    "Geography_Spain"
]



with st.form("customer_form"):

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=0,
            max_value=1000,
            value=600,
            step=1
        )

        geography = st.selectbox(
            "Geography",
            options=["France", "Germany", "Spain"],
            index=None,
            placeholder="Select country"
        )

        gender = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            index=None,
            placeholder="Select gender"
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=120,
            value=40,
            step=1
        )

        tenure = st.number_input(
            "Tenure (years)",
            min_value=0,
            max_value=10,
            value=3,
            step=1
        )

    with col2:
        balance = st.number_input(
            "Account Balance",
            min_value=0.0,
            value=60000.0,
            step=1000.0
        )

        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=4,
            value=2,
            step=1
        )

        has_card = st.selectbox(
            "Has Credit Card?",
            options=["Yes", "No"],
            index=None,
            placeholder="Select an option"
        )

        active_member = st.selectbox(
            "Is Active Member?",
            options=["Yes", "No"],
            index=None,
            placeholder="Select an option"
        )

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        type="primary",
        use_container_width=True
    )


if submitted:

    if geography is None or gender is None:
        st.error("Please select both Geography and Gender.")

    elif has_card is None or active_member is None:
        st.error("Please answer both Yes/No questions.")

    elif gender not in label_encoder.classes_:
        st.error("Gender value is not supported by the saved encoder.")

    else:
        try:
            # Build one customer's input
            input_data = {
                "CreditScore": credit_score,
                "Gender": int(
                    label_encoder.transform([gender])[0]
                ),
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": 1 if has_card == "Yes" else 0,
                "IsActiveMember": (
                    1 if active_member == "Yes" else 0
                ),
                "EstimatedSalary": estimated_salary
            }

            # One-hot encode Geography using the fitted encoder
            geo_values = ohe.transform([[geography]])
            geo_columns = ohe.get_feature_names_out(
                ["Geography"]
            )

            for i, column in enumerate(geo_columns):
                input_data[column] = float(geo_values[0][i])


            df = pd.DataFrame([input_data])

            
            df = df.reindex(columns=FEATURE_COLUMNS)

            if df.isnull().any().any():
                raise ValueError(
                    "A required feature is missing. "
                    "Check the training feature names."
                )

           
            final_input = scaler.transform(df)

            
            probability = float(
                model.predict(final_input, verbose=0)[0][0]
            )

            if not np.isfinite(probability):
                raise ValueError("The model returned an invalid probability.")

            if not 0 <= probability <= 1:
                raise ValueError(
                    "The model output is outside the expected probability range."
                )

            
            st.divider()
            st.subheader("Prediction Result")

            st.metric(
                label="Churn Probability",
                value=f"{probability:.2%}"
            )

            st.progress(probability)

            if probability >= 0.5:
                st.error("Prediction: Likely to Leave")
            else:
                st.success("Prediction: Likely to Stay")

            st.caption(
                "This is a model estimate, not a guarantee. "
                "The 0.5 threshold is a starting choice."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
