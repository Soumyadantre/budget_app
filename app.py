import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------
# PAGE SETTINGS
# ----------------------------------------------------
st.set_page_config(
    page_title="Next Month Budget Predictor",
    layout="wide",
    page_icon="💰"
)

st.title("💰 Next Month Budget Prediction App")
st.write("Enter your monthly expenses and get an AI-powered prediction of next month's budget.")


# ----------------------------------------------------
# LOAD MODEL + DATA
# ----------------------------------------------------
@st.cache_resource
def load_model():
   

    if os.path.exists("budget_model.joblib"):
         model = joblib.load("budget_model.joblib")
    else:
      st.warning("Model not found — training a new model...")
      df = pd.read_csv("processed_budget_data.csv")

   feature_cols = [
        'Income','Age','Dependents','City_Tier','Rent','Loan_Repayment',
        'Insurance','Groceries','Transport','Eating_Out','Entertainment',
        'Utilities','Healthcare','Education','Miscellaneous',
        'Total_Expenses','Lag_1','Lag_2','MA_3'
    ]

    X = df[feature_cols]
    y = df["Next_Month_Budget"]

    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "budget_model.joblib")

    st.success("Model trained successfully!")

    return model

@st.cache_data
def load_processed_data():
    df = pd.read_csv("processed_budget_data.csv")
    return df

model = load_model()
df = load_processed_data()


# ----------------------------------------------------
# USER INPUT FORM
# ----------------------------------------------------
st.header("📝 Enter Your Monthly Details")

col1, col2, col3 = st.columns(3)

with col1:
    Income = st.number_input("Income", min_value=0, value=50000)
    Age = st.number_input("Age", min_value=18, value=25)
    Dependents = st.number_input("Dependents", min_value=0, value=0)
    City_Tier = st.selectbox("City Tier", [1, 2, 3])

with col2:
    Rent = st.number_input("Rent", min_value=0, value=12000)
    Loan_Repayment = st.number_input("Loan Repayment", min_value=0, value=0)
    Insurance = st.number_input("Insurance", min_value=0, value=1500)
    Groceries = st.number_input("Groceries", min_value=0, value=4000)
    Transport = st.number_input("Transport", min_value=0, value=2000)

with col3:
    Eating_Out = st.number_input("Eating Out", min_value=0, value=1500)
    Entertainment = st.number_input("Entertainment", min_value=0, value=1200)
    Utilities = st.number_input("Utilities", min_value=0, value=2500)
    Healthcare = st.number_input("Healthcare", min_value=0, value=1000)
    Education = st.number_input("Education", min_value=0, value=0)
    Miscellaneous = st.number_input("Miscellaneous", min_value=0, value=800)


# ----------------------------------------------------
# FEATURE ENGINEERING (same as in training)
# ----------------------------------------------------
def make_features():
    Total_Expenses = (
        Rent + Loan_Repayment + Insurance + Groceries + Transport +
        Eating_Out + Entertainment + Utilities + Healthcare +
        Education + Miscellaneous
    )

    Lag_1 = df["Total_Expenses"].iloc[-1]      # last known month
    Lag_2 = df["Total_Expenses"].iloc[-2]      # second last month
    MA_3  = df["Total_Expenses"].tail(3).mean()

    features = pd.DataFrame([[
        Income, Age, Dependents, City_Tier,
        Rent, Loan_Repayment, Insurance, Groceries, Transport,
        Eating_Out, Entertainment, Utilities, Healthcare,
        Education, Miscellaneous,
        Total_Expenses, Lag_1, Lag_2, MA_3
    ]], columns=[
        'Income','Age','Dependents','City_Tier',
        'Rent','Loan_Repayment','Insurance','Groceries','Transport',
        'Eating_Out','Entertainment','Utilities','Healthcare',
        'Education','Miscellaneous',
        'Total_Expenses','Lag_1','Lag_2','MA_3'
    ])

    return features, Total_Expenses

features, total_expenses = make_features()


# ----------------------------------------------------
# PREDICT BUTTON
# ----------------------------------------------------
if st.button("🔮 Predict Next Month Budget"):
    prediction = model.predict(features)[0]

    st.success(f"### ✅ Predicted Next Month Budget: **₹ {prediction:,.2f}**")

    # Pie Chart of current expenses
    exp_dict = {
        "Rent": Rent,
        "Loan_Repayment": Loan_Repayment,
        "Insurance": Insurance,
        "Groceries": Groceries,
        "Transport": Transport,
        "Eating_Out": Eating_Out,
        "Entertainment": Entertainment,
        "Utilities": Utilities,
        "Healthcare": Healthcare,
        "Education": Education,
        "Miscellaneous": Miscellaneous
    }

    st.subheader("📊 Your Current Month Expense Breakdown")
    fig, ax = plt.subplots()
    ax.pie(exp_dict.values(), labels=exp_dict.keys(), autopct="%1.1f%%")
    st.pyplot(fig)

    # Comparison Bar Chart
    st.subheader("📈 Comparison: This Month vs AI Predicted Next Month")
    fig2, ax2 = plt.subplots()
    ax2.bar(["This Month", "Next Month (Predicted)"], [total_expenses, prediction])
    ax2.set_ylabel("Amount (₹)")
    st.pyplot(fig2)


# ----------------------------------------------------
# SHOW RAW DATA (optional)
# ----------------------------------------------------
st.sidebar.header("📂 Raw Dataset")
if st.sidebar.checkbox("Show Processed Data"):
    st.sidebar.write(df)
