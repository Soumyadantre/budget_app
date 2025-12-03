import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.set_page_config(page_title="Budget Prediction App", layout="wide")

st.title(" Budget Prediction App")
st.write("Enter your details below to predict next month's total expenses.")

# --------------------------
# Load dataset
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dat1a.csv")
    return df

df = load_data()

# --------------------------
# Define feature groups
# --------------------------
expense_cols = [
    'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
    'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare',
    'Education', 'Miscellaneous'
]

demo_cols = [
    'Income', 'Age', 'Dependents', 'Occupation', 'City_Tier'
]

savings_cols = [
    'Desired_Savings_Percentage', 'Desired_Savings', 'Disposable_Income'
]

potential_cols = [
    'Potential_Savings_Groceries', 'Potential_Savings_Transport',
    'Potential_Savings_Eating_Out', 'Potential_Savings_Entertainment',
    'Potential_Savings_Utilities', 'Potential_Savings_Healthcare',
    'Potential_Savings_Education', 'Potential_Savings_Miscellaneous'
]

all_features = expense_cols + demo_cols + savings_cols + potential_cols

# --------------------------
# Create target variable
# --------------------------
df['Total_Expenses'] = df[expense_cols].sum(axis=1)

# --------------------------
# One-hot encode training data
# --------------------------
df_encoded = pd.get_dummies(df[all_features + ['Total_Expenses']], drop_first=True)

X = df_encoded.drop('Total_Expenses', axis=1)
y = df_encoded['Total_Expenses']

mean_values = df_encoded.mean()

# --------------------------
# Train-test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# Train RandomForest
# --------------------------
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# --------------------------
# Show model accuracy
# --------------------------
pred = rf.predict(X_test)
st.sidebar.subheader("Model Performance")
st.sidebar.write("R² Score:", round(r2_score(y_test, pred), 4))


# --------------------------------------------
# USER INPUT FORM
# --------------------------------------------
st.header("📝 Enter Your Information")

with st.form("user_form"):
    st.subheader("Expenses")
    expense_input = {col: st.number_input(col, min_value=0.0, value=0.0) for col in expense_cols}

    st.subheader("Demographics")
    demo_input = {}
    demo_input['Income'] = st.number_input("Income", min_value=0.0, value=0.0)
    demo_input['Age'] = st.number_input("Age", min_value=1, value=25)
    demo_input['Dependents'] = st.number_input("Dependents", min_value=0, value=0)
    demo_input['Occupation'] = st.selectbox("Occupation", df['Occupation'].unique())
    demo_input['City_Tier'] = st.selectbox("City Tier", df['City_Tier'].unique())

    submitted = st.form_submit_button("Predict Budget")

if submitted:
    user_input = {**expense_input, **demo_input}

    user_df = pd.DataFrame([user_input])

    user_encoded = pd.get_dummies(user_df, drop_first=True)

    # Align columns
    for col in X.columns:
        if col not in user_encoded:
            user_encoded[col] = mean_values[col]

    user_encoded = user_encoded[X.columns]

    predicted_budget = rf.predict(user_encoded)[0]

    st.success(f"### ✅ Predicted Next Month Budget: **₹ {round(predicted_budget, 2)}**")

    # Optional: Show user data
    st.write("#### Your Processed Input Data")
    st.dataframe(user_encoded)
