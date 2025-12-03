# budget_app
# 💰 Budget Prediction App  
A Streamlit web application that predicts a user's next-month **Total Expenses** using a trained **Random Forest Regressor**.  
The model uses expense details, demographics, and other financial factors to estimate the budget.

---

## 🚀 Features

### ✅ User-Friendly Web UI (Streamlit)
- Clean input form for all required financial details  
- Dropdowns for categorical data  
- Automatic handling of one-hot encoding  
- Real-time prediction  
- Shows processed user input  

### ✅ Machine Learning (Scikit-Learn)
- Random Forest model  
- One-hot encoding of categorical data  
- Feature alignment between training and user data  
- Dataset mean imputation for missing columns  

### ✅ Model Training
- Extracts features from expense, demographic, savings, and potential saving columns  
- Creates a target variable: **Total_Expenses = sum of all expense columns**  
- 80/20 train-test split  
- Displays model R² score  

---

## 📂 Project Structure

