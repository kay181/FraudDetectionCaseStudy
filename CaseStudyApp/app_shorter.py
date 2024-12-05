import pandas as pd 
import streamlit as st 
from scikit-learn import preprocessing
import pickle

# Load the model and encoder
model = pickle.load(open('/Users/kayvidushinie/Desktop/CaseStudy/model_top.pkl', 'rb'))

# Define the feature columns
cols = ['delinquency_status', 'number_of_credit_applications',
       'avg_balance_last_12months', 'fico_score', 'max_balance',
       'time_between_account_open_and_trade', 'debt_to_income_ratio',
       'credit_history_length', 'days_since_recent_trade', 'income_level']

# Define the feature names for the input form
feature_names = [col[1] for col in cols]

def main(): 
    st.title("Credit Risk Assessment")
    
    html_temp = """
    <div style="background:#E2725B ;padding:10px">
    <h2 style="color:white;text-align:center;">Fraud Prediction and Credit Risk</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Input fields for each feature (you can change the inputs to fit your use case)
    income_level = st.text_input("Income", "0") 
    fico_score = st.text_input("FICO Score", "0") 
    delinquency_status = st.text_input("Delinquency status (how late is payment in days)", "0") 
    number_of_credit_applications = st.text_input("Number of Credit Applications", "0") 
    debt_to_income_ratio = st.text_input("Debt to Income Ratio", "0") 
    max_balance = st.text_input("Max Balance", "0") 
    avg_balance_last_12months = st.text_input("Average Balance Last 12 months", "0") 
    days_since_recent_trade = st.text_input("Days Since Recent Trade", "0") 
    time_between_account_open_and_trade = st.text_input("Time Between Account Open and Trade", "0") 
    credit_history_length = st.text_input("Credit History Length", "0") 
    
    # When the predict button is clicked
    if st.button("Predict"): 
        features = [[income_level, fico_score, delinquency_status, 
                     number_of_credit_applications, debt_to_income_ratio, 
                     max_balance, avg_balance_last_12months,
                     days_since_recent_trade, time_between_account_open_and_trade, 
                     credit_history_length]]
        
        # Create a dataframe from the input data
        data = {'income_level': income_level, 'fico_score': fico_score, 
                'delinquency_status': delinquency_status, 
                'number_of_credit_applications': number_of_credit_applications, 
                'debt_to_income_ratio': debt_to_income_ratio, 
                'max_balance': max_balance, 
                'avg_balance_last_12months': avg_balance_last_12months, 
                'days_since_recent_trade': days_since_recent_trade, 
                'time_between_account_open_and_trade': time_between_account_open_and_trade, 
                'credit_history_length': credit_history_length}
        
        df = pd.DataFrame([list(data.values())], columns=cols)
        
        features_list = df.values.tolist()
        
        # Fraud prediction (0 or 1)
        fraud_prediction = model.predict(features_list)
        
        # Fraud probability (used to calculate credit risk score)
        fraud_probabilities = model.predict_proba(features_list)
        
        # Calculate credit risk score based on fraud probability (scaled between 1 and 1000)
        credit_risk_score = (fraud_probabilities[:, 1] * 999) + 1
        
        # Output fraud prediction and credit risk score
        fraud_text = "Fraud predicted: {}".format(fraud_prediction[0])  # 0 or 1
        credit_risk_text = "Credit Risk Score: {:.2f}".format(credit_risk_score[0])  # between 1 and 1000
        
        st.success(fraud_text)
        st.success(credit_risk_text)

if __name__ == '__main__': 
    main()
