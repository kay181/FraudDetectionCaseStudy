import pandas as pd 
import streamlit as st 
from sklearn import preprocessing
import pickle

# Load the model and encoder
model = pickle.load(open('/Users/kayvidushinie/Desktop/CaseStudy/model.pkl', 'rb'))
encoder_dict = pickle.load(open('/Users/kayvidushinie/Desktop/CaseStudy/encoder.pkl', 'rb')) 

# Define the feature columns
cols = ['age', 'income_level', 'fico_score', 'delinquency_status', 
        'number_of_credit_applications', 'debt_to_income_ratio', 
        'payment_methods_high_risk', 'max_balance', 
        'avg_balance_last_12months', 'number_of_defaulted_accounts', 
        'new_accounts_opened_last_12months', 
        'multiple_applications_short_time_period', 
        'unusual_submission_pattern', 
        'applications_submitted_during_odd_hours', 
        'watchlist_blacklist_flag', 'public_records_flag', 
        'location_encoded', 'occupation_encoded', 
        'days_since_recent_trade', 'time_between_account_open_and_trade', 
        'credit_history_length']

# Define the feature names for the input form
feature_names = [col[1] for col in cols]

# Define the categorical columns
categorical_cols = ['payment_methods_high_risk', 'multiple_applications_short_time_period',
                    'unusual_submission_pattern', 'applications_submitted_during_odd_hours',
                    'watchlist_blacklist_flag', 'public_records_flag', 'location_encoded',
                    'occupation_encoded']

def main(): 
    st.title("Credit Risk Assessment")
    
    html_temp = """
    <div style="background:#E2725B ;padding:10px">
    <h2 style="color:white;text-align:center;">Fraud Prediction and Credit Risk</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Input fields for each feature (you can change the inputs to fit your use case)
    age = st.text_input("Age", "0") 
    income_level = st.text_input("Income Level", "0") 
    fico_score = st.text_input("FICO Score", "0") 
    delinquency_status = st.text_input("Delinquency Status", "0") 
    number_of_credit_applications = st.text_input("Number of Credit Applications", "0") 
    debt_to_income_ratio = st.text_input("Debt to Income Ratio", "0") 
    payment_methods_high_risk = st.selectbox("Payment Methods High Risk", [0, 1, "Unknown"]) 
    max_balance = st.text_input("Max Balance", "0") 
    avg_balance_last_12months = st.text_input("Avg Balance Last 12 months", "0") 
    number_of_defaulted_accounts = st.text_input("Number of Defaulted Accounts", "0") 
    new_accounts_opened_last_12months = st.text_input("New Accounts Opened Last 12 months", "0") 
    multiple_applications_short_time_period = st.selectbox("Multiple Applications Short Time Period", [0, 1, "Unknown"]) 
    unusual_submission_pattern = st.selectbox("Unusual Submission Pattern", [0, 1, "Unknown"])
    applications_submitted_during_odd_hours = st.selectbox("Applications Submitted During Odd Hours", [0, 1, "Unknown"]) 
    watchlist_blacklist_flag = st.selectbox("Watchlist Blacklist Flag", [0, 1, "Unknown"]) 
    public_records_flag = st.selectbox("Public Records Flag", [0, 1, "Unknown"]) 
    location_encoded = st.text_input("Location Encoded", "0") 
    occupation_encoded = st.text_input("Occupation Encoded", "0") 
    days_since_recent_trade = st.text_input("Days Since Recent Trade", "0") 
    time_between_account_open_and_trade = st.text_input("Time Between Account Open and Trade", "0") 
    credit_history_length = st.text_input("Credit History Length", "0") 
    
    # When the predict button is clicked
    if st.button("Predict"): 
        features = [[age, income_level, fico_score, delinquency_status, 
                     number_of_credit_applications, debt_to_income_ratio, 
                     payment_methods_high_risk, max_balance, 
                     avg_balance_last_12months, number_of_defaulted_accounts, 
                     new_accounts_opened_last_12months, 
                     multiple_applications_short_time_period, 
                     unusual_submission_pattern, 
                     applications_submitted_during_odd_hours, 
                     watchlist_blacklist_flag, public_records_flag, 
                     location_encoded, occupation_encoded, 
                     days_since_recent_trade, time_between_account_open_and_trade, 
                     credit_history_length]]
        
        # Create a dataframe from the input data
        data = {'age': age, 'income_level': income_level, 'fico_score': fico_score, 
                'delinquency_status': delinquency_status, 
                'number_of_credit_applications': number_of_credit_applications, 
                'debt_to_income_ratio': debt_to_income_ratio, 
                'payment_methods_high_risk': payment_methods_high_risk, 
                'max_balance': max_balance, 
                'avg_balance_last_12months': avg_balance_last_12months, 
                'number_of_defaulted_accounts': number_of_defaulted_accounts, 
                'new_accounts_opened_last_12months': new_accounts_opened_last_12months, 
                'multiple_applications_short_time_period': multiple_applications_short_time_period, 
                'unusual_submission_pattern': unusual_submission_pattern, 
                'applications_submitted_during_odd_hours': applications_submitted_during_odd_hours, 
                'watchlist_blacklist_flag': watchlist_blacklist_flag, 
                'public_records_flag': public_records_flag, 
                'location_encoded': location_encoded, 'occupation_encoded': occupation_encoded, 
                'days_since_recent_trade': days_since_recent_trade, 
                'time_between_account_open_and_trade': time_between_account_open_and_trade, 
                'credit_history_length': credit_history_length}
        
        df = pd.DataFrame([list(data.values())], columns=cols)
        
        # Encoding categorical columns
        for cat in categorical_cols:
            le = preprocessing.LabelEncoder()
            if cat in df.columns:
                le.classes_ = encoder_dict[cat]
                df[cat] = df[cat].apply(lambda x: x if x in le.classes_ else 'Unknown')
                le.fit(df[cat])
                df[cat] = le.transform(df[cat])
        
        # Convert the dataframe to a list for prediction
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

