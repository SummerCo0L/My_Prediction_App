import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Load your models and pipelines
model_1 = joblib.load('model/final_GBT_model_mediated.joblib')
pipeline_1 = joblib.load('model/pipeline_1.joblib')
model_2 = joblib.load('model/final_XGB_model_settled.joblib')
pipeline_2 = joblib.load('model/pipeline_2.joblib')

# Create the Streamlit interface
st.title('Community Mediation Centre (CMC) Case Prediction App')

# Input field for Date of Registration
date_of_registration = st.date_input('Date of Case Registration at CMC', \
                                     min_value=datetime(2000, 1, 1), max_value=datetime(2030, 12, 31))

# Extract features from the entered date
month = date_of_registration.month
day_of_week = date_of_registration.weekday()  # Monday=0, Sunday=6
is_weekend = 1 if day_of_week in [5, 6] else 0  # 1 if Saturday or Sunday


# Input fields for other features
type_of_intake = st.selectbox('Type of Intake', ['External Agency Referrals - Housing Development Board (HDB)',
       'External Agency Referrals - Singapore Police Force (SPF)',
       'External Agency Referrals - Member of Parliament',
       'External Agency Referrals - Town Council',
       'External Agency Referrals - Others',
       ])
type_of_dispute = st.selectbox('Type of Dispute', ['Friends', 'Neighbour', 'Family', 'Others',
       ])


if date_of_registration and type_of_intake and type_of_dispute:
    input_data = pd.DataFrame([[month, day_of_week, is_weekend, type_of_intake, type_of_dispute]],
                              columns=['month', 'day_of_week', 'is_weekend', 'type_of_intake', 'type_of_dispute'])
    
    # Mediation prediction
    X_transformed_mediation = pipeline_1.transform(input_data)
    prediction_proba_mediation = model_1.predict_proba(X_transformed_mediation)
    mediation_probability = prediction_proba_mediation[0][1]  # Probability for mediation
    mediation_probability_percent = (mediation_probability * 100).round()

    if mediation_probability_percent < 20:
        st.write(f'The case is very unlikely to proceed to mediation. \
        ({mediation_probability_percent}% chance)')
    elif 20 <= mediation_probability_percent < 40:
        st.write(f'The case is unlikely to proceed to mediation. \
        ({mediation_probability_percent}% chance)')
    elif 40 <= mediation_probability_percent < 60:
        st.write(f'It is difficult to tell if the case will proceed to mediation. \
        ({mediation_probability_percent}% chance)')
    elif 60 <= mediation_probability_percent < 80:
        st.write(f'The case is likely to proceed to mediation. \
        ({mediation_probability_percent}% chance)')
    else:
        st.write(f'The case is very likely to proceed to mediation. \
        ({mediation_probability_percent}% chance)')

    # Settlement prediction
    X_transformed_settlement = pipeline_2.transform(input_data)
    prediction_proba_settlement = model_2.predict_proba(X_transformed_settlement)
    settlement_probability = prediction_proba_settlement[0][1]  # Probability for settlement
    settlement_probability_percent = (settlement_probability * 100).round()

    if settlement_probability_percent < 20:
        st.write(f'The case is very unlikely to be settled if it proceeds to mediation. \
        ({settlement_probability_percent}% chance)')
    elif 20 <= settlement_probability_percent < 40:
        st.write(f'The case is unlikely to be settled if it proceeds to mediation. \
        ({settlement_probability_percent}% chance)')
    elif 40 <= settlement_probability_percent < 60:
        st.write(f'It is difficult to tell if the case will be settled if it proceeds to mediation. \
        {settlement_probability_percent}% chance)')
    elif 60 <= settlement_probability_percent < 80:
        st.write(f'The case is likely to be settled if it proceeds to mediation. \
        ({settlement_probability_percent}% chance)')
    else:
        st.write(f'The case is very likely to be settled if it proceeds to mediation. \
        ({settlement_probability_percent}% chance)')

    # Overall probability of mediation success
    overall_probability = mediation_probability * settlement_probability
    overall_probability_percent = (overall_probability * 100).round()

    if overall_probability_percent < 20:
        st.write(f'The case is very unlikely to be mediated and settled from the point of registration. \
        ({overall_probability_percent}% chance)')
    elif 20 <= overall_probability_percent < 40:
        st.write(f'The case is unlikely to be mediated and settled from the point of registration. \
        ({overall_probability_percent}% chance)')
    elif 40 <= overall_probability_percent < 60:
        st.write(f'It is difficult to tell if the case will be settled and settled from the point of registration. \
        ({overall_probability_percent}% chance)')
    elif 60 <= overall_probability_percent < 80:
        st.write(f'The case is likely to be mediated and settled from the point of registration. \
        ({overall_probability_percent}% chance)')
    else:
        st.write(f'The case is very likely to be mediated and settled from the point of registration. \
        ({overall_probability_percent}% chance)')