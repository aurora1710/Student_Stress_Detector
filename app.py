import streamlit as st
import joblib
import pandas as pd


# Load the model and preprocessor
Log_Reg_model = joblib.load('Student_Stress_logistic_regression_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')


# Streamlit app title and description
st.title("Academic Stress Level Prediction")
st.write("Enter the student's information to predict their academic stress level.")

# Create input fields for each feature
# We need the original column names and their unique values to create the selectboxes
# Since we don't want to load the dataset, we'll need to manually define these or load them from a separate file if the dataset is large
# For this example, I will manually define them based on the original dataframe structure and unique values.
academic_stage_options = ['undergraduate', 'postgraduate'] # Replace with actual unique values from your data
study_environment_options = ['Noisy', 'Peaceful', 'disrupted'] # Replace with actual unique values from your data
coping_strategy_options = ['Analyze the situation and handle it with intellect', 'Social support (friends, family)', 'Time management strategies', 'Seeking professional help', 'Engaging in hobbies or recreational activities'] # Replace with actual unique values from your data
bad_habits_options = ['No', 'Yes'] # Replace with actual unique values from your data


academic_stage = st.selectbox("Your Academic Stage", academic_stage_options)
peer_pressure = st.slider("Peer Pressure (1-5)", 1, 5, 3)
academic_pressure_home = st.slider("Academic Pressure from Home (1-5)", 1, 5, 3)
study_environment = st.selectbox("Study Environment", study_environment_options)
coping_strategy = st.selectbox("What coping strategy you use as a student?", coping_strategy_options)
bad_habits = st.selectbox("Do you have any bad habits like smoking, drinking on a daily basis?", bad_habits_options)
academic_competition = st.slider("Rate your academic competition in your student life (1-5)", 1, 5, 3)


# Create a dictionary from the input values
input_data = {
    'Your_Academic_Stage': [academic_stage],
    'Peer_pressure': [peer_pressure],
    'Academic_pressure_from_your_home': [academic_pressure_home],
    'Study_Environment': [study_environment],
    'What_coping_strategy_you_use_as _a_student?': [coping_strategy],
    'Do_you_ have_any_bad_habits_like_smoking,_drinking_on_a_daily_basis?': [bad_habits],
    'What_would_you_rate_the_academic_competition_in_your_student_life': [academic_competition]
}

# Convert the dictionary to a DataFrame
input_df = pd.DataFrame(input_data)

# Preprocess the input data
input_scaled = preprocessor.transform(input_df)

# Make a prediction
prediction = Log_Reg_model.predict(input_scaled)

# Display the prediction
st.subheader("Predicted Academic Stress Index:")
st.write(prediction[0])
