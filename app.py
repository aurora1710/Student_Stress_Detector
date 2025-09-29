import streamlit as st
import joblib
import pandas as pd
import numpy as np # Import numpy for array manipulation
import pickle
# Load the model
with open('Student_Stress_logistic_regression_model.pkl', 'rb') as f:
    Log_Reg_model = pickle.load(f)

# --- Manual Preprocessing Setup (Requires knowledge from training data) ---
# Define the expected order of features after manual preprocessing.
# This order MUST match the order of features the model was trained on.
# You need to replace these with the actual values from your training data.
# Example: Numerical features first, then one-hot encoded categorical features in a specific order.

# Placeholder for numerical feature scaling parameters (mean and std deviation from training)
# ** Replace these with the actual mean and standard deviation from your training data **
numerical_means = {'Peer_pressure': 0, 'Academic_pressure_from_your_home': 0, 'What_would_you_rate_the_academic_competition_in_your_student_life': 0} # Replace with actual means
numerical_stds = {'Peer_pressure': 1, 'Academic_pressure_from_your_home': 1, 'What_would_you_rate_the_academic_competition_in_your_student_life': 1} # Replace with actual std deviations

# Define the unique categories for one-hot encoding and their expected order in the final feature vector
# ** Replace these with the actual unique categories and their order from your training data's OneHotEncoder output **
categorical_features_info = {
    'Your_Academic_Stage': ['undergraduate', 'postgraduate'],
    'Study_Environment': ['Noisy', 'Peaceful', 'disrupted'],
    'What_coping_strategy_you_use_as _a_student?': ['Analyze the situation and handle it with intellect', 'Social support (friends, family)', 'Time management strategies', 'Seeking professional help', 'Engaging in hobbies or recreational activities'],
    'Do_you_ have_any_bad_habits_like_smoking,_drinking_on_a_daily_basis?': ['No', 'Yes']
}

# Define the exact order of all features as expected by the model
# This is crucial and error-prone if not precisely matched to the training data's processed features.
# Example (replace with your actual feature order):
expected_feature_order = [
    'Peer_pressure',
    'Academic_pressure_from_your_home',
    'What_would_you_rate_the_academic_competition_in_your_student_life',
    'Your_Academic_Stage_undergraduate',
    'Your_Academic_Stage_postgraduate',
    'Study_Environment_Noisy',
    'Study_Environment_Peaceful',
    'Study_Environment_disrupted',
    'What_coping_strategy_you_use_as _a_student?_Analyze the situation and handle it with intellect',
    'What_coping_strategy_you_use_as _a_student?_Social support (friends, family)',
    'What_coping_strategy_you_use_as _a_student?_Time management strategies',
    'What_coping_strategy_you_use_as _a_student?_Seeking professional help',
    'What_coping_strategy_you_use_as _a_student?_Engaging in hobbies or recreational activities',
    'Do_you_ have_any_bad_habits_like_smoking,_drinking_on_a_daily_basis?_No',
    'Do_you_ have_any_bad_habits_like_smoking,_drinking_on_a_daily_basis?_Yes'
]


# Streamlit app title and description
st.title("Academic Stress Level Prediction")
st.write("Enter the student's information to predict their academic stress level.")

# Create input fields for each feature
academic_stage = st.selectbox("Your Academic Stage", categorical_features_info['Your_Academic_Stage'])
peer_pressure = st.slider("Peer Pressure (1-5)", 1, 5, 3)
academic_pressure_home = st.slider("Academic Pressure from Home (1-5)", 1, 5, 3)
study_environment = st.selectbox("Study Environment", categorical_features_info['Study_Environment'])
coping_strategy = st.selectbox("What coping strategy you use as a student?", categorical_features_info['What_coping_strategy_you_use_as _a_student?'])
bad_habits = st.selectbox("Do you have any bad habits like smoking, drinking on a daily basis?", categorical_features_info['Do_you_ have_any_bad_habits_like_smoking,_drinking_on_a_daily_basis?'])
academic_competition = st.slider("Rate your academic competition in your student life (1-5)", 1, 5, 3)


# --- Manual Preprocessing of Input ---
# Create a dictionary from the input values
input_data = {
    'Your_Academic_Stage': academic_stage,
    'Peer_pressure': peer_pressure,
    'Academic_pressure_from_your_home': academic_pressure_home,
    'Study_Environment': study_environment,
    'What_coping_strategy_you_use_as _a_student?': coping_strategy,
    'Do_you_ have_any_bad_habits_like_smoking,_drinking_on_a_daily_basis?': bad_habits,
    'What_would_you_rate_the_academic_competition_in_your_student_life': academic_competition
}

# Manually create the feature vector in the expected order and apply scaling
processed_input = []
for feature_name in expected_feature_order:
    if feature_name in numerical_means: # It's a numerical feature
        # Apply standard scaling
        scaled_value = (input_data[feature_name] - numerical_means[feature_name]) / numerical_stds[feature_name]
        processed_input.append(scaled_value)
    else: # It's a one-hot encoded categorical feature
        # Determine the original categorical column name and the category
        original_col, category = feature_name.rsplit('_', 1)
        # Handle the case where the category name might contain underscores
        if original_col not in input_data:
             # Try a different split if the simple rsplit didn't work
             for col_name in categorical_features_info.keys():
                 if feature_name.startswith(col_name):
                     original_col = col_name
                     category = feature_name[len(col_name)+1:] # Get the category after the column name and underscore
                     break

        if input_data.get(original_col) == category:
            processed_input.append(1)
        else:
            processed_input.append(0)

# Convert the processed input list to a numpy array with the correct shape (1 sample, n features)
input_for_prediction = np.array([processed_input])


# Make a prediction
prediction = Log_Reg_model.predict(input_for_prediction)

# Display the prediction
st.subheader("Predicted Academic Stress Index:")
st.write(prediction[0])
