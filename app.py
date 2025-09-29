import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the model and scaler
Log_Reg_model = joblib.load('Student_Stress_logistic_regression_model.pkl')
# Assuming the scaler was saved after fitting on the entire X, including categorical features handled by the preprocessor pipeline
# We need to recreate the preprocessor pipeline to handle both numerical and categorical features
# We will load the original data to fit the preprocessor
df = pd.read_csv('/content/academic Stress level - maintainance.csv')
df.columns = df.columns.str.strip()
X = df.drop('Rate_your_academic_stress_index', axis=1)

categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Fit the preprocessor on the original data
preprocessor.fit(X)


# Streamlit app title and description
st.title("Academic Stress Level Prediction")
st.write("Enter the student's information to predict their academic stress level.")

# Create input fields for each feature
academic_stage = st.selectbox("Your Academic Stage", X['Your_Academic_Stage'].unique())
peer_pressure = st.slider("Peer Pressure (1-5)", 1, 5, 3)
academic_pressure_home = st.slider("Academic Pressure from Home (1-5)", 1, 5, 3)
study_environment = st.selectbox("Study Environment", X['Study_Environment'].unique())
coping_strategy = st.selectbox("What coping strategy you use as a student?", X['What_coping_strategy_you_use_as _a_student?'].unique())
bad_habits = st.selectbox("Do you have any bad habits like smoking, drinking on a daily basis?", X['Do_you_ have_any_bad_habits_like_smoking,_drinking_on_a_daily_basis?'].unique())
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
