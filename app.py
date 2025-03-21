import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model and preprocessor
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Set page config
st.set_page_config(
    page_title="Student Math Score Predictor",
    page_icon="📚",
    layout="wide"
)

# Add title and description
st.title("Student Math Score Predictor 📚")
st.markdown("""
This app predicts a student's math score based on various features like gender, race/ethnicity, 
parental education level, lunch type, and test preparation course.
""")

# Create input fields
st.subheader("Enter Student Information")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["female", "male"])
    race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parental_education = st.selectbox("Parental Level of Education", 
                                    ["some high school", "high school", "some college", 
                                     "associate's degree", "bachelor's degree", "master's degree"])

with col2:
    lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
    test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])
    reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=70)
    writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=70)

# Create a button for prediction
if st.button("Predict Math Score"):
    # Create input data
    input_data = pd.DataFrame({
        'gender': [gender],
        'race_ethnicity': [race_ethnicity],
        'parental_level_of_education': [parental_education],
        'lunch': [lunch],
        'test_preparation_course': [test_prep],
        'reading_score': [reading_score],
        'writing_score': [writing_score]
    })

    # Preprocess the input data
    input_processed = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(input_processed)[0]

    # Display the prediction
    st.subheader("Prediction Result")
    st.markdown(f"""
    ### Predicted Math Score: {prediction:.2f}
    """)

    # Add some insights
    st.subheader("Score Analysis")
    if prediction >= 90:
        st.success("Excellent performance! The student is likely to achieve an A grade.")
    elif prediction >= 80:
        st.info("Good performance! The student is likely to achieve a B grade.")
    elif prediction >= 70:
        st.warning("Average performance. The student might need additional support.")
    else:
        st.error("Below average performance. The student might need significant support.")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit</p>
    <p>Data Science Project - Student Math Performance Predictor</p>
</div>
""", unsafe_allow_html=True)
