import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import base64
from pathlib import Path

# Load the trained model and TF-IDF vectorizer
model = joblib.load(r'C:\Users\jiaji\OneDrive - Universiti Malaya\Share document with PC\4. SEM 1 - 2024\P2\best_rf_model.pkl')
tfidf_vectorizer = joblib.load(r'C:\Users\jiaji\OneDrive - Universiti Malaya\Share document with PC\4. SEM 1 - 2024\P2\job_title_tfidf_vectorizer.pkl')

# Function to load image as base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Load image in base64
image_path = 'C:/Users/jiaji/OneDrive - Universiti Malaya/Share document with PC/4. SEM 1 - 2024/P2/dataproduct background.jpg'
image_base64 = image_to_base64(image_path)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('data:image/jpeg;base64,{image_base64}');
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
    }}
    /* Title in yellow */
    h1 {{
        color: yellow !important;
    }}
    /* Subheader in white */
    .stSubheader {{
        color: yellow !important;
    }}
    /* Text in white (for general text and markdown) */
    .stText, .stMarkdown {{
        color: white !important;
    }}
    /* Make input fields and dropdown labels white */
    .stTextInput label, .stSelectbox label, .stButton, .stSlider label {{
        color: white !important;
    }}
    /* Add transparent background box for inputs with dark color */
    .stTextInput, .stSelectbox, .stButton, .stSlider {{
        background-color: rgba(0, 0, 0, 0.5) !important;  /* Semi-transparent dark background */
        border-radius: 5px;
        padding: 10px;
    }}
    /* Input and dropdown text color */
    .stTextInput input, .stSelectbox select, .stButton, .stSlider input {{
        color: white;  /* Set text color in inputs to white */
    }}
    /* Placeholder text for inputs */
    .stTextInput input::placeholder {{
        color: grey;
    }}
    /* Button text color */
    .stButton {{
        color: red !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Streamlit app title
st.title("Salary Prediction Tool 2024")
# Show model performance metrics with a more informative intro
st.write(
    "This machine learning model has been trained to predict salaries based on various factors, including region, experience level, and job title."
    " After thorough evaluation, the model has achieved an accuracy of 77%. This means the model successfully predicts salary trends with a reasonable degree of reliability."
)

st.write(
    "Provide the information to predict the salary."
)

# Dropdown for selecting Region
region = st.selectbox(
    "Select Region",
    ["North America", "Europe", "Asia", "Other"]
)

# Dropdown for selecting Experience Level
experience_level = st.selectbox(
    "Select Experience Level",
    ["Entry Level", "Mid Level", "Senior Level", "Executive"]
)

# Text input for entering Job Title
job_title = st.text_input("Enter Job Title")

# Fixed Parameter: Work Year
work_year = 2024


# Create a button to predict salary
if st.button("Predict Salary"):
    # One-hot encoding for region
    region_list = ["North America", "Europe", "Asia", "Other"]
    region_one_hot = [1 if r == region else 0 for r in region_list]

    # Experience Level encoding
    experience_dict = {"Entry Level": 0, "Mid Level": 1, "Senior Level": 2, "Executive": 3}
    experience_numeric = experience_dict.get(experience_level, 0)

    # Preprocess the entered job title (lowercasing)
    job_title_cleaned = job_title.lower()

    # Apply the same TF-IDF vectorizer to transform the input job title
    job_title_tfidf = tfidf_vectorizer.transform([job_title_cleaned]).toarray()

    # Flatten the TF-IDF vector to include it in the input features
    job_title_tfidf_flat = job_title_tfidf.flatten()

    # Prepare input features: one-hot encoded region, experience level, job title TF-IDF, and fixed work year
    input_features = np.array([region_one_hot + [experience_numeric] + list(job_title_tfidf_flat) + [work_year]])

    # Make prediction using the trained model
    predicted_salary_log = model.predict(input_features)[0]  # If log-transformation was applied to the target during training

    # If the target salary was log-transformed during training, reverse the log transformation (e.g., np.exp())
    predicted_salary = np.exp(predicted_salary_log)

    # Display the predicted salary
    st.write(f"Predicted Salary in USD: ${predicted_salary:,.2f}")




