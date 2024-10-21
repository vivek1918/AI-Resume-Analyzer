# app.py
import streamlit as st
import PyPDF2  # For PDF file processing
import re  # For skill extraction
import spacy
import pickle  # To load the model and vectorizer

# Load the SpaCy English model
nlp = spacy.load('en_core_web_sm')

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to extract skills from the resume text
def extract_skills(text):
    skill_keywords = [
        'Python', 'Java', 'C++', 'SQL', 'JavaScript', 'HTML', 'CSS', 'Machine Learning', 
        'Data Science', 'Deep Learning', 'NLP', 'Django', 'Flask', 'React', 'Node.js', 'Git', 'AWS',
        'Docker', 'Kubernetes', 'TensorFlow', 'PyTorch', 'Excel', 'Tableau', 'Power BI', 'Figma'
    ]
    found_skills = []
    for skill in skill_keywords:
        if re.search(r'\b' + skill + r'\b', text, re.IGNORECASE):
            found_skills.append(skill)
    return found_skills

# Streamlit Web App
st.title("AI-based Resume Screening Tool")

st.write("Upload a resume (in plain text or PDF format) to check its suitability:")

# File uploader to accept both txt and pdf files
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])

# Load the saved model and vectorizer
with open("model.pkl", "rb") as model_file:
    model, vectorizer = pickle.load(model_file)

if uploaded_file is not None:
    # Process the uploaded file
    if uploaded_file.name.endswith(".txt"):
        resume_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(uploaded_file)
    
    # Extract skills from the resume
    skills = extract_skills(resume_text)
    st.write("Extracted Skills:")
    st.write(", ".join(skills) if skills else "No skills found.")

    # Preprocess the resume and vectorize it
    cleaned_resume = preprocess_text(resume_text)
    resume_vector = vectorizer.transform([cleaned_resume]).toarray()

    # Make predictions using the trained model
    prediction = model.predict(resume_vector)
    prediction_proba = model.predict_proba(resume_vector)

    # Display the result and confidence
    st.write(f"Predicted Category: {prediction[0]}")
    st.write(f"Confidence: {max(prediction_proba[0]) * 100:.2f}%")
