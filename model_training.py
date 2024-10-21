# model_training.py
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load the SpaCy English model
nlp = spacy.load('en_core_web_sm')

# Load dataset
data = pd.read_csv('C:/Users/Vivek Vasani/Desktop/resume_screening_app/UpdatedResumeDataSet.csv')

# Preprocessing function for resumes
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

# Preprocess the resume data
data['Cleaned_Resume'] = data['Resume'].apply(preprocess_text)

# Vectorize the text data
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['Cleaned_Resume']).toarray()
y = data['Category']  # Adjust this to match your dataset's column name for categories

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=sorted(data['Category'].unique()))
print(report)

# Save the trained model and vectorizer as a .pkl file
with open("model.pkl", "wb") as model_file:
    pickle.dump((model, vectorizer), model_file)

print("Model and vectorizer saved to model.pkl")
