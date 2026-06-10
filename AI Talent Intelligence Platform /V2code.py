# ==========================================
# 1. DEPENDENCIES & ENVIRONMENT SETUP
# ==========================================
# Ensure required language models and dataset support are installed
# Run these in your environment if needed:
# !python -m spacy download en_core_web_sm
# !pip install openpyxl pypdf beautifulsoup4 scikit-learn pandas numpy matplotlib spacy

from collections import Counter
import os
import re

import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import pandas as pd
from pypdf import PdfReader
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, cosine_similarity 
import spacy

# Load spaCy core NLP model
nlp = spacy.load("en_core_web_sm")

# ==========================================
# 2. FILE EXTRACTION & TEXT PARSING
# ==========================================
def extract_pdf_text(path):
    """Extract raw text from a target resume PDF file."""
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

resume_path = "data/Salma_Kasssem.pdf"
resume_text = extract_pdf_text(resume_path)

# ==========================================
# 3. TEXT CLEANING & NLP PREPROCESSING
# ==========================================
def clean_text_initial(text):
    """Applies initial cleaning using BeautifulSoup and standard regex routines."""
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\xa0', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if resume_text:
    clean_resume = clean_text_initial(resume_text)

    df_resume = pd.DataFrame({
        "Document": ["Resume"],
        "Raw_Text": [resume_text],
        "Clean_Text": [clean_resume]
    })

    # Deep linguistic processing via spaCy
    resume_doc = nlp(clean_resume)

    processed_tokens = [
        token.lemma_.lower().strip()
        for token in resume_doc
        if not token.is_stop 
        and not token.like_num 
        and token.is_alpha 
        and token.text.strip()
    ]
    processed_text = " ".join(processed_tokens)
else:
    raise FileNotFoundError(f"Could not load or extract text from {resume_path}")

# ==========================================
# 4. SINGLE TARGET ANALYSIS & VISUALIZATION
# ==========================================
documents = [processed_text]

# Generate Count Vectors
cv = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1, max_df=1.0)
X_cv = cv.fit_transform(documents)
df_vectors = pd.DataFrame(X_cv.toarray(), index=["Resume"], columns=cv.get_feature_names_out())

# Generate TF-IDF Matrix for the standalone document
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=1.0)
X_tfidf = tfidf_vectorizer.fit_transform(documents)
df_tfidf = pd.DataFrame(X_tfidf.toarray(), index=["Resume"], columns=tfidf_vectorizer.get_feature_names_out())

# Render inline visualization of highest weighted weights
top_tfidf_words = df_tfidf.sum().sort_values(ascending=True).tail(15)

plt.figure(figsize=(10, 6))
top_tfidf_words.plot(kind="barh", color="lightgreen")
plt.title("Top 15 Tokens by TF-IDF Weight", fontsize=14, fontweight="bold")
plt.xlabel("TF-IDF Score", fontsize=12)
plt.ylabel("Tokens / N-grams", fontsize=12)
plt.tight_layout()
plt.show()  

# ==========================================
# 5. TRAINING CLASSIFIERS ON REFERENCE DATASET
# ==========================================
df_dataset = pd.read_excel("data/resume_dataset_v2.xlsx")

# Drop blank entries and immediately reset index to keep data vectors aligned
df_dataset = df_dataset.dropna(subset=["resume_text", "role"]).reset_index(drop=True)
df_dataset = df_dataset[df_dataset["resume_text"].str.strip() != ""].reset_index(drop=True)
df_dataset = df_dataset[df_dataset["role"].str.strip() != ""].reset_index(drop=True)

X_text = df_dataset["resume_text"]
y = df_dataset["role"]

# Keep vocabulary focused on top 1000 items to control training dimensions
classifier_tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
X = classifier_tfidf.fit_transform(X_text)

# Stratified Split for Evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Initialize and train Model 1: Naive Bayes
model_nb = MultinomialNB(alpha=0.5, fit_prior=False)
model_nb.fit(X_train, y_train)

# Initialize and train Model 2: Logistic Regression
model_lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
model_lr.fit(X_train, y_train)

# ==========================================
# 6. PIPELINE PREDICTIONS & COSINE SIMILARITY
# ==========================================
candidate_name = os.path.splitext(os.path.basename(resume_path))[0]
X_target_resume = classifier_tfidf.transform([processed_text])

# Get separate predictions from each trained model
nb_prediction = model_nb.predict(X_target_resume)[0]
lr_prediction = model_lr.predict(X_target_resume)[0]

# Calculate cosine similarities against training matrix rows for the Vector Engine
raw_similarity_scores = cosine_similarity(X_target_resume, X).flatten()

analysis_df = pd.DataFrame({
    "role": df_dataset["role"],
    "similarity": raw_similarity_scores
})

# Exponential distribution normalization for match confidence scaling
role_averages = analysis_df.groupby("role")["similarity"].mean()
exp_averages = np.exp(role_averages - np.max(role_averages))
role_percentages = (exp_averages / np.sum(exp_averages)) * 100

df_role_metrics = pd.DataFrame({
    "Role Domain": role_percentages.index,
    "Match Confidence": role_percentages.values
}).sort_values(by="Match Confidence", ascending=False)

similarity_predicted_role = df_role_metrics.iloc[0]["Role Domain"]
similarity_confidence_score = df_role_metrics.iloc[0]["Match Confidence"]

# System Performance Reports
print("\n" + "="*75)
print(f" ADVANCED TALENT INTELLIGENCE PIPELINE REPORT: {candidate_name.upper()}")
print("="*75)
print("» COMPLETE FIT PERCENTAGE DISTRIBUTION BY ROLE DOMAIN (COSINE MATRIX):")
print("-"*75)

for _, row in df_role_metrics.iterrows():
    print(f"  • {row['Role Domain'].ljust(35)} : {row['Match Confidence']:.2f}%")
    
print("-"*75)
print(f"» NAIVE BAYES PREDICTION               : {nb_prediction.upper()}")
print(f"» LOGISTIC REGRESSION PREDICTION       : {lr_prediction.upper()}")
print(f"» VECTOR SIMILARITY PREDICTED DOMAIN   : {similarity_predicted_role.upper()}")
print(f"» COSINE ENGINE CONFIDENCE MATCH SCORE  : {similarity_confidence_score:.2f}%")
print("="*75 + "\n")

# ==========================================
# 7. PERFORMANCE METRICS & EVALUATION REPORT
# ==========================================
print("\n" + "="*75)
print(" SYSTEM MODEL EVALUATION COMPARISON")
print("="*75)

# 1. Benchmark Naive Bayes
y_pred_nb = model_nb.predict(X_test)
nb_acc = accuracy_score(y_test, y_pred_nb)
print(f"» Overall Naive Bayes Test Accuracy: {nb_acc:.2%}")

# 2. Benchmark Logistic Regression
y_pred_lr = model_lr.predict(X_test)
lr_acc = accuracy_score(y_test, y_pred_lr)
print(f"» Overall Logistic Regression Test Accuracy: {lr_acc:.2%}\n")

print("="*75)
print("DETAILED LOGISTIC REGRESSION BREAKDOWN REPORT:")
print("="*75)
print(classification_report(y_test, y_pred_lr, zero_division=0))
