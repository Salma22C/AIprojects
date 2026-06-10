# ==========================================
# 1. DEPENDENCIES & ENVIRONMENT SETUP
# ==========================================
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from pypdf import PdfReader

import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
# Load spaCy core NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    import sys
    print("spaCy model missing. Downloading en_core_web_sm...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# ==========================================
# 2. CORE TEXT PIPELINE MODULES
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
        print(f"Error reading PDF {path}: {e}")
        return None


def clean_text_initial(text):
    """Applies initial cleaning using BeautifulSoup and standard regex routines."""
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\xa0', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_linguistic(clean_text):
    """Performs deep linguistic lemmatization, stopword processing, and token filtering."""
    doc = nlp(clean_text)
    processed_tokens = [
        token.lemma_.lower().strip()
        for token in doc
        if not token.is_stop 
        and not token.like_num 
        and token.is_alpha 
        and token.text.strip()
    ]
    return " ".join(processed_tokens)


# ==========================================
# 3. EXTRACT & INITIALIZE TARGET PROFILE
# ==========================================
resume_path = "data/Salma_Kasssem.pdf"
resume_text = extract_pdf_text(resume_path)

if resume_text:
    clean_resume = clean_text_initial(resume_text)
    processed_resume_text = preprocess_linguistic(clean_resume)
    
    df_resume = pd.DataFrame({
        "Document": ["Resume"],
        "Raw_Text": [resume_text],
        "Clean_Text": [clean_resume]
    })
else:
    raise FileNotFoundError(f"CRITICAL ERROR: Failed to extract text from {resume_path}")


# ==========================================
# 4. SINGLE TARGET DIAGNOSTIC VISUALIZATION
# ==========================================
# Look closely at vocabulary profile using standalone parameters
diagnostic_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=1.0)
X_diag = diagnostic_vectorizer.fit_transform([processed_resume_text])
df_diag = pd.DataFrame(X_diag.toarray(), index=["Resume"], columns=diagnostic_vectorizer.get_feature_names_out())

top_tfidf_words = df_diag.sum().sort_values(ascending=True).tail(15)

plt.figure(figsize=(10, 6))
top_tfidf_words.plot(kind="barh", color="lightgreen")
plt.title("Top 15 Tokens by Standalone Target TF-IDF Weight", fontsize=12, fontweight="bold")
plt.xlabel("TF-IDF Score", fontsize=10)
plt.ylabel("Tokens / N-grams", fontsize=10)
plt.tight_layout()
plt.show()  


# ==========================================
# 5. DATA LOADING & VECTOR PIPELINE UNIFICATION
# ==========================================
df_dataset = pd.read_excel("data/resume_dataset_v2.xlsx")

# Drop completely blank entries and force realign indices
df_dataset = df_dataset.dropna(subset=["resume_text", "role"]).reset_index(drop=True)
df_dataset = df_dataset[df_dataset["resume_text"].str.strip() != ""].reset_index(drop=True)
df_dataset = df_dataset[df_dataset["role"].str.strip() != ""].reset_index(drop=True)

X_text_corpus = df_dataset["resume_text"]
y_labels = df_dataset["role"]

# The Unified Core Vectorizer (Max 1000 items helps control training sparse dimension size)
classifier_tfidf = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
X_corpus_sparse = classifier_tfidf.fit_transform(X_text_corpus)

# Project our target resume through the unified vectorizer space cleanly
X_target_resume = classifier_tfidf.transform([processed_resume_text])


# ==========================================
# 6. MODEL TRAINING & SPLIT EVALUATION
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_corpus_sparse, y_labels, test_size=0.25, random_state=42, stratify=y_labels
)

# Initialize and train Model 1: Naive Bayes
model_nb = MultinomialNB(alpha=0.5, fit_prior=False)
model_nb.fit(X_train, y_train)

# Initialize and train Model 2: Logistic Regression
model_lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
model_lr.fit(X_train, y_train)


# ==========================================
# 7. COSINE SIMILARITY ENGINE
# ==========================================
candidate_name = os.path.splitext(os.path.basename(resume_path))[0]

# Calculate metrics across full unified dataset rows
raw_similarity_scores = cosine_similarity(X_target_resume, X_corpus_sparse).flatten()

analysis_df = pd.DataFrame({
    "role": df_dataset["role"],
    "similarity": raw_similarity_scores
})

# Normalize score weights using exponential scaling parameters
role_averages = analysis_df.groupby("role")["similarity"].mean()
exp_averages = np.exp(role_averages - np.max(role_averages))
role_percentages = (exp_averages / np.sum(exp_averages)) * 100

df_role_metrics = pd.DataFrame({
    "Role Domain": role_percentages.index,
    "Match Confidence": role_percentages.values
}).sort_values(by="Match Confidence", ascending=False)

similarity_predicted_role = df_role_metrics.iloc[0]["Role Domain"]
similarity_confidence_score = df_role_metrics.iloc[0]["Match Confidence"]


# ==========================================
# 8. OUTPUT PRODUCTION INTELLIGENCE REPORTS
# ==========================================
print("\n" + "="*75)
print(f" ADVANCED TALENT INTELLIGENCE PIPELINE REPORT: {candidate_name.upper()}")
print("="*75)
print("» COMPLETE FIT PERCENTAGE DISTRIBUTION BY ROLE DOMAIN (COSINE MATRIX):")
print("-"*75)

for _, row in df_role_metrics.iterrows():
    print(f"  • {row['Role Domain'].ljust(35)} : {row['Match Confidence']:.2f}%")
    
print("-"*75)
print(f"» NAIVE BAYES PREDICTION               : {model_nb.predict(X_target_resume)[0].upper()}")
print(f"» LOGISTIC REGRESSION PREDICTION       : {model_lr.predict(X_target_resume)[0].upper()}")
print(f"» VECTOR SIMILARITY PREDICTED DOMAIN   : {similarity_predicted_role.upper()}")
print(f"» COSINE ENGINE CONFIDENCE MATCH SCORE  : {similarity_confidence_score:.2f}%")
print("="*75 + "\n")

print("\n" + "="*75)
print(" SYSTEM MODEL EVALUATION COMPARISON")
print("="*75)

# Benchmark 1: Naive Bayes Test Performance Evaluation
y_pred_nb = model_nb.predict(X_test)
print(f"» Overall Naive Bayes Test Accuracy: {accuracy_score(y_test, y_pred_nb):.2%}")

# Benchmark 2: Logistic Regression Test Performance Evaluation
y_pred_lr = model_lr.predict(X_test)
print(f"» Overall Logistic Regression Test Accuracy: {accuracy_score(y_test, y_pred_lr):.2%}\n")

print("="*75)
print("DETAILED LOGISTIC REGRESSION BREAKDOWN REPORT:")
print("="*75)
print(classification_report(y_test, y_pred_lr, zero_division=0))
