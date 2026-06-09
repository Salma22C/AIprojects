Here is the clean, production-ready Python script extracted from your Jupyter Notebook. The cell blocks have been combined into a logical workflow structured for a repository or deployment environment.

```python
"""
AI Talent Intelligence Platform - Level 2
Author: Salma Kassem (Pipeline Telemetry)
Description: Extracts text from a candidate's resume, preprocesses the natural 
             language data using spaCy, computes TF-IDF weights, evaluates role 
             fit using a Multinomial Naive Bayes classifier, and runs a vector 
             similarity engine using Cosine Similarity.
"""

# =====================================================================
# 1. IMPORTS & DEPENDENCIES
# =====================================================================
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_similarity  
import spacy


# =====================================================================
# 2. PDF TEXT EXTRACTION ENGINE
# =====================================================================
def extract_pdf_text(path):
    """Extract raw string text from a target PDF file."""
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


# =====================================================================
# 3. TEXT PREPROCESSING & NLP PIPELINE
# =====================================================================
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


def main():
    # Define file paths
    resume_path = "data/Salma_Kasssem.pdf"
    dataset_path = "data/resume_dataset_v2.xlsx"
    output_dir = "AI Talent Intelligence Platform/screenshots"
    
    # Check dependencies and data paths
    if not os.path.exists(resume_path):
        print(f"[ERROR] Target resume missing at {resume_path}. Please place your PDF in the correct path.")
        return
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset file missing at {dataset_path}.")
        return

    print("[INFO] Initializing PDF extraction and Spacy English tokenization...")
    resume_text = extract_pdf_text(resume_path)
    clean_resume = clean_text_initial(resume_text)

    # Initialize SpaCy pipeline
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("[INFO] Spacy model 'en_core_web_sm' missing. Downloading model...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    resume_doc = nlp(clean_resume)

    # Filter out stop words, numeric terms, and punctuation
    processed_tokens = [
        token.lemma_.lower().strip()
        for token in resume_doc
        if not token.is_stop 
        and not token.like_num 
        and token.is_alpha 
        and token.text.strip()
    ]
    processed_text = " ".join(processed_tokens)


    # =====================================================================
    # 4. FEATURE VECTORIZATION & TF-IDF EDA
    # =====================================================================
    documents = [processed_text]

    cv = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1, max_df=1.0)
    X_cv = cv.fit_transform(documents)

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=1.0)
    X_tfidf = tfidf_vectorizer.fit_transform(documents)
    df_tfidf = pd.DataFrame(X_tfidf.toarray(), index=["Resume"], columns=tfidf_vectorizer.get_feature_names_out())

    # Build internal horizontal bar plot for top TF-IDF weights
    top_tfidf_words = df_tfidf.sum().sort_values(ascending=True).tail(15)
    plt.figure(figsize=(10, 6))
    top_tfidf_words.plot(kind="barh", color="lightgreen")
    plt.title("Top 15 Tokens by TF-IDF Weight", fontsize=14, fontweight="bold")
    plt.xlabel("TF-IDF Score", fontsize=12)
    plt.ylabel("Tokens / N-grams", fontsize=12)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/top_tfidf_features.png", dpi=300)
    plt.close()


    # =====================================================================
    # 5. MODEL TRAINING & PIPELINE ARCHITECTURE
    # =====================================================================
    df_dataset = pd.read_excel(dataset_path)

    # Clean empty matrix slots
    df_dataset = df_dataset.dropna(subset=["resume_text", "role"])
    df_dataset = df_dataset[df_dataset["resume_text"].str.strip() != ""]
    df_dataset = df_dataset[df_dataset["role"].str.strip() != ""]

    X_text = df_dataset["resume_text"]
    y = df_dataset["role"]

    # Limit to top 1000 features to reduce dimension noise
    classifier_tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    X = classifier_tfidf.fit_transform(X_text)

    # Split dataset stratifying category targets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Train Multinomial Naive Bayes Model
    model = MultinomialNB(alpha=0.5, fit_prior=False)
    model.fit(X_train, y_train)


    # =====================================================================
    # 6. CANDIDATE RESUME CLASSIFICATION & SIMILARITY EVALUATION
    # =====================================================================
    candidate_name = os.path.splitext(os.path.basename(resume_path))[0]
    X_target_resume = classifier_tfidf.transform([processed_text])
    resume_prediction = model.predict(X_target_resume)[0]

    # Calculate exact Cosine Similarity across document vector distributions
    raw_similarity_scores = cosine_similarity(X_target_resume, X).flatten()

    analysis_df = pd.DataFrame({
        "role": df_dataset["role"],
        "similarity": raw_similarity_scores
    })

    # Transform scores using Softmax distribution approach
    role_averages = analysis_df.groupby("role")["similarity"].mean()
    exp_averages = np.exp(role_averages - np.max(role_averages))
    role_percentages = (exp_averages / np.sum(exp_averages)) * 100

    df_role_metrics = pd.DataFrame({
        "Role Domain": role_percentages.index,
        "Match Confidence": role_percentages.values
    }).sort_values(by="Match Confidence", ascending=False)

    similarity_predicted_role = df_role_metrics.iloc[0]["Role Domain"]
    similarity_confidence_score = df_role_metrics.iloc[0]["Match Confidence"]


    # =====================================================================
    # 7. TELEMETRY REPORT GENERATION
    # =====================================================================
    print("\n" + "="*75)
    print(f" ADVANCED TALENT INTELLIGENCE PIPELINE REPORT: {candidate_name.upper()}")
    print("="*75)
    print("» COMPLETE FIT PERCENTAGE DISTRIBUTION BY ROLE DOMAIN:")
    print("-"*75)

    for _, row in df_role_metrics.iterrows():
        print(f"  • {row['Role Domain'].ljust(35)} : {row['Match Confidence']:.2f}%")
        
    print("-"*75)
    print(f"» NAIVE BAYES CLASSIFIER PREDICTION    : {resume_prediction.upper()}")
    print(f"» VECTOR SIMILARITY PREDICTED DOMAIN   : {similarity_predicted_role.upper()}")
    print(f"» COSINE ENGINE CONFIDENCE MATCH SCORE  : {similarity_confidence_score:.2f}%")
    print("="*75 + "\n")


    # =====================================================================
    # 8. STANDALONE MODEL EVALUATION METRICS
    # =====================================================================
    print("\n" + "="*75)
    print(" SYSTEM MODEL EVALUATION METRICS")
    print("="*75)

    y_pred = model.predict(X_test)

    # Accuracy Metric Validation
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Model Verification Accuracy: {test_accuracy:.2%}\n")

    # Classification Breakdown Report
    print("Detailed Class Breakdown Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("-"*75)

    # Compute Confusion Matrix Layout & Plot Save
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title("Talent Classifier: Confusion Matrix Heatmap", fontsize=14, fontweight=\"bold\")\n", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plt.savefig(f"{output_dir}/model_confusion_matrix.png", dpi=300)
    print(f"[SYSTEM INFO] Matrix validation and TF-IDF visual plots successfully saved to: {output_dir}")
    plt.close()


if __name__ == "__main__":
    main()

```
