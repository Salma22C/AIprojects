# 🧠 AI Talent Intelligence Platform (Version 2)

An NLP and Machine Learning project that automatically processes resumes, extracts meaningful text features, and predicts the most suitable job role using supervised learning.

The project demonstrates a complete NLP pipeline from raw PDF resumes to machine learning-based role classification.

---

# 🚀 Project Overview

The goal of this project is to automate resume understanding and job role identification.

Instead of manually reviewing resumes, the system:

1. Extracts text from PDF resumes
2. Cleans and preprocesses text
3. Converts text into numerical features
4. Trains machine learning models on labeled resume data
5. Predicts the most likely job role for unseen resumes

---

# 🔄 Project Evolution

## Version 1 – Resume Matching System

Version 1 focused on semantic similarity.

### Features

* PDF Resume Parsing
* Text Cleaning
* spaCy Preprocessing
* CountVectorizer
* TF-IDF Vectorization
* Cosine Similarity Matching

The system compared a resume against a job description and generated a similarity score.

### Example

Resume ↔ Job Description

Match Score: 30.36%

---

## Version 2 – Talent Intelligence Platform

Version 2 upgrades the system from similarity matching to machine learning classification.

Instead of comparing resumes with job descriptions, the system learns patterns from labeled resume datasets and predicts job roles directly.

### Features

* Resume PDF Parsing
* Text Cleaning using Regex and BeautifulSoup
* Tokenization and Lemmatization using spaCy
* TF-IDF Feature Engineering
* Train/Test Dataset Splitting
* Naive Bayes Classification
* Logistic Regression Evaluation
* Unseen Resume Prediction

### Supported Roles

* AI Engineer
* Data Scientist
* Data Analyst
* Backend Developer
* Cloud Engineer
* Product Manager

---

# 🏗️ System Architecture

Raw Resume PDF
↓
PDF Text Extraction (PyPDF)
↓
Text Cleaning (Regex + BeautifulSoup)
↓
spaCy NLP Processing
↓
Tokenization + Stopword Removal + Lemmatization
↓
TF-IDF Vectorization
↓
Machine Learning Model
↓
Job Role Prediction

---

# 🛠️ Technologies Used

## Programming

* Python

## NLP

* spaCy
* Regex
* BeautifulSoup

## Machine Learning

* Scikit-Learn
* TF-IDF Vectorizer
* Multinomial Naive Bayes
* Logistic Regression

## Data Handling

* Pandas

## Visualization

* Matplotlib

## File Processing

* PyPDF

---

# 📊 Model Evaluation

Two machine learning models were evaluated:

### Multinomial Naive Bayes

* Fast baseline classifier
* Effective for text classification problems
* Works well on smaller datasets

### Logistic Regression

* Linear classification model
* Better generalization on overlapping classes
* Used to compare performance against Naive Bayes

Example evaluation:

* Naive Bayes Accuracy: 71.43%
* Logistic Regression Accuracy: 71.43%

---

# 🔍 Sample Prediction Workflow

Input:
Resume PDF

Output:

Predicted Role:
AI Engineer

The model predicts the most likely role based on patterns learned from previously labeled resumes.

---

# 📚 Key Concepts Learned

Through this project I applied:

* Text Cleaning
* Tokenization
* Lemmatization
* Feature Engineering
* TF-IDF
* Text Classification
* Naive Bayes
* Logistic Regression
* Model Evaluation
* Precision
* Recall
* F1 Score
* Accuracy Analysis

---
## Future Experiments

During development, I explored Topic Modeling using Non-Negative Matrix Factorization (NMF) to discover hidden skill clusters across resumes. While the generated topics were interpretable and aligned with role domains such as AI, Cloud, Backend Development, and Data Analytics, the experiment highlighted an important machine learning insight: increasing feature complexity does not always improve classification performance, particularly on smaller datasets.

Future versions may explore semantic topic modeling and embedding-based approaches for improved role understanding.

# 👩‍💻 Author

Salma Mohamed

AI Engineer | Cloud Computing Teaching Assistant

Interested in NLP, Machine Learning, LLM Applications, and AI Systems Engineering.
