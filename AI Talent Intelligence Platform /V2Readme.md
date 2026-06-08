# 🧠 AI Talent Intelligence & Sentiment Platform (Version 2)

> 🔍 An end-to-end Talent Intelligence system that parses unstructured candidate data, maps core semantic frequencies, profiles text sentiment intensity, and uses supervised machine learning to classify profiles into explicit job roles.

🚀 Built to simulate an enterprise-level **AI hiring analytics engine** capable of automating high-volume resume parsing, talent discovery benchmarking, and role categorization.

---

## ⚡ Why This Project Exists

Manual applicant screening creates severe operational bottlenecks in modern recruitment pipelines. This platform optimizes and automates talent operations by shifting from shallow keyword matching to structured semantic understanding and predictive modeling.

### The Core Solution:
* **Automated Parsing:** Translates raw, noisy unstructured resume PDFs into highly normalized, clean text.
* **Sentiment Profiling:** Applies contextual intensity rules to extract language patterns, distinguishing tone and structural confidence levels.
* **Predictive Role Mapping:** Automatically classifies applicant data into concrete job titles with deterministic ML pipelines.

---

## 🧬 System Evolution

### 🟢 Version 1 — The Semantic Foundation
* Robust PDF text parsing using `pypdf`.
* Regex token normalization, advanced stop-word filtration, and language lemmatization via `spaCy`.
* N-gram feature tracking (`ngram_range=(1, 2)`) leveraging Scikit-Learn's `CountVectorizer`.
* Exploratory Data Analysis (EDA) plotting frequency metrics directly with `matplotlib`.

### 🔵 Version 2 — Predictive Intelligence Layer (Current)
* **Standalone VADER Engine Integration:** Evaluates contextual polarity scoring directly on text structures, tracking multi-tiered sentiment features without heavy framework dependencies.
* **Vector Space Feature Engineering:** Implements advanced token matrix modeling via `TfidfVectorizer` to balance term relevance across unbalanced text corpuses.
* **Supervised Classifiers:** Integrates a trained **Multinomial Naive Bayes (`MultinomialNB`)** model to predict candidate roles across complex domains with **90% model test split accuracy**.
* **Fully Dynamic Input Pipeline:** Fully decoupled runtime architecture that dynamically profiles incoming candidate payloads, automatically resolving metadata profiles (e.g., extracting candidate identification parameters directly from system file paths).

---

## 🏗️ Architecture Flow

```text
               ┌─────────────────────────────────┐
               │    Raw Candidate Resume PDF    │
               └────────────────┬────────────────┘
                                │
                                ▼
               ┌─────────────────────────────────┐
               │   RegEx Cleaning & Text Scrub   │
               │  (HTML, URLs, Emails Stripped)  │
               └────────────────┬────────────────┘
                                │
                                ▼
               ┌─────────────────────────────────┐
               │      NLP Token Processing       │
               │ (Lemmatization & Stop-words)    │
               └────────┬────────────────────────┴────────┐
                        │                                 │
                        ▼                                 ▼
         ┌─────────────────────────────┐   ┌─────────────────────────────┐
         │     vaderSentiment Engine   │   │  TfidfVectorizer Pipeline   │
         │  (Contextual Polarity Docs) │   │ (Bi-gram Vector Generation) │
         └──────────────┬──────────────┘   └──────────────┬──────────────┘
                        │                                 │
                        ▼                                 ▼
         ┌─────────────────────────────┐   ┌─────────────────────────────┐
         │ Top/Bottom Sentiment Arrays │   │    MultinomialNB Engine     │
         │   (Confidence Extraction)   │   │ (Deterministic Role Class)  │
         └─────────────────────────────┘   └──────────────┬──────────────┘
                                                          │
                                                          ▼
                                           ┌─────────────────────────────┐
                                           │ Dynamic Terminal Dashboard  │
                                           │ (Visual Artifact + Class)   │
                                           └─────────────────────────────┘
## 🛠️ Tech Stack & Dependencies
Core Runtime: Python 3.10+

Data Layout Engine: Pandas, NumPy

Natural Language Processing: spaCy (en_core_web_sm), vaderSentiment

Machine Learning Frame: Scikit-Learn

System Files Parsing: pypdf, BeautifulSoup4, openpyxl

Data Visualization: Matplotlib

## Model Prediction Output
Model Verification Accuracy: 0.90

==========================================
FINAL CLASSIFICATION FOR: SARAH_JOHNSON
Predicted Designation: Data Analyst
==========================================

## 👩‍💻 Author
Salma Mohamed — AI Engineer & Core Automation Developer 🚀

Building scalable, production-grade NLP and Machine Learning workflows from the ground up.
