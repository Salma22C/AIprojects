# 🧠 AI Talent Intelligence & Sentiment Platform (Version 2)

> 🔍 An end-to-end Talent Intelligence system that parses unstructured candidate data, maps core semantic frequencies, profiles text sentiment intensity, and uses a hybrid machine learning framework to classify profiles and measure multi-role domain similarity.

🚀 Built to simulate an enterprise-level **AI hiring analytics engine** capable of automating high-volume resume parsing, talent discovery benchmarking, and cross-functional role categorization.

---

## ⚡ Why This Project Exists

Manual applicant screening creates severe operational bottlenecks in modern recruitment pipelines. This platform optimizes and automates talent operations by shifting from shallow keyword matching to structured semantic understanding, predictive modeling, and vector-space alignment.

### The Core Solution:
* **Automated Parsing:** Translates raw, noisy unstructured resume PDFs into highly normalized, clean text layers.
* **Sentiment Profiling:** Applies contextual intensity rules via VADER to extract language patterns, distinguishing tone and structural confidence levels.
* **Predictive Role Mapping:** Automatically classifies applicant data into concrete job titles using supervised ML.
* **Hybrid Skill Analytics:** Evaluates directional vector alignment to output an executive percentage distribution match across all repository domains.

---

## 🧬 System Evolution

### 🟢 Version 1 — The Semantic Foundation
* Robust PDF text parsing using `pypdf`.
* Regex token normalization, advanced stop-word filtration, and language lemmatization via `spaCy`.
* N-gram feature tracking (`ngram_range=(1, 2)`) leveraging Scikit-Learn's `CountVectorizer`.
* Exploratory Data Analysis (EDA) plotting frequency metrics directly with `matplotlib`.

### 🔵 Version 2 — Predictive & Vector Intelligence Layer (Current)
* **Standalone VADER Engine Integration:** Evaluates contextual polarity scoring directly on text structures, tracking multi-tiered sentiment features without heavy framework dependencies.
* **Vector Space Feature Engineering:** Implements advanced token matrix modeling via `TfidfVectorizer` to balance term relevance across unbalanced text corpuses.
* **Supervised Classifiers:** Integrates a trained **Multinomial Naive Bayes (`MultinomialNB`)** model to predict candidate roles across complex domains with **90% model test split accuracy**.
* **Aggregated Cosine Similarity Matching:** Features a mathematical validation engine that aggregates vector angles to output a multi-role affinity percentage, spotlighting candidates with hybrid or cross-functional technical backgrounds.
* **Fully Dynamic Input Pipeline:** Fully decoupled runtime architecture that dynamically profiles incoming candidate payloads, automatically resolving metadata profiles directly from system file paths.

---

## 🏗️ Architecture Flow

```text
               ┌─────────────────────────────────┐
               │    Raw Candidate Resume PDF     │
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
               │       NLP Token Processing      │
               │  (Lemmatization & Stop-words)   │
               └────────┬────────────────────────┴────────┐
                        │                                 │
                        ▼                                 ▼
         ┌─────────────────────────────┐   ┌─────────────────────────────┐
         │    vaderSentiment Engine    │   │  TfidfVectorizer Pipeline   │
         │  (Contextual Polarity Docs) │   │ (Bi-gram Vector Generation) │
         └──────────────┬──────────────┘   └──────────────┬──────────────┘
                        │                                 │
                        ▼                                 ▼
         ┌─────────────────────────────┐   ┌─────────────────────────────┐
         │ Top/Bottom Sentiment Arrays │   │    Multi-Engine Analyser    │
         │   (Confidence Extraction)   │   │  (Naive Bayes Matrix Head)  │
         └─────────────────────────────┘   └──────────────┬──────────────┘
                                                          │
                                                          ▼
                                           ┌─────────────────────────────┐
                                           │  Cosine Similarity Engine   │
                                           │  (Weighted Role Aggregator) │
                                           └──────────────┬──────────────┘
                                                          │
                                                          ▼
                                           ┌─────────────────────────────┐
                                           │ Dynamic Terminal Dashboard  │
                                           │ (Visual Metrics + Report)   │
                                           └─────────────────────────────┘
## 🛠️ Tech Stack & Dependencies
Core Runtime: Python 3.10+

Data Layout Engine: Pandas, NumPy, OpenPyXL

Natural Language Processing: spaCy (en_core_web_sm), vaderSentiment

Machine Learning & Core Metrics: Scikit-Learn

System Files Parsing: PyPDF, BeautifulSoup4

Data Visualization: Matplotlib

## 📊 System Output Telemetry Report
===========================================================================
 ADVANCED TALENT INTELLIGENCE PIPELINE REPORT: SALMA_KASSSEM
===========================================================================
» COMPLETE FIT PERCENTAGE DISTRIBUTION BY ROLE DOMAIN:
---------------------------------------------------------------------------
  • AI Engineer                       : 56.42%
  • Data Scientist                    : 24.15%
  • DevOps Engineer                   : 11.30%
  • Full Stack Developer              : 8.13%
---------------------------------------------------------------------------
» NAIVE BAYES CLASSIFIER PREDICTION    : AI ENGINEER
» VECTOR SIMILARITY PREDICTED DOMAIN   : AI ENGINEER
» COSINE ENGINE CONFIDENCE MATCH SCORE  : 56.42%
===========================================================================

---

## 👩‍💻 Author

**Salma Mohamed** **AI Engineer & Cloud Architect** | NLP & Machine Learning Builder  
Specializing in end-to-end RAG pipelines, multi-agent frameworks, and intelligent automation systems. 🚀
