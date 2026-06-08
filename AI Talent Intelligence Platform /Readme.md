# AI Talent Intelligence Platform — Version 1

## Overview

AI Talent Intelligence Platform Version 1 is a Natural Language Processing (NLP) project designed to measure the similarity between a candidate's resume and a job description.

The system extracts text from PDF resumes, applies a complete NLP preprocessing pipeline, converts text into numerical representations using TF-IDF vectorization, and calculates a matching score using Cosine Similarity.

This version focuses on traditional NLP techniques and serves as the foundation for future machine learning and transformer-based enhancements.

---

## Features

- PDF Resume Text Extraction
- Text Cleaning and Normalization
- Tokenization using spaCy
- Stopword Removal
- Lemmatization
- TF-IDF Vectorization
- Resume–Job Description Similarity Scoring
- Match Percentage Calculation

---

## Project Pipeline

```text
Resume PDF
      ↓
PDF Text Extraction (PyPDF)
      ↓
Text Cleaning
      ↓
Tokenization
      ↓
Stopword Removal
      ↓
Lemmatization
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Resume Match Score
```

---

## Technologies Used

- Python
- Pandas
- Regular Expressions (re)
- BeautifulSoup4
- spaCy
- PyPDF
- Scikit-learn

---

## NLP Techniques Implemented

### 1. Text Cleaning

The extracted text is cleaned by:

- Removing HTML tags
- Removing URLs
- Removing email addresses
- Removing special characters
- Removing extra whitespace
- Converting text to lowercase

---

### 2. Tokenization

Text is split into individual tokens using spaCy.

Example:

```python
["ai", "engineer", "python", "aws"]
```

---

### 3. Stopword Removal

Common words such as:

```text
the
is
and
in
of
```

are removed to reduce noise.

---

### 4. Lemmatization

Words are converted to their base form.

Examples:

```text
running → run
systems → system
applications → application
```

---

### 5. TF-IDF Vectorization

TF-IDF (Term Frequency–Inverse Document Frequency) converts text into numerical vectors while assigning greater importance to informative terms and reducing the weight of common terms.

---

### 6. Cosine Similarity

Cosine Similarity measures the similarity between the resume and job description vectors.

Output:

```text
0.00 = No similarity
1.00 = Perfect similarity
```

The score is converted into a percentage-based match score.

Example:

```text
Match Score: 82.45%
```

---

## Example Output

```text
Resume Match Score: 78.36%
```

The score indicates how closely the candidate's resume aligns with the job requirements.

---

## Project Structure

```text
AI Talent Intelligence Platform
│
├── Resume PDF
├── Job Description TXT
│
├── Text Extraction
├── Text Cleaning
├── NLP Preprocessing
│   ├── Tokenization
│   ├── Stopword Removal
│   └── Lemmatization
│
├── TF-IDF Vectorization
├── Cosine Similarity
│
└── Match Score Output
```

---

## Future Enhancements

### Version 2

- Text Classification
- Job Role Prediction using Machine Learning

### Version 3

- Topic Modeling
- Resume Expertise Detection

### Version 4

- Transformer Embeddings
- Semantic Resume Matching

### Version 5

- Hugging Face Models
- Skill Extraction
- Advanced Candidate Analysis

---
pip install -r requirements.txt
python -m spacy download en_core_web_sm

## Learning Outcomes

Through this project, I gained hands-on experience with:

- NLP preprocessing pipelines
- Text vectorization techniques
- TF-IDF and Bag-of-Words concepts
- Similarity measurement using Cosine Similarity
- PDF text extraction
- Traditional NLP workflows
- Resume-job matching systems

---
