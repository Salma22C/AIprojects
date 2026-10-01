
# 🧠 AI Engineering Portfolio

A collection of AI, Machine Learning, NLP, and RAG projects focused on building,
evaluating, and improving practical intelligent systems.

My work combines:

- Machine Learning & Data Mining
- NLP & Transformer Models
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- AI Agents & LLM Evaluation
- Semantic Search & Embeddings
- AI Engineering & Deployment

A major focus of my recent work is moving beyond simply building AI applications
toward understanding **how to evaluate, diagnose, and improve them**.

---

# 🚀 Featured Projects

## 📱 Personal Behavior & Habit Pattern Mining — V1

A Data Mining and Machine Learning project using smartphone behavioral data
to discover distinct usage patterns through K-Means clustering.

### What I built

- End-to-end Data Mining pipeline
- Feature preprocessing using `StandardScaler`
- K-Means clustering
- Evaluation across `K=2–10`
- Inertia and silhouette-score analysis
- Cluster profiling and interpretation
- Model persistence for inference
- New-user cluster prediction
- Interactive Streamlit application

### V1 Dataset

- 700 users
- 5 behavioral features

### Features

- App Usage Time
- Screen On Time
- Battery Drain
- Number of Apps Installed
- Data Usage

### V1 Modeling

K-Means was evaluated across multiple values of `K`.

For V1, `K=5` was selected to create five behavioral usage profiles for
further exploration.

The selection is treated as a modeling decision rather than a claim that
`K=5` is universally optimal.

### Engineering

Saved model artifacts:

```text
models/
├── scaler.pkl
└── kmeans.pkl
```

The same fitted scaler and K-Means model are reused during inference so that
new users are assigned to existing clusters without retraining.

### Application

The Streamlit application allows users to:

- Explore cluster sizes
- Explore cluster profiles
- Enter behavioral information
- Predict the user's existing cluster
- View the corresponding behavioral profile

**Status:** V1 complete — further analysis planned.

👉 [Explore the Project](https://github.com/Salma22C/Personal-Behavior-Habit-Pattern-Mining)

---

# 🔬 RAG Research & Engineering

My recent work focuses on understanding RAG systems not only as application
architectures, but also as systems that need to be **evaluated, diagnosed,
and improved**.

The workflow I follow is:

```text
Research
   ↓
Understanding
   ↓
Implementation
   ↓
Experiments
   ↓
Observations
   ↓
Findings
   ↓
Reusable Engineering Tools
```

---

## 📚 Research Studies

### COCOM — Context Embeddings for Efficient Answer Generation in RAG

A research paper I studied to understand context compression for
Retrieval-Augmented Generation.

The study focused on how retrieved contexts can be represented more
efficiently while preserving useful information for downstream answer
generation.

My work with COCOM focuses on:

- Understanding the proposed architecture
- Studying context embeddings
- Analyzing compression trade-offs
- Understanding offline and online components
- Implementing related concepts experimentally
- Investigating how context compression affects RAG behavior

**Important:** COCOM is published research by its original authors.
It is presented here as a research study and reference for my own
experimentation.

---

### RAGChecker — RAG Evaluation & Diagnostic Framework

A research framework I studied to understand systematic evaluation of
Retrieval-Augmented Generation systems.

My study focused on:

- Claim-level evaluation
- Retrieval diagnostics
- Generation diagnostics
- Understanding retrieval vs. generation failures
- Analyzing why traditional evaluation metrics can hide specific failure
  patterns

The framework influenced the design of my own RAG diagnosis experiments.

**Important:** RAGChecker is the work of its original authors.
My contribution is the implementation, experimentation, and engineering
work built around the concepts I studied.

---

# 🛠️ Independent RAG Engineering

## rag-diagnose — RAG Diagnosis & Evaluation

An independent engineering project developed from my study of RAG evaluation
and diagnostic methods.

The goal is to move beyond:

> "Did the RAG system produce a good answer?"

and instead investigate:

> "Why did the RAG system produce this answer?"

### Current direction

The project explores diagnostic analysis of RAG pipelines, including:

- Retrieval quality
- Retrieved-context relevance
- Context sufficiency
- Answer correctness
- Retrieval failures
- Generation failures
- Claim-level analysis
- Failure categorization

### Engineering Goal

I am developing `rag-diagnose` toward a reusable RAG evaluation and
diagnostic toolkit that can be integrated into future RAG projects.

Instead of rebuilding evaluation logic for every RAG application, the goal is
to create reusable components for:

```text
RAG System
    ↓
Evaluation
    ↓
Diagnosis
    ↓
Failure Analysis
    ↓
Experiment
    ↓
Improvement
```

**Status:** Active development.

---

# 🧪 RAG Experiments & Findings

My RAG work also includes independent experiments designed to understand
failure modes rather than only measure final answer quality.

Examples of questions investigated include:

### Retrieval Relevance ≠ Retrieval Sufficiency

A retriever can return a document that is semantically related to a query
while still failing to provide the information required to answer it.

This distinction became an important part of my understanding of RAG
diagnostics.

### Retrieval vs. Generation

A wrong answer does not necessarily mean the LLM generated incorrectly.

The failure may originate from:

- Poor retrieval
- Missing evidence
- Insufficient context
- Incorrect context usage
- Generation reasoning

The experiments therefore treat the RAG pipeline as multiple components
rather than a single black box.

### Build → Measure → Evaluate → Improve

My RAG engineering workflow follows:

```text
Build
  ↓
Measure
  ↓
Evaluate
  ↓
Identify Failure
  ↓
Experiment
  ↓
Improve
```

The goal is to support future RAG projects with measurable evaluation rather
than relying only on qualitative inspection.

---

# 🤖 LLM & AI Applications

## 🧠 TalentCheck AI

Evaluator–Optimizer resume screening system using a self-correcting
multi-agent architecture.

### Features

- Resume parsing
- LLM-based candidate evaluation
- Multi-agent validation
- Hallucination detection
- Iterative correction
- Structured JSON outputs

### Architecture

```text
Resume + Job Description
          ↓
   Optimizer Agent
          ↓
   Candidate Evaluation
          ↓
    Evaluator Agent
          ↓
     PASS / FAIL
          ↓
      Correction
          ↓
      Re-evaluation
```

The project explores how separating generation from evaluation can improve
the reliability of LLM-based decision-support systems.

### Tech Stack

`Python` • `OpenRouter` • `LLM Agents` • `pypdf`

---

## 🎯 AI Career Advisor — RAG System

A Retrieval-Augmented Generation application that transforms educational
course catalogs into an interactive career advising system.

### Features

- PDF knowledge base
- Semantic retrieval
- FAISS vector search
- Embeddings
- Learning-path generation
- Grounded course recommendations
- Conversational interface

### Tech Stack

`Python` • `FAISS` • `SentenceTransformers` • `OpenRouter` • `Gradio`

---

## 🏢 Intelligent Support Router & Triage

An end-to-end NLP system combining Transformer-based classification with
LLM-powered response generation.

### Features

- Customer support ticket classification
- Fine-tuned BERT encoder
- Three ticket categories
- AI-generated draft responses
- Flask inference application
- Model evaluation
- Manual stress testing

### Architecture

```text
Customer Ticket
      ↓
AutoTokenizer
      ↓
Fine-Tuned BERT
      ↓
Ticket Category
      ↓
Generative LLM
      ↓
Draft Response
```

### Tech Stack

`Python` • `TensorFlow` • `Hugging Face Transformers` • `BERT`
• `Flask` • `Qwen`

👉 [Explore the Repository](https://github.com/Salma22C/Intelligent-Support-Router)

---

# 📊 Data Mining & Classical Machine Learning

## 📈 Customer Churn Prediction

A Data Mining classification project investigating customer churn using
multiple classification algorithms.

### Models Compared

- Logistic Regression
- Decision Tree
- Random Forest

### Methodology

```text
Data Understanding
       ↓
Data Cleaning
       ↓
EDA
       ↓
Preprocessing
       ↓
Train/Test Split
       ↓
Classification
       ↓
Model Evaluation
       ↓
Model Comparison
       ↓
Model Selection
       ↓
New Customer Prediction
```

### Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The Decision Tree achieved **99.99% accuracy on the reported test split**.
This result is interpreted specifically in the context of this dataset and
split rather than as evidence of general real-world performance.

### Engineering

The project includes:

- Reusable preprocessing pipeline
- Saved model artifacts
- New-customer prediction
- Streamlit interface
- Model interpretation

👉 [Explore the Repository](https://github.com/Salma22C/Customer-Churn-Predication)

---

## 🧠 AI Talent Intelligence & Sentiment Platform — V2

An NLP-based talent analytics system that transforms unstructured applicant
text into structured talent information.

### Architecture

The system combines:

- Supervised role classification
- TF-IDF representations
- Cosine similarity
- Multi-domain skill matching
- Sentiment analysis

### Features

- PDF resume processing
- Text preprocessing
- Lemmatization
- Custom stop-word handling
- Job-role classification
- Multi-domain talent matching
- Sentiment analysis
- Talent analytics

### Tech Stack

`Python` • `Scikit-learn` • `spaCy` • `Pandas` • `Matplotlib`
• `PyPDF` • `VADER`

👉 [Explore the Repository](https://github.com/Salma22C/AIprojects)

---

# 💬 Additional AI Projects

## 🤖 AI-Powered Chat Assistant

Conversational AI assistant built using LLM APIs and prompt engineering.

### Features

- Interactive chatbot
- Context-aware responses
- API-based LLM integration
- Custom prompting workflows

### Tech Stack

`Python` • `Transformers` • `OpenRouter` • `Gradio`

---

## 📚 Atomic Habits RAG Chatbot

A Retrieval-Augmented Generation experiment for contextual question
answering over a book-based knowledge source.

### Features

- Document chunking
- Embeddings
- Semantic retrieval
- Context-aware generation
- Book-based Q&A

---

## ✍️ Text Summarization Tool

An NLP application for generating concise summaries from long-form text.

### Features

- Text preprocessing
- Text cleaning
- Extractive/abstractive summarization
- Interactive interface

---

## 🔄 AI + HITL LinkedIn Posting Workflow

A Human-in-the-Loop AI workflow for generating, reviewing, and improving
LinkedIn content before publishing.

### Features

- AI-generated drafts
- Human review loop
- Prompt engineering
- Content refinement
- Workflow automation

---

# 🧰 Core Technologies

### Programming

`Python`

### Machine Learning

`Scikit-learn` • `TensorFlow` • `K-Means` • Classification
• Clustering • Model Evaluation

### NLP

`spaCy` • `Hugging Face Transformers` • `BERT` • `TF-IDF`
• Tokenization • Lemmatization

### LLM & Generative AI

`LLMs` • `RAG` • `Prompt Engineering` • `Structured Outputs`
• `OpenRouter` • `OpenAI-compatible APIs`

### Retrieval & Search

`FAISS` • `SentenceTransformers` • `Embeddings` • `Semantic Search`

### AI Systems

`LLM Agents` • `Multi-Agent Workflows` • `Evaluator-Optimizer Architectures`
• `AI Evaluation` • `RAG Diagnostics`

### Backend & Deployment

`FastAPI` • `Flask` • `Streamlit` • `Gradio` • `Docker`

### Data

`Pandas` • `NumPy` • Data Preprocessing • Data Analysis

### Tools

`Git` • `GitHub` • `Jupyter Notebook` • `VS Code`

---

# 🎯 Current Engineering Focus

My current focus is evolving from building individual AI applications
toward building **reliable and measurable AI systems**.

Areas I am currently exploring:

- RAG evaluation
- RAG diagnostics
- Retrieval quality
- LLM reliability
- Context quality
- AI evaluation pipelines
- Multi-agent systems
- Semantic retrieval
- Embedding-based systems
- Research-to-engineering workflows

---

# 🔬 Research → Engineering

My recent learning approach follows a practical research-to-engineering
workflow:

```text
Published Research
       ↓
Understand the Method
       ↓
Implement Relevant Concepts
       ↓
Design Experiments
       ↓
Measure Results
       ↓
Analyze Failures
       ↓
Extract Findings
       ↓
Build Reusable Tools
```

The goal is not simply to reproduce papers.

It is to use research as a foundation for understanding AI systems and then
translate useful concepts into independently implemented experiments and
engineering components.

---

# 📌 Current Projects

| Project | Area | Status |
|---|---|---|
| `rag-diagnose` | RAG Evaluation & Diagnostics | 🚧 Active Development |
| Personal Behavior & Habit Pattern Mining | Data Mining / ML | ✅ V1 Complete |
| TalentCheck AI | LLM Agents / Evaluation | ✅ Built |
| AI Career Advisor | RAG / Semantic Search | ✅ Built |
| Intelligent Support Router | NLP / BERT / LLM | ✅ Built |
| Customer Churn Prediction | Data Mining / Classification | ✅ Built |

---

# 📬 Contact

**Salma Kassem**

AI Engineer | Machine Learning | NLP | RAG | LLM Systems

- [LinkedIn](https://linkedin.com/in/salma-mohamed-kassem)
- [GitHub](https://github.com/Salma22C/)
- [Portfolio](https://salmakassem.framer.website/)
- Email: salmakassem6@gmail.com
```

