# Intelligent Document Summarization & Insight Engine

An advanced NLP-powered web application designed to ingest multi-format documents, perform semantic text chunking, and generate highly structured summaries along with context-weighted key highlights. 

Built using state-of-the-art Transformer architectures combined with traditional statistical feature extraction methodologies, this tool ensures accurate information synthesis from noisy unstructured data.

---

## 🚀 Key Features

* **Multi-Format Document Parsing:** Native support for parsing unstructured text from raw user input, `.txt`, `.pdf`, `.docx`, and structured `.csv` files.
* **Abstractive Text Summarization:** Utilizes deep learning sequence-to-sequence models to generate human-like text summaries.
* **Hybrid Key Highlight Generation:** Features a proprietary ranking algorithm that intelligently scores and filters key sentences by blending statistical text importance with keyword extraction.
* **Robust Text Normalization & Chunking:** Implements deterministic regex-driven text cleaning and token-based chunking pipelines to circumvent model context window limitations.
* **Streamlit UI:** A clean, intuitive dashboard interface optimized for speed and seamless file processing.

---

## 🧠 System Architecture & NLP Pipeline

The engine processes text data through a structured pipeline combining both deep learning and classical machine learning paradigms:

1. **Extraction Layer:** Uses custom file parsers (`PyPDF2`, `pdfplumber`, `python-docx`, `pandas`) to extract raw string text from diverse file types.
2. **Preprocessing Layer:** Tokenizes sentences using NLTK’s `punkt` tokenizer, handles whitespace normalization, and strips out noisy syntax (URLs, raw numeric formatting, and corrupted characters).
3. **Chunking Pipeline:** Dynamically groups text into fixed-size overlapping token chunks to satisfy input constraints of deep transformer models.
4. **Abstractive Core:** Generates semantic summaries across chunks using a deep learning pipeline.
5. **Statistical Ranking Engine:** Extracts domain-specific terms via keyword algorithms configured with a standard TF-IDF vectorizer. Sentences are scored dynamically based on a weighted combination of their matrix token frequencies and high-value keyword occurrences.

---

## 🛠️ Technical Stack

* **Frontend Framework:** Streamlit
* **Deep Learning & Transformers:** Hugging Face `transformers` API
* **Underlying Model Architecture:** `facebook/bart-large-cnn` (BART Sequence-to-Sequence Architecture)
* **Keyword & Feature Extraction:** KeyBERT, Scikit-Learn (`TfidfVectorizer`)
* **Natural Language Processing:** NLTK (Tokenization, Stopword Removal)
* **Document Parsing Libraries:** PyPDF2, pdfplumber, python-docx, Pandas

---

## 📊 Core Algorithms Explained

### Abstractive Summarization Strategy
Instead of simply pulling existing sentences, the platform routes chunked text inputs through a pretrained **BART (Bidirectional and Auto-Regressive Transformer)** model optimized on the CNN/DailyMail dataset. The system dynamically truncates text while preserving end-context integrity to safely fit model parameters.

### Hybrid Sentence Ranking Equation
To extract the most impactful document highlights, sentences are ranked using a custom heuristic formula:

$$\text{Combined Score} = \text{TF-IDF Vector Score} + (2 \times \text{Keyword Match Count})$$

* **TF-IDF Vector Score:** Reflects the statistical uniqueness of words in the sentence across the document space.
* **Keyword Match Count:** Calculated using a BERT-embedding based keyword distance metric. 

Sentences are heavily penalized and filtered out if they contain parsing artifacts like numeric listing structures, high ratios of uppercase strings, web links, or abnormally short sentence lengths.

---

## 📈 Performance & Capabilities

* **Context Length Mitigation:** Seamlessly processes documents exceeding standard model token constraints through abstractive map-reduce chunking workflows.
* **Data Privacy:** Runs model inferencing efficiently, ensuring data handling remains safe, secure, and isolated to the configured infrastructure.
* **Output Format:** Generates standard Markdown formatting structures containing clean structural summaries and bulleted insights.
