# AI-Powered Document Summarization, Insights Extraction

## Project Overview
This project combines two powerful AI-driven functionalities:
1. **Document Summarization and Insights Extraction:** An advanced tool for processing, analyzing, and summarizing documents.


## Features

### Document Summarization and Insights Extraction
- **Summarization:** Generate concise summaries of long documents using Hugging Face's BART-large model.
- **Keyword Extraction:** Extract the most relevant keywords from the document using TF-IDF vectorization.
- **Highlights:** Identify and present key sentences that align with extracted keywords.
- **Key Insights:** Extract actionable insights from the content for quick decision-making.
- **Multi-format Support:** Process text from PDFs, DOCX files, plain text, and CSV files.
- **User-Friendly Interface:** Powered by Streamlit for seamless interaction.



## Getting Started

### Prerequisites
- Python 3.8 or later
- Install required packages using:

```bash
pip install -r requirements.txt
```

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/document-summarization
   ```
2. Navigate to the project directory:
   ```bash
   cd document-summarization
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### Document Summarization Tool
1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open the provided local URL in your browser to access the interface.



## Workflow and Architecture

### Document Summarization
1. **Text Extraction:** Extract text from various document formats using libraries like PyPDF2, pdfplumber, and python-docx.
2. **Text Cleaning:** Preprocess text by removing unwanted characters and whitespace.
3. **Chunking:** Split large documents into manageable chunks to ensure efficient processing.
4. **Summarization:** Use the Hugging Face BART-large model to generate concise summaries.
5. **Keyword Extraction:** Employ TF-IDF vectorization to identify the most relevant keywords.
6. **Highlights:** Match keywords to sentences in the text to create meaningful highlights.
7. **Key Insights:** Generate actionable insights using NLP prompts and summarization models.

## Models and Metrics

### Model
- **Hugging Face BART-large:** A transformer-based model for summarization tasks.

### Metrics
- **Rouge Score:** Measures the quality of summaries by comparing them to reference summaries.
  - **Rouge-1:** Unigram overlap.
  - **Rouge-2:** Bigram overlap.
  - **Rouge-L:** Longest common subsequence.

### Observations
- Longer inputs enhance summary quality while preserving conciseness.
- High textual content leads to better keyword relevance.

## Lessons Learned
1. **Effective Chunking:** Breaking down large documents improves model performance.
2. **Balancing Quality and Speed:** Optimizing hyperparameters is key to maintaining summary quality and processing efficiency.
3. **User Experience Matters:** A clean and intuitive interface boosts usability.

## Future Enhancements
- Add multilingual support for summarization.
- Integrate additional NLP models for comparison.
- Explore real-time document processing.
- Enhance chatbot functionality with real-time feedback and expanded response capabilities.

## Contributors
- Salma Kassem
- Habiba Hakiem

## Acknowledgments
Special thanks to Hugging Face, NLTK, and Scikit-learn for their incredible tools and libraries.


