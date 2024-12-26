import streamlit as st
import os
import io
#import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
import pdfplumber
import nltk
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Hugging Face summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class EnhancedSummarizer:
    def __init__(self, summarizer):
        self.summarizer = summarizer

    def summarize(self, text: str) -> str:
        max_length = 1024  # Adjust based on model limits
        truncated_text = text[:max_length]
        summary = self.summarizer(
            truncated_text,
            max_length=150,  # Adjust summary length
            min_length=50,
            do_sample=False
        )[0]['summary_text']
        return summary

    def extract_keywords(self, text: str, num_keywords: int = 5) -> list:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=num_keywords)
        vectors = vectorizer.fit_transform([text])
        keywords = vectorizer.get_feature_names_out()
        return list(keywords)

    def generate_highlights(self, text: str, keywords: list) -> list:
        sentences = sent_tokenize(text)
        highlights = [
            f"- 📊 *{sentence.strip()}*"
            for sentence in sentences
            if any(keyword in sentence.lower() for keyword in keywords)
        ]
        return highlights[:7]  # Limit to top 7 highlights

    def generate_key_insights(self, text: str) -> list:
        prompt = f"Extract actionable insights from this text: {text}"
        insights = self.summarizer(
            prompt,
            max_length=100,
            min_length=50,
            do_sample=False
        )[0]['summary_text']
        insights_sentences = sent_tokenize(insights)
        return [f"- 🔍 *{sentence.strip()}*" for sentence in insights_sentences]

# Create an instance of EnhancedSummarizer
enhanced_summarizer = EnhancedSummarizer(summarizer)

# Function to extract text from DOCX file
def extract_text_from_docx(docx_path):
    document = Document(docx_path)
    text = ""
    for para in document.paragraphs:
        text += para.text + "\n"
    return text

# Function to extract text from PDF file using pdfplumber
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to extract text from CSV file
def extract_text_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    text = df.to_string(index=False)  # Converts the entire dataframe to a string
    return text

# Function to clean the extracted text (optional, depends on your use case)
def clean_text(text: str) -> str:
    # Remove unwanted characters, extra spaces, and clean up
    text = text.replace("\n", " ").strip()
    text = " ".join(text.split())  # Removes extra spaces
    return text

# Function to split large documents into manageable chunks
def chunk_text(text, chunk_size=1000):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Function to generate structured output
def generate_structured_output(text):
    cleaned_text = clean_text(text)
    
    # Split large document into chunks
    text_chunks = chunk_text(cleaned_text)

    # Generate summaries for each chunk
    summaries = [enhanced_summarizer.summarize(chunk) for chunk in text_chunks if chunk.strip()]
    full_summary = " ".join(summaries)

    # Limit the summary to 10 sentences
    summary_sentences = full_summary.split(". ")
    limited_summary = ". ".join(summary_sentences[:10]) + "."  # Add a period at the end if truncated

    # Extract key sentences for highlights
    important_sentences = summary_sentences[:7]  # Take the first 7 sentences for highlights

    # Generate highlights from the key sentences
    highlights = [
        f"- 📊 *{sentence.strip()}*" for sentence in important_sentences if sentence.strip()
    ]

    # Generate Key Insights from the full summary
    key_insights_prompt = f"Extract key insights from this text: {limited_summary}"
    key_insights = enhanced_summarizer.summarizer(key_insights_prompt, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

    # Structure the Key Insights
    key_insights_list = key_insights.split(". ")
    structured_insights = [
        f"- 🔍 *{insight.strip()}*" for insight in key_insights_list if insight.strip()
    ]

    # Combine all into structured output
    structured_output = (
        "### Summary\n"
        f"{limited_summary}\n\n"
        "### Highlights\n" +
        "\n".join(highlights) + 
        "\n\n### Key Insights\n" +
        "\n".join(structured_insights) + 
        "\n"
    )
    return structured_output

# Streamlit frontend
def main():
    st.title("Documents Summarization")

    # Text input field to allow user to enter text directly
    text_input_from_user = st.text_area("Type or paste text directly here", height=300)

    # File upload option
    uploaded_file = st.file_uploader("Or upload a file", type=["txt", "pdf", "docx", "csv"])

    if st.button("Generate Summary and Insights"):
        if text_input_from_user.strip():
            # If text is entered directly by the user
            text_input = text_input_from_user
            st.success("Text input received successfully.")
        elif uploaded_file is not None:
            # If file is uploaded, read the file content
            if uploaded_file.name.endswith('.txt'):
                text_input = uploaded_file.read().decode("utf-8")
                st.success(f"File uploaded successfully: {uploaded_file.name}")
            elif uploaded_file.name.endswith('.pdf'):
                # Reading PDF file
                pdf_reader = PdfReader(uploaded_file)
                text_input = ""
                for page in pdf_reader.pages:
                    text_input += page.extract_text()
                st.success(f"PDF file uploaded successfully: {uploaded_file.name}")
            elif uploaded_file.name.endswith('.docx'):
                # Reading DOCX file
                doc = Document(uploaded_file)
                text_input = "\n".join([para.text for para in doc.paragraphs])
                st.success(f"File uploaded successfully: {uploaded_file.name}")
            elif uploaded_file.name.endswith('.csv'):
                # Reading CSV file
                df = pd.read_csv(uploaded_file)
                text_input = df.to_string(index=False)
                st.success(f"CSV file uploaded successfully: {uploaded_file.name}")
            else:
                st.error("Please upload a valid .txt, .pdf, .docx, or .csv file.")
                return
        else:
            st.error("Please either enter text or upload a file.")
            return
        
        if text_input.strip():
            # Generate structured output from the backend
            structured_output = generate_structured_output(text_input)
            
            # Display the output on Streamlit
            st.subheader("Structured Output:")
            st.markdown(structured_output)
        else:
            st.error("The file is empty, unreadable, or no text provided. Please try again!")

if __name__ == "__main__":
    main()
