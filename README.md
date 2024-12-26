# AI-Powered Document Summarization, Insights Extraction, and FAQ Chatbot

## Project Overview
This project combines two powerful AI-driven functionalities:
1. **Document Summarization and Insights Extraction:** An advanced tool for processing, analyzing, and summarizing documents.
2. **FAQ Chatbot for Café by Mars:** A chatbot designed to provide pre-defined responses to frequently asked questions about a café's operations.

## Features

### Document Summarization and Insights Extraction
- **Summarization:** Generate concise summaries of long documents using Hugging Face's BART-large model.
- **Keyword Extraction:** Extract the most relevant keywords from the document using TF-IDF vectorization.
- **Highlights:** Identify and present key sentences that align with extracted keywords.
- **Key Insights:** Extract actionable insights from the content for quick decision-making.
- **Multi-format Support:** Process text from PDFs, DOCX files, plain text, and CSV files.
- **User-Friendly Interface:** Powered by Streamlit for seamless interaction.

### FAQ Chatbot for Café by Mars
- **Pre-defined Contextual Responses:** Handles frequently asked questions about café hours, menu, location, contact details, specials, and more.
- **Keyword Matching:** Matches user queries with relevant responses based on keywords.
- **Default Responses:** Provides a polite fallback response if the input is unclear.

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

#### FAQ Chatbot
1. Run the chatbot:
   ```bash
   python chatbot.py
   ```
2. Interact with the bot by asking questions like:
   - "What are your hours?"
   - "Do you offer free Wi-Fi?"
   - "What's today's special?"

3. Exit anytime by typing `exit` or `quit`.

### Example FAQ Responses
The chatbot uses the following context for responses:

| **Keyword** | **Response** |
|-------------|--------------|
| `hours`     | "We are open from 8:00 AM to 12:00 AM, Sunday through Sunday." |
| `location`  | "Mars is located at 123 Main Street, Downtown." |
| `menu`      | "Our menu includes coffee, tea, latte, Spanish latte, hot chocolate, and sandwiches." |
| `contact`   | "You can reach us at (123) 456-7890 or email us at contact@bymars.com." |
| `wifi`      | "Yes, we offer free Wi-Fi. The password is 'MarsCafe2024121'." |
| `specials`  | "Today's specials include Pumpkin Spice Latte and Blueberry Muffins." |
| `prices`    | "Coffee is 20LE, Sandwiches are 15LE, tea is 15LE, Spanish latte is 30LE, and latte is 25LE." |

## Workflow and Architecture

### Document Summarization
1. **Text Extraction:** Extract text from various document formats using libraries like PyPDF2, pdfplumber, and python-docx.
2. **Text Cleaning:** Preprocess text by removing unwanted characters and whitespace.
3. **Chunking:** Split large documents into manageable chunks to ensure efficient processing.
4. **Summarization:** Use the Hugging Face BART-large model to generate concise summaries.
5. **Keyword Extraction:** Employ TF-IDF vectorization to identify the most relevant keywords.
6. **Highlights:** Match keywords to sentences in the text to create meaningful highlights.
7. **Key Insights:** Generate actionable insights using NLP prompts and summarization models.

### FAQ Chatbot
1. **Keyword Matching:** Identify user intent based on pre-defined keywords.
2. **Contextual Responses:** Provide accurate responses from the pre-defined context dictionary.
3. **Default Fallback:** Handle unknown queries with a polite response.

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


