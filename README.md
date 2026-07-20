# Semantic Book Recommender with LLMs

A modern, semantic book recommendation system that uses natural language processing (NLP), zero-shot classification, emotion analysis, and vector search to recommend books based on user queries, categories, and emotional tones.

The application features a clean, glassmorphism-themed **Gradio Dashboard** allowing users to input a descriptive prompt (e.g., *"A story about a young wizard overcoming adversity"*) and filter results by book category and desired emotional tone (e.g., Happy, Sad, Suspenseful, Angry, Surprising).

---

## Key Features

*   **Semantic Search**: Leverages OpenAI Embeddings and Chroma vector database to search books by conceptual meaning rather than exact keyword matches.
*   **Zero-shot Category Classification**: Employs `facebook/bart-large-mnli` model to classify books with missing or unmapped categories into broad classes like *Fiction* or *Nonfiction*.
*   **Emotion/Sentiment Analysis**: Utilizes `j-hartmann/emotion-english-distilroberta-base` to analyze the emotions present in book descriptions, scoring them on anger, disgust, fear, joy, neutral, sadness, and surprise.
*   **Interactive Gradio Dashboard**: Features a web UI with filters for book categories and emotional tones (sorting recommendations by matching emotions).

---

## Project Pipeline & Architecture

The project is designed in modular phases represented by Jupyter Notebooks:

```
    A[Kaggle Dataset: 7k Books] --> B[data-exploration.ipynb]
    B -->|books_cleaned.csv| C[text-classification.ipynb]
    B -->|books_cleaned.csv| E[vector-search.ipynb]
    C -->|books_categories.csv| D[sentiment-analysis.ipynb]
    D -->|books_with_emotions.csv| G[gradio-dashboard.py]
    E -->|tagged_description.txt| F[Vector Storage & Search]
    F -->|OpenAI Embeddings & Chroma| G
    G -->|Gradio Web App| H[User Recommendations]
```

### 1. Data Preprocessing (`data-exploration.ipynb`)
*   Downloads the **7k Books with Metadata** dataset from Kaggle.
*   Performs exploratory data analysis (EDA) and correlation checks.
*   Cleans missing fields and filters descriptions with fewer than 25 words.
*   Generates a cleaned dataset: `books_cleaned.csv`.

### 2. Category Classification (`text-classification.ipynb`)
*   Maps existing book categories to simple categories (*Fiction*, *Nonfiction*, *Children's Fiction*, *Children's Nonfiction*).
*   Applies a BART zero-shot classification pipeline to classify any unmapped or missing categories.
*   Saves the mapped dataset to `books_categories.csv`.

### 3. Emotion Analysis (`sentiment-analysis.ipynb`)
*   Applies a RoBERTa-based emotion classifier to the book descriptions sentence-by-sentence.
*   Extracts scores for various emotions (joy, sadness, anger, fear, surprise, disgust, neutral) by taking the maximum score across all sentences.
*   Saves the final enriched dataset to `books_with_emotions.csv`.

### 4. Semantic Indexing & Vector Search (`vector-search.ipynb`)
*   Prepares descriptions mapped with ISBN identifiers (`tagged_description.txt`).
*   Splits documents and creates a Chroma vector database using `OpenAIEmbeddings`.
*   Tests semantic similarity search queries.

---

## Setup and Installation

### 1. Clone the repository and navigate to the directory
```bash
git clone <repository-url>
cd book-recommender
```

### 2. Set up virtual environment
```bash
python -m venv .venv
# On Windows (CMD/PowerShell)
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your OpenAI API key:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Running the Application

Before running the Gradio dashboard, ensure the dataset pipeline notebooks have been executed in sequence to generate the necessary files (`books_with_emotions.csv` and `tagged_description.txt`).

To launch the web dashboard, run:
```bash
python gradio-dashboard.py
```
Open the local URL displayed in your console (usually `http://127.0.0.1:7860`) to interact with the recommender.

---

## Requirements

*   `python >= 3.10`
*   `gradio`
*   `langchain`
*   `langchain-chroma`
*   `langchain-openai`
*   `transformers`
*   `pandas`
*   `numpy`
*   `torch`
*   `kagglehub`
*   `python-dotenv`
