# RALF: Retrieval Agent for Long Files

**RALF (Retrieval Agent for Long Files)** is a powerful and intuitive tool designed to help you retrieve specific information from large text-based files. Whether you're a researcher, a student, or anyone who works with dense documents, RALF makes it easy to ask questions and get precise answers with highlighted source text.

## Features

- **Question-Answering Mode**: Ask questions in natural language and get direct answers from your documents.
- **Retrieval Mode**: Extract the smallest possible quotes that directly answer your query, complete with source highlighting.
- **Multi-File Support**: Upload and query multiple files at once, including PDF, DOCX, and TXT formats.
- **Intuitive UI**: A clean and simple interface built with Gradio makes it easy to upload files and interact with the chatbot.
- **Secure**: Your documents and API key are kept private. Only the necessary API calls leave your machine.

## Project Structure

```
/home/bastien/RALF/
├──.env
├──.env.example
├──.git/
├──chatbot.py
├──config.py
├──corpus.py
├──llm.py
├──main.py
├──README.md
├──requirements.txt
├──ui.py
└──utils.py
```

## Setup and Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/khulnasoft/ralf.git
   cd ralf
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your environment variables**:

   - Create a `.env` file by copying the example file:

     ```bash
     cp .env.example .env
     ```

   - Open the `.env` file and add your OpenAI API key:

     ```
     OPENAI_API_KEY="YOUR_API_KEY_HERE"
     ```

5. **Run the application**:

   ```bash
   python main.py
   ```

   The application will be accessible at a local or shared URL provided by Gradio.

## How It Works

RALF uses a combination of a large language model (LLM) and a custom retrieval system to provide answers. When you upload your documents, they are stored in a corpus. When you ask a question, the relevant text from the corpus is passed to the LLM, which then generates an answer based on the provided information.