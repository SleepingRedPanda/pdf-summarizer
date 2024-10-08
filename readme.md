# PDF Summarizer

## Description
PDF Summarizer is a tool that extracts structured information from PDF documents using Langchain and OpenAI models. The application extracts key details such as the title, author, and publication year, and generates a concise summary of the document.

## Features
- Extracts title, author, and publication year from PDF documents
- Generates a short summary of the PDF content
- Utilizes Langchain and OpenAI models for information extraction and summarization
- User-friendly interface built with Streamlit

## Installation

To run this application, you'll need Docker installed on your system. Then follow these steps:

1. Pull the Docker image:
   ```bash
   docker pull redsleepingpanda/default-repo:latest
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8501:8501 redsleepingpanda/default-repo:latest
   ```

3. Open your web browser and navigate to `http://localhost:8501` to access the application.

## Usage

1. When you open the application, you'll be prompted to insert your OpenAI API key.
2. After entering your API key, you can select a PDF file for processing.
3. The application will extract the structured information and generate a summary.
4. Review the extracted information and summary on the interface.
