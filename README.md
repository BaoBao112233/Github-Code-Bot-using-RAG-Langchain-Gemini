# Github-Code-Bot-using-RAG-Langchain-Gemini
Github Code Bot using RAG Langchain Gemini

## Overview
This project integrates various tools and APIs to manage and interact with GitHub issues, embedding them into a vector store for efficient retrieval, and providing an interactive agent for querying these issues.

## Features
- Fetch GitHub issues and convert them into document format for processing.
- Embed and store issues in a vector database using Chroma from LangChain.
- Provide an interactive agent to query the GitHub issues using natural language through a Google Generative AI model.

## Setup
To set up the project, follow these steps:

### Prerequisites
- Python 3.8 or higher
- pip for installing Python packages

### Installation
1. Clone the repository:
```bash
   git clone https://github.com/BaoBao112233/Github-Code-Bot-using-RAG-Langchain-Gemini.git
```
2. Install the required Python packages:
```bash
   pip install -r requirements.txt
```
   or
```bash
   pip3 install -r requirements.txt
```
3. Set up the environment variables:
   - Copy the `.env.example` file to `.env` and fill in the necessary API keys and model names:
```plaintext
    GEMINI_API_KEY=your-gemini-api-key
    MODEL_NAME=models/gemini-1.5-flash
    GITHUB_TOKEN=your-github-token
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
    HUGGINGFACE_TOKEN=your-huggingface-token
```

## Usage
To use the system, run the `main.py` script:
```bash
   python main.py
```
or
```bash
   python3 main.py
```

Follow the interactive prompts to query the GitHub issues or update the vector store.

### Key Functions
- **Fetch GitHub Issues**: Retrieves issues from a specified GitHub repository and processes them into a suitable format for embedding.
- **Connect to Vector Store**: Embeds the issues and stores them in a vector database for quick retrieval.
- **Interactive Agent**: Allows users to interactively query the GitHub issues using natural language.

## Code Structure
- `src/github.py`: Contains functions to fetch and process GitHub issues.
- `src/note.py`: Provides a simple tool for note-taking within the application.
- `main.py`: The main script that sets up the environment, initializes the vector store, and runs the interactive agent.

## Contributing
Contributions to this project are welcome. Please ensure to follow the existing code style and add unit tests for any new or changed functionality.

## License
Specify the license under which the project is released.

