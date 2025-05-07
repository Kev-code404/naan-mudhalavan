# AI Content Generator with DeepSeek R1 and NLTK

This web application uses GPT4All with DeepSeek R1 model and NLTK for content generation and text analysis.

## Prerequisites

- Python 3.8 or higher
- GPT4All installed with DeepSeek R1 model
- NLTK data packages

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the DeepSeek R1 model file (`deepseek-coder-1.3b.Q4_0.gguf`) in your GPT4All models directory.

3. Run the application:
```bash
python app.py
```

4. Open your web browser and navigate to `http://localhost:5000`

## Features

- Content generation using DeepSeek R1 model
- Text analysis using NLTK:
  - Sentence tokenization
  - Word count
  - Unique word count
  - Stopword removal

## Usage

1. Enter your prompt in the text area
2. Click "Generate Content"
3. View the generated content and text analysis results

## Note

The first time you run the application, it will download the required NLTK data packages automatically. 