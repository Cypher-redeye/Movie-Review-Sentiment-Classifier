# 🎬 Review AI: The Sentiment of Cinema

![UI Showcase](https://img.shields.io/badge/UI-Cinematic_Editorial-000000?style=flat-square)
![Framework](https://img.shields.io/badge/Backend-Flask-white?style=flat-square&logo=flask)
![AI](https://img.shields.io/badge/AI-HuggingFace_DistilBERT-FF9D00?style=flat-square&logo=huggingface)

An elegant, high-performance web application that uses state-of-the-art Artificial Intelligence to analyze and classify the emotional polarity (sentiment) of movie reviews. 

Designed with a high-fashion "Cinematic Editorial" aesthetic, this project marries beautiful UI/UX with cutting-edge Natural Language Processing.

## ✨ Features

- **State-of-the-Art NLP:** Powered by HuggingFace's `DistilBERT` (fine-tuned on the Stanford Sentiment Treebank), capable of understanding complex grammatical structures, nuance, and structural sentiment shifts out-of-the-box.
- **Cinematic UI/UX:** A bespoke, split-screen layout built with raw HTML/CSS/JS. Features massive kinetic typography (`Cormorant Garamond`), a pristine Alabaster interaction zone, and buttery-smooth reveal animations.
- **Zero-Friction Inference:** Raw text is passed directly to the Transformer pipeline, eliminating the need for clunky NLTK tokenization or manual stopword removal.
- **Legacy PyTorch Model:** Includes the original `sentiment_classifier.py` script—a complete pipeline for training a custom Bidirectional LSTM on the IMDB 50K dataset from scratch (useful for educational purposes).

## 🚀 Quick Start

### Prerequisites
Make sure you have Python 3 installed. You will also need the `transformers` library and a PyTorch backend.

```bash
pip install flask transformers torch
```

### Running the App
1. Clone the repository.
2. Start the Flask server:
```bash
python app.py
```
3. Navigate to `http://127.0.0.1:5000` in your browser.

*(Note: The first time you run the app, it will take ~15-30 seconds to download the DistilBERT model weights to your machine).*

## 🧠 The "Sarcasm" Problem (A Note on NLP)
While DistilBERT is incredibly accurate at reading structural sentiment, sarcasm remains the "Final Boss" of NLP. Because sarcasm relies on using overwhelmingly positive vocabulary to convey negative intent (e.g., *"Oh wow, another superhero movie with a sky beam. Groundbreaking."*), literal classification models will often score it as highly positive. Beating sarcasm requires either chained irony-detection models or Large Language Models (LLMs) with deep real-world context!

## 🛠 Tech Stack
- **Frontend:** HTML5, Vanilla CSS3, Vanilla JavaScript
- **Backend:** Python, Flask
- **Machine Learning:** HuggingFace Transformers (`distilbert-base-uncased-finetuned-sst-2-english`), PyTorch
