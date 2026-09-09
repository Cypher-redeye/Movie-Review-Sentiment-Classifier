import os
from flask import Flask, render_template, request, jsonify

# Suppress HuggingFace/TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from transformers import pipeline

app = Flask(__name__)

# ====================================================================
# TRANSFORMER PIPELINE INITIALIZATION
# ====================================================================
# DistilBERT fine-tuned on SST-2 (Stanford Sentiment Treebank)
# This model understands context, sarcasm, and complex sentence structure
# natively. No custom NLTK preprocessing is required.
sentiment_pipeline = None

try:
    print("Loading Transformer model (this may take a few seconds...)")
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=-1 # CPU
    )
    print("✅ Transformer loaded successfully!")
except Exception as e:
    print("❌ Failed to load Transformer. Is the 'transformers' library installed?", e)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    if sentiment_pipeline is None:
        return jsonify({'error': 'Transformer model not loaded. Please restart the app.'}), 503

    text = data['text']

    try:
        # The pipeline handles all tokenization, context analysis, and inference!
        # We truncate to 512 tokens to prevent length errors on huge essays
        result = sentiment_pipeline(text, truncation=True, max_length=512)[0]
        
        raw_label = result['label'].lower()
        confidence = result['score']
        
        if raw_label == "positive":
            label = "Positive"
        elif raw_label == "negative":
            label = "Negative"
        else:
            label = "Neutral"

        return jsonify({
            'label': label,
            'confidence': float(confidence),
            'probability': float(confidence)
        })
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting Web UI on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
