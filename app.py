from flask import Flask, render_template, request, jsonify
from tokenizers import Tokenizer
import os
import json
import re
from pathlib import Path

app = Flask(__name__)

# Dictionary to store loaded tokenizers
tokenizers = {}

# List of available models with their display names
AVAILABLE_MODELS = [
    {"id": "Qwen/Qwen2.5-Coder-32B-Instruct", "name": "Qwen2.5-Coder-32B"},
    {"id": "microsoft/phi-4", "name": "Phi-4"},
]

# Simple tokenization methods for demonstration
def simple_word_tokenize(text):
    """Simple word tokenizer that splits on whitespace and punctuation"""
    # Split on whitespace and keep punctuation
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens

def get_tokenizer(model_id):
    """Load tokenizer if not already loaded"""
    if model_id not in tokenizers:
        try:
            # For all models, use simple tokenization as a fallback
            # In a production app, you would download the actual tokenizers
            if model_id == "Qwen/Qwen2.5-Coder-32B-Instruct":
                return {"type": "simple", "name": "Qwen2.5-Coder-32B (Approximation)"}
            elif model_id == "microsoft/phi-4":
                return {"type": "simple", "name": "Phi-4 (Approximation)"}
            else:
                return {"type": "simple", "name": f"{model_id} (Approximation)"}
        except Exception as e:
            print(f"Error loading tokenizer for {model_id}: {e}")
            return {"type": "simple", "name": f"{model_id} (Approximation)"}
    
    return tokenizers.get(model_id)

@app.route('/')
def index():
    return render_template('index.html', models=AVAILABLE_MODELS)

@app.route('/count_tokens', methods=['POST'])
def count_tokens():
    data = request.json
    model_id = data.get('model_id')
    text = data.get('text', '')
    
    if not model_id or not text:
        return jsonify({"error": "Missing model_id or text"}), 400
    
    tokenizer_info = get_tokenizer(model_id)
    if not tokenizer_info:
        return jsonify({"error": f"Failed to load tokenizer for {model_id}"}), 500
    
    # Process based on tokenizer type
    if tokenizer_info["type"] == "tokenizer":
        # Use the actual tokenizer
        encoding = tokenizer_info["tokenizer"].encode(text)
        tokens = encoding.tokens
        token_count = len(tokens)
    else:
        # Use simple tokenization
        tokens = simple_word_tokenize(text)
        token_count = len(tokens)
    
    return jsonify({
        "count": token_count,
        "tokens": tokens
    })

if __name__ == '__main__':
    app.run(debug=True)
