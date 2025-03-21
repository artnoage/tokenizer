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
    {"id": "gpt2", "name": "GPT-2"},
    {"id": "cl100k_base", "name": "OpenAI cl100k (GPT-3.5/4)"},
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
            if model_id == "gpt2":
                # Use a pre-trained tokenizer if available locally
                cache_dir = Path.home() / ".cache" / "huggingface" / "tokenizers"
                cache_dir.mkdir(parents=True, exist_ok=True)
                tokenizer_path = cache_dir / "gpt2.json"
                
                if not tokenizer_path.exists():
                    # If not available, use simple tokenization
                    return {"type": "simple", "name": "GPT-2 (Simple Approximation)"}
                
                tokenizers[model_id] = {
                    "type": "tokenizer",
                    "tokenizer": Tokenizer.from_file(str(tokenizer_path)),
                    "name": "GPT-2"
                }
            elif model_id == "cl100k_base":
                # Simple approximation for cl100k
                return {"type": "simple", "name": "OpenAI cl100k (Simple Approximation)"}
            else:
                return None
        except Exception as e:
            print(f"Error loading tokenizer for {model_id}: {e}")
            return {"type": "simple", "name": f"{model_id} (Simple Approximation)"}
    
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
