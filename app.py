from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer
import os

app = Flask(__name__)

# Dictionary to store loaded tokenizers
tokenizers = {}

# List of available models with their display names
AVAILABLE_MODELS = [
    {"id": "Qwen/Qwen2.5-Coder-32B-Instruct", "name": "Qwen2.5-Coder-32B"},
    {"id": "microsoft/phi-4", "name": "Phi-4"},
]

def get_tokenizer(model_id):
    """Load tokenizer if not already loaded"""
    if model_id not in tokenizers:
        try:
            tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            print(f"Error loading tokenizer for {model_id}: {e}")
            return None
    return tokenizers[model_id]

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
    
    tokenizer = get_tokenizer(model_id)
    if not tokenizer:
        return jsonify({"error": f"Failed to load tokenizer for {model_id}"}), 500
    
    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    
    # Get the actual tokens for display
    token_strings = []
    for token_id in tokens:
        token_string = tokenizer.decode([token_id])
        token_strings.append(token_string)
    
    return jsonify({
        "count": token_count,
        "tokens": token_strings
    })

if __name__ == '__main__':
    app.run(debug=True)
