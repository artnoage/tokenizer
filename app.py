from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer
import os
import traceback

app = Flask(__name__)

# Dictionary to store loaded tokenizers
tokenizers = {}

# List of available models with their display names
AVAILABLE_MODELS = [
    {"id": "Qwen/Qwen2.5-Coder-32B-Instruct", "name": "Qwen"},
    {"id": "microsoft/phi-4", "name": "Phi-4"},
    {"id": "meta-llama/Llama-3.3-70B-Instruct", "name": "Llama 3.3"},
    {"id": "gpt-4", "name": "GPT-4 (OpenAI)"},
    {"id": "gpt-3.5-turbo", "name": "GPT-3.5 (OpenAI)"},
    {"id": "claude-3-opus", "name": "Claude 3 Opus (Anthropic)"},
    {"id": "claude-3-sonnet", "name": "Claude 3 Sonnet (Anthropic)"},
]

def get_tokenizer(model_id):
    """Load tokenizer if not already loaded"""
    if model_id not in tokenizers:
        try:
            # Handle special cases for non-HF models
            if model_id.startswith("gpt-"):
                # OpenAI models use cl100k_base tokenizer (for GPT-3.5/4)
                print(f"Loading cl100k tokenizer for OpenAI model {model_id}...")
                tokenizers[model_id] = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
                print(f"Successfully loaded tokenizer for {model_id}")
            elif model_id.startswith("claude-"):
                # Claude models use similar tokenization to Llama 2
                print(f"Loading tokenizer for Anthropic model {model_id}...")
                tokenizers[model_id] = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
                print(f"Successfully loaded tokenizer for {model_id}")
            else:
                # Regular Hugging Face models
                print(f"Loading tokenizer for {model_id}...")
                tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id, use_fast=True)
                print(f"Successfully loaded tokenizer for {model_id}")
        except Exception as e:
            print(f"Error loading tokenizer for {model_id}: {e}")
            traceback.print_exc()
            return None
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
    
    tokenizer = get_tokenizer(model_id)
    if not tokenizer:
        return jsonify({"error": f"Failed to load tokenizer for {model_id}"}), 500
    
    # Tokenize the text
    encoding = tokenizer.encode(text, add_special_tokens=False)
    token_count = len(encoding)
    
    # Get the actual tokens for display
    tokens = []
    for token_id in encoding:
        token_text = tokenizer.decode([token_id])
        tokens.append(token_text)
    
    # Add note for approximated tokenizers
    is_approximation = model_id.startswith(("gpt-", "claude-"))
    
    return jsonify({
        "count": token_count,
        "tokens": tokens,
        "is_approximation": is_approximation
    })

if __name__ == '__main__':
    app.run(debug=True)
