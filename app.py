from flask import Flask, render_template, request, jsonify, url_for
from transformers import AutoTokenizer
import os
import traceback
import tiktoken
app = Flask(__name__, 
            static_folder=os.path.abspath('static'),
            static_url_path='/static')

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
                # OpenAI models use tiktoken for accurate tokenization
                print(f"Loading tiktoken for OpenAI model {model_id}...")
                if model_id.startswith("gpt-4"):
                    encoding_name = "cl100k_base"  # GPT-4 uses cl100k_base encoding
                elif model_id.startswith("gpt-3.5"):
                    encoding_name = "cl100k_base"  # GPT-3.5-turbo also uses cl100k_base
                else:
                    encoding_name = "p50k_base"    # Default for older models
                
                tokenizers[model_id] = {
                    "type": "tiktoken",
                    "encoding": tiktoken.get_encoding(encoding_name),
                    "name": model_id
                }
                print(f"Successfully loaded tiktoken for {model_id}")
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

# Configure Flask for subdirectory deployment if needed
# This helps when the app is not deployed at the root of the domain
app.config['APPLICATION_ROOT'] = '/'
app.config['PREFERRED_URL_SCHEME'] = 'https'

@app.route('/')
def index():
    return render_template('index.html', models=AVAILABLE_MODELS)

# Make sure routes work regardless of trailing slash
@app.route('/count_tokens/', methods=['POST', 'GET'])
@app.route('/count_tokens', methods=['POST', 'GET'])  # Allow both POST and GET for debugging
def count_tokens():
    # Add debugging information
    print(f"Request method: {request.method}")
    print(f"Request headers: {request.headers}")
    
    # Handle both GET and POST for testing
    if request.method == 'GET':
        return jsonify({"status": "Token counting endpoint is working. Use POST for actual counting."}), 200
    
    # Process POST request
    data = request.json
    print(f"Received data: {data}")
    
    model_id = data.get('model_id')
    text = data.get('text', '')
    
    if not model_id or not text:
        return jsonify({"error": "Missing model_id or text"}), 400
    
    tokenizer = get_tokenizer(model_id)
    if not tokenizer:
        return jsonify({"error": f"Failed to load tokenizer for {model_id}"}), 500
    
    # Process based on tokenizer type
    if isinstance(tokenizer, dict) and tokenizer.get("type") == "tiktoken":
        # Use tiktoken for OpenAI models
        encoding = tokenizer["encoding"]
        token_ids = encoding.encode(text)
        token_count = len(token_ids)
        
        # Get the actual tokens for display
        tokens = []
        for token_id in token_ids:
            token_bytes = encoding.decode_single_token_bytes(token_id)
            try:
                token_text = token_bytes.decode('utf-8')
            except UnicodeDecodeError:
                token_text = repr(token_bytes)[2:-1]  # Remove b' and ' from repr
            tokens.append(token_text)
        
        # Not an approximation since we're using the official tokenizer
        is_approximation = False
    else:
        # Use transformers tokenizer
        encoding = tokenizer.encode(text, add_special_tokens=False)
        token_count = len(encoding)
        
        # Get the actual tokens for display
        tokens = []
        for token_id in encoding:
            token_text = tokenizer.decode([token_id])
            tokens.append(token_text)
        
        # Only Claude is an approximation now
        is_approximation = model_id.startswith("claude-")
    
    return jsonify({
        "count": token_count,
        "tokens": tokens,
        "is_approximation": is_approximation
    })

if __name__ == '__main__':
    try:
        import gunicorn.app.base
        
        class StandaloneApplication(gunicorn.app.base.BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()

            def load_config(self):
                for key, value in self.options.items():
                    self.cfg.set(key.lower(), value)

            def load(self):
                return self.application

        options = {
            'bind': '0.0.0.0:8003',
            'workers': 4,
        }
        print("Starting production server with Gunicorn...")
        StandaloneApplication(app, options).run()
    except ImportError:
        print("Gunicorn not found. Starting development server...")
        print("WARNING: For production use, install gunicorn: pip install gunicorn")
        app.run(debug=True)
