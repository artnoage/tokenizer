document.addEventListener('DOMContentLoaded', function() {
    const modelSelect = document.getElementById('model-select');
    const textInput = document.getElementById('text-input');
    const calculateBtn = document.getElementById('calculate-btn');
    const tokenCount = document.getElementById('token-count');
    const tokenList = document.getElementById('token-list');
    const results = document.getElementById('results');
    const loading = document.getElementById('loading');

    calculateBtn.addEventListener('click', async function() {
        const modelId = modelSelect.value;
        const text = textInput.value.trim();
        
        if (!text) {
            alert('Please enter some text to tokenize');
            return;
        }
        
        // Show loading indicator
        loading.style.display = 'block';
        results.style.display = 'none';
        
        try {
            const response = await fetch('/count_tokens', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_id: modelId,
                    text: text
                }),
            });
            
            if (!response.ok) {
                throw new Error('Server error');
            }
            
            const data = await response.json();
            
            // Update token count
            tokenCount.textContent = data.count;
            
            // Display token breakdown
            tokenList.innerHTML = '';
            
            // Add approximation note if needed
            if (data.is_approximation) {
                const noteElement = document.createElement('div');
                noteElement.className = 'approximation-note';
                noteElement.textContent = 'Note: This is an approximation as the exact tokenizer is not publicly available.';
                tokenList.appendChild(noteElement);
            }
            
            // Add tokens
            data.tokens.forEach(token => {
                const tokenElement = document.createElement('span');
                tokenElement.className = 'token';
                // Replace spaces with visible space character for display
                const displayToken = token.replace(/ /g, '␣');
                tokenElement.textContent = displayToken;
                tokenList.appendChild(tokenElement);
            });
            
            // Show results
            results.style.display = 'block';
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while calculating tokens. Please try again.');
        } finally {
            // Hide loading indicator
            loading.style.display = 'none';
        }
    });
});
