from flask import Flask, render_template, request, jsonify
from gpt4all import GPT4All
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Initialize GPT4All
def initialize_model():
    try:
        # Use a compatible model
        model_path = os.path.join(os.path.expanduser("~"), "AppData", "Local", "nomic.ai", "GPT4All")
        logger.info(f"Attempting to load model from {model_path}")
        
        # Try to load with GPU first, fall back to CPU if GPU fails
        try:
            model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf", model_path=model_path, device='gpu')
            logger.info("GPT4All model loaded successfully with GPU")
        except Exception as gpu_error:
            logger.warning(f"GPU loading failed: {gpu_error}. Falling back to CPU.")
            model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf", model_path=model_path, device='cpu')
            logger.info("GPT4All model loaded successfully with CPU")
            
        return model
    except Exception as e:
        logger.error(f"Error loading GPT4All model: {e}")
        return None

# Initialize the model
model = initialize_model()

def generate_with_gpt4all(prompt):
    global model
    try:
        if model is None:
            model = initialize_model()
            if model is None:
                return None, "Failed to initialize the model. Please try again."
        
        logger.info(f"Generating response for prompt: {prompt}")
        
        # Enhanced prompt template for better content generation
        full_prompt = f"""<|im_start|>system
You are an expert content writer and researcher. Your task is to create comprehensive, well-structured, and engaging content. Follow these guidelines:
1. Start with a clear introduction that hooks the reader
2. Use clear headings and subheadings to organize content
3. Include relevant examples and real-world applications
4. Support key points with data or expert opinions when possible
5. Write in a professional yet engaging tone
6. End with a strong conclusion that summarizes key takeaways
7. Ensure the content is informative, accurate, and valuable to the reader
<|im_end|>
<|im_start|>user
Write a detailed, well-structured article about {prompt}. Make it comprehensive and engaging while maintaining professional standards.
<|im_end|>
<|im_start|>assistant
"""
        
        logger.info(f"Using enhanced prompt format: {full_prompt}")
        
        try:
            # Optimized generation parameters for better quality
            response = model.generate(
                full_prompt,
                max_tokens=2000,     # Increased for more comprehensive content
                temp=0.8,           # Slightly increased for more creative content
                top_k=50,           # Increased for more diverse word choices
                top_p=0.95,         # Increased for more natural flow
                repeat_penalty=1.2,  # Increased to reduce repetition
                n_batch=8,          # Maintained for GPU efficiency
                streaming=False      # Disable streaming for better logging
            )
        except Exception as gen_error:
            logger.error(f"Generation error: {gen_error}")
            # Try to reinitialize the model
            model = initialize_model()
            if model is None:
                return None, "Model encountered an error and failed to recover. Please try again."
            # Retry generation once
            response = model.generate(
                full_prompt,
                max_tokens=2000,
                temp=0.8,
                top_k=50,
                top_p=0.95,
                repeat_penalty=1.2,
                n_batch=8,
                streaming=False
            )
        
        logger.info(f"Raw response from model: {response}")
        
        # Check if response is empty
        if not response or len(response.strip()) == 0:
            logger.error("Model generated an empty response")
            return None, "Model generated an empty response. Please try again with a different prompt."
        
        # Post-process the response to clean up formatting
        response = response.strip()
        
        # Remove prompt markers
        response = response.replace("<|im_start|>", "").replace("<|im_end|>", "")
        
        # Remove any remaining assistant/user markers
        response = response.replace("assistant:", "").replace("user:", "")
        
        # Clean up any extra whitespace
        response = " ".join(response.split())
        
        # Ensure proper sentence endings
        if not response.endswith(('.', '!', '?')):
            response += '.'
            
        logger.info(f"Successfully generated response of length: {len(response)}")
        return response, None
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logger.error(error_msg)
        # Try to reinitialize the model for next time
        model = initialize_model()
        return None, error_msg

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_content():
    try:
        prompt = request.json.get('prompt', '')
        logger.info(f"Received prompt: {prompt}")
        
        if not prompt:
            return jsonify({
                "error": "Please provide a prompt"
            }), 400
        
        # Generate content using GPT4All
        response, error = generate_with_gpt4all(prompt)
        
        if error:
            return jsonify({
                "error": error
            }), 500
        
        if response is None:
            return jsonify({
                "error": "Failed to generate response. Please try again."
            }), 500
        
        # Process the generated text with NLTK
        nltk_analysis = process_text_with_nltk(response)
        
        return jsonify({
            'generated_text': response,
            'analysis': nltk_analysis
        })
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            "error": error_msg
        }), 500

def process_text_with_nltk(text):
    # Tokenize sentences
    sentences = sent_tokenize(text)
    
    # Tokenize words and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    
    return {
        'sentences': sentences,
        'word_count': len(filtered_words),
        'unique_words': len(set(filtered_words))
    }

if __name__ == '__main__':
    app.run(debug=True) 