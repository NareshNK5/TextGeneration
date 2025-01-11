from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can choose other models as well, like 'gpt2-medium', 'gpt2-large', or 'gpt3' (via API)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_text(seed_text, max_length=100):
    inputs = tokenizer.encode(seed_text, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    seed_text = data.get('seed_text', '')
    length = int(data.get('length', 100))
    generated_text = generate_text(seed_text, max_length=length)
    return jsonify({'generated_text': generated_text})
