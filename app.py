from flask import Flask, render_template, request, jsonify
from utils.predictor import predict_news
from utils.metrics import get_model_metrics

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    model_name = data.get('model', 'bert')

    if not text:
        return jsonify({'error': 'No text provided'}), 400
    if len(text) < 20:
        return jsonify({'error': 'Text too short. Please enter at least 20 characters.'}), 400

    result = predict_news(text, model_name)
    return jsonify(result)

@app.route('/metrics')
def metrics():
    data = get_model_metrics()
    return jsonify(data)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
