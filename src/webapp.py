# src/webapp.py
import os
from flask import Flask, render_template, request
from predict import predict
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'), static_folder=os.path.join(os.path.dirname(__file__), '..', 'static'))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def handle_predict():
    if 'file' not in request.files:
        return "No file part", 400
    f = request.files['file']
    if f.filename == '':
        return "No selected file", 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(filepath)
    try:
        label = predict(filepath)
    except Exception as e:
        return f"Error during prediction: {e}", 500
    return render_template('index.html', prediction=label)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
