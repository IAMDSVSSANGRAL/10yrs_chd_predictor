from flask import Flask, jsonify
from src.predictor.logger import logging

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Welcome to the Ten-Year CHD Risk Prediction Pipeline!",
        "status": "Data Ingestion and Transformation components are ready."
    })

if __name__ == '__main__':
    logging.info("Starting the Flask app...")
    app.run(debug=True)
