from flask import Flask, request, jsonify, render_template
import pandas as pd
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Load pre-trained sentiment analysis model
model = keras.models.load_model('sentiment_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    prediction = model.predict([text])
    sentiment = 'neutral'
    if prediction > 0.5:
        sentiment = 'positiv'
    elif prediction < 0.5:
        sentiment = 'negativ'
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
