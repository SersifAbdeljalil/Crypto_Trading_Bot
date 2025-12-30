"""
Flask app for the trading agent. Backend components for the trading dashboard interface.
"""
from pathlib import Path
import sys
import os
import subprocess

import pandas as pd
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from flask_cors import CORS
from csv import DictWriter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from dotenv import load_dotenv

# ----------------------------
# Load environment variables
# ----------------------------
# Load .env from the root of the project
dotenv_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Optional: print to check
print("SECRET_KEY:", os.getenv("SECRET_KEY"))
print("API_KEY:", os.getenv("API_KEY"))

# ----------------------------
# Add project root to sys.path
# ----------------------------
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root / 'src'))

# ----------------------------
# Imports after sys.path append
# ----------------------------
from data_handler.crypto_news_scraper import CryptoNewsScraper
from trading_bot.trading_bot import bot
import configs.config

# Download NLTK lexicon
nltk.download('vader_lexicon')

# ----------------------------
# Initialize Flask app
# ----------------------------
app = Flask(__name__)
api = Api(app)
CORS(app)  # Enable CORS

@app.route("/")
def home():
    return "Trading Bot flask app is successfully running."

# ----------------------------
# Helper to run scripts as subprocesses
# ----------------------------
def run_script(script_name):
    print(f'Starting subprocess for {script_name}...')
    subprocess.Popen(["python", script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(f'Subprocess for {script_name} started.')

# Start background tasks
run_script("app/run_top_news_scraper.py")
run_script("app/run_all_news_scraper.py")
run_script("app/run_trading_bot.py")

# ----------------------------
# REST API Resources
# ----------------------------
class AllTransactionHistory(Resource):
    def get(self, limit):
        df = pd.read_csv("app/output_data/transaction_history.csv")
        limited_data = df.tail(limit).iloc[::-1]
        return limited_data.to_dict(orient="records")

api.add_resource(AllTransactionHistory, "/all_transaction_history/<int:limit>")

class News(Resource):
    def get(self, type_, limit):
        path = f'app/output_data/{type_}News.csv'
        df = pd.read_csv(path)
        if df.shape[1] != 4:
            return jsonify({"error": "CSV file format is incorrect. Expected columns: title, link, date, article"}), 400
        df.fillna("Empty", inplace=True)
        sia = SentimentIntensityAnalyzer()

        def get_sentiment(article):
            score = sia.polarity_scores(article)
            if score['compound'] >= 0.05:
                return "Positive"
            elif score['compound'] <= -0.05:
                return "Negative"
            else:
                return "Neutral"

        df['sentiment'] = df['article'].apply(get_sentiment)
        limited_data = df.tail(limit).iloc[::-1]
        return jsonify(limited_data.to_dict(orient="records"))

api.add_resource(News, "/news/<string:type_>/<int:limit>")

# Other classes like BotStatistics, BotFeeder, etc. remain unchanged
# ----------------------------

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
