"""
Flask app COMPLET pour le trading agent
Backend avec tous les endpoints + WebSocket + Indicateurs techniques
"""
from pathlib import Path
import sys
import os
import subprocess
import pandas as pd
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from csv import DictWriter, writer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from dotenv import load_dotenv
import json
from datetime import datetime
import threading
import time

# ----------------------------
# Load environment variables
# ----------------------------
dotenv_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=dotenv_path)

# ----------------------------
# Add project root to sys.path
# ----------------------------
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root / 'src'))

# ----------------------------
# Imports
# ----------------------------
from data_handler.crypto_news_scraper import CryptoNewsScraper
# On importe NOT trading_bot.bot pour éviter conflit
# from trading_bot.trading_bot import bot
import configs.config

# Import du module d'indicateurs techniques
sys.path.append(str(project_root / 'src' / 'data_handler'))
try:
    from technical_indicators import TechnicalIndicators, fetch_binance_klines
except ImportError:
    print("⚠️ technical_indicators.py not found. Creating placeholder...")
    TechnicalIndicators = None
    fetch_binance_klines = None

# Download NLTK lexicon
try:
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# ----------------------------
# Initialize Flask app
# ----------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "default_secret_key")
api = Api(app)
CORS(app)  # Enable CORS
socketio = SocketIO(app, cors_allowed_origins="*")

# ----------------------------
# Global variables
# ----------------------------
bot_status = {
    "running": False,
    "last_action": "HOLD",
    "confidence": 0.0,
    "net_profit": 0.0,
    "win_rate": 0.0,
    "total_trades": 0,
    "winning_trades": 0,
    "last_update": None
}

# Indicateurs techniques globaux
tech_indicators = None
if TechnicalIndicators:
    tech_indicators = TechnicalIndicators(symbol="ETHUSDT", interval="1m", lookback=100)

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "message": "Trading Bot Flask API v2.0",
        "endpoints": {
            "bot_status": "/bot_status",
            "bot_control": "/bot_control (POST)",
            "technical_indicators": "/technical_indicators",
            "market_sentiment": "/market_sentiment",
            "model_prediction": "/model_prediction",
            "statistics": "/statistics",
            "transactions": "/all_transaction_history/<limit>",
            "news": "/news/<type>/<limit>"
        }
    })

# ----------------------------
# Bot Status & Control
# ----------------------------
class BotStatus(Resource):
    def get(self):
        """Retourne l'état actuel du bot"""
        return jsonify(bot_status)

class BotControl(Resource):
    def post(self):
        """Démarre ou arrête le bot"""
        data = request.get_json()
        action = data.get('action')  # 'start' ou 'stop'
        
        if action == 'start':
            bot_status['running'] = True
            bot_status['last_update'] = datetime.now().isoformat()
            
            # Démarrer le bot dans un thread séparé
            # threading.Thread(target=run_trading_bot, daemon=True).start()
            
            return jsonify({
                "success": True,
                "message": "Bot started successfully",
                "status": bot_status
            })
            
        elif action == 'stop':
            bot_status['running'] = False
            bot_status['last_update'] = datetime.now().isoformat()
            
            return jsonify({
                "success": True,
                "message": "Bot stopped successfully",
                "status": bot_status
            })
        
        else:
            return jsonify({
                "success": False,
                "message": "Invalid action. Use 'start' or 'stop'"
            }), 400

api.add_resource(BotStatus, "/bot_status")
api.add_resource(BotControl, "/bot_control")

# ----------------------------
# Technical Indicators
# ----------------------------
class TechnicalIndicatorsAPI(Resource):
    def get(self):
        """Retourne tous les indicateurs techniques"""
        if not tech_indicators:
            return jsonify({
                "error": "Technical indicators module not available"
            }), 503
        
        try:
            # Mettre à jour avec les dernières données
            candles = fetch_binance_klines(symbol="ETHUSDT", interval="1m", limit=100)
            
            if candles:
                # Vider et remplir avec les nouvelles données
                tech_indicators.candles.clear()
                for candle in candles:
                    tech_indicators.add_candle(candle)
            
            # Calculer les indicateurs
            indicators = tech_indicators.get_all_indicators()
            
            return jsonify({
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "symbol": "ETHUSDT",
                "indicators": indicators
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

class MarketSentiment(Resource):
    def get(self):
        """Retourne le sentiment du marché"""
        if not tech_indicators:
            return jsonify({
                "error": "Technical indicators module not available"
            }), 503
        
        try:
            sentiment = tech_indicators.get_market_sentiment()
            
            return jsonify({
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "sentiment": sentiment
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

api.add_resource(TechnicalIndicatorsAPI, "/technical_indicators")
api.add_resource(MarketSentiment, "/market_sentiment")

# ----------------------------
# Model Prediction
# ----------------------------
class ModelPrediction(Resource):
    def get(self):
        """Retourne la dernière prédiction du modèle"""
        # TODO: Intégrer avec le vrai modèle
        # Pour l'instant, données mockées
        
        return jsonify({
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "prediction": {
                "hold": 0.35,
                "buy": 0.45,
                "sell": 0.20
            },
            "action": bot_status['last_action'],
            "confidence": bot_status['confidence']
        })

api.add_resource(ModelPrediction, "/model_prediction")

# ----------------------------
# Statistics
# ----------------------------
class BotStatistics(Resource):
    def get(self):
        """Retourne les statistiques du bot"""
        try:
            # Lire l'historique des transactions
            df = pd.read_csv("app/output_data/transaction_history.csv")
            
            if len(df) == 0:
                return jsonify({
                    "success": True,
                    "statistics": {
                        "net_profit": 0.0,
                        "win_rate": 0.0,
                        "total_trades": 0,
                        "winning_trades": 0,
                        "losing_trades": 0,
                        "average_profit": 0.0,
                        "average_loss": 0.0,
                        "profit_factor": 0.0
                    }
                })
            
            # Calculer les statistiques
            sells = df[df['side'] == 'SELL']
            
            if len(sells) > 0:
                total_trades = len(sells)
                
                # Profits
                profits = []
                for idx, row in sells.iterrows():
                    if row['profits'] != '---' and pd.notna(row['profits']):
                        try:
                            profit = float(row['profits'])
                            profits.append(profit)
                        except:
                            pass
                
                if profits:
                    winning_trades = len([p for p in profits if p > 0])
                    losing_trades = len([p for p in profits if p < 0])
                    net_profit = sum(profits)
                    win_rate = (winning_trades / total_trades) * 100
                    
                    winning_profits = [p for p in profits if p > 0]
                    losing_profits = [p for p in profits if p < 0]
                    
                    avg_profit = sum(winning_profits) / len(winning_profits) if winning_profits else 0
                    avg_loss = sum(losing_profits) / len(losing_profits) if losing_profits else 0
                    
                    profit_factor = abs(sum(winning_profits) / sum(losing_profits)) if losing_profits else 0
                    
                    # Mettre à jour le statut global
                    bot_status['net_profit'] = net_profit
                    bot_status['win_rate'] = win_rate
                    bot_status['total_trades'] = total_trades
                    bot_status['winning_trades'] = winning_trades
                    
                    return jsonify({
                        "success": True,
                        "statistics": {
                            "net_profit": round(net_profit, 2),
                            "win_rate": round(win_rate, 2),
                            "total_trades": total_trades,
                            "winning_trades": winning_trades,
                            "losing_trades": losing_trades,
                            "average_profit": round(avg_profit, 2),
                            "average_loss": round(avg_loss, 2),
                            "profit_factor": round(profit_factor, 2)
                        }
                    })
            
            return jsonify({
                "success": True,
                "statistics": {
                    "net_profit": 0.0,
                    "win_rate": 0.0,
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "average_profit": 0.0,
                    "average_loss": 0.0,
                    "profit_factor": 0.0
                }
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

api.add_resource(BotStatistics, "/statistics")

# ----------------------------
# Transactions History (existing)
# ----------------------------
class AllTransactionHistory(Resource):
    def get(self, limit):
        try:
            df = pd.read_csv("app/output_data/transaction_history.csv")
            limited_data = df.tail(limit).iloc[::-1]
            return jsonify(limited_data.to_dict(orient="records"))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

api.add_resource(AllTransactionHistory, "/all_transaction_history/<int:limit>")

# ----------------------------
# News (existing avec améliorations)
# ----------------------------
class News(Resource):
    def get(self, type_, limit):
        try:
            path = f'app/output_data/{type_}News.csv'
            df = pd.read_csv(path)
            
            if df.shape[1] < 3:
                return jsonify({"error": "CSV file format is incorrect"}), 400
            
            df.fillna("Empty", inplace=True)
            
            # Sentiment analysis
            sia = SentimentIntensityAnalyzer()
            
            def get_sentiment(article):
                if article == "Empty" or pd.isna(article):
                    return "Neutral"
                    
                score = sia.polarity_scores(str(article))
                if score['compound'] >= 0.05:
                    return "Positive"
                elif score['compound'] <= -0.05:
                    return "Negative"
                else:
                    return "Neutral"
            
            df['sentiment'] = df['article'].apply(get_sentiment)
            
            limited_data = df.tail(limit).iloc[::-1]
            return jsonify(limited_data.to_dict(orient="records"))
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

api.add_resource(News, "/news/<string:type_>/<int:limit>")

# ----------------------------
# WebSocket Events
# ----------------------------
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('bot_status', bot_status)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('request_update')
def handle_request_update():
    """Client demande une mise à jour"""
    emit('bot_status', bot_status)
    
    if tech_indicators:
        try:
            indicators = tech_indicators.get_all_indicators()
            emit('technical_indicators', indicators)
        except:
            pass

# ----------------------------
# Background Tasks
# ----------------------------
def broadcast_updates():
    """Envoie des mises à jour périodiques via WebSocket"""
    while True:
        time.sleep(5)  # Toutes les 5 secondes
        
        if tech_indicators:
            try:
                # Mettre à jour les indicateurs
                candles = fetch_binance_klines(symbol="ETHUSDT", interval="1m", limit=100)
                if candles:
                    tech_indicators.candles.clear()
                    for candle in candles[-100:]:
                        tech_indicators.add_candle(candle)
                    
                    indicators = tech_indicators.get_all_indicators()
                    socketio.emit('technical_indicators_update', indicators)
            except Exception as e:
                print(f"Error broadcasting updates: {e}")

# Démarrer le thread de broadcast
# threading.Thread(target=broadcast_updates, daemon=True).start()

# ----------------------------
# Start Background Scripts
# ----------------------------
def run_script(script_name):
    """Démarre un script Python en subprocess"""
    script_path = Path(__file__).parent / script_name
    if script_path.exists():
        print(f'Starting subprocess for {script_name}...')
        subprocess.Popen(
            ["python", str(script_path)], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        print(f'Subprocess for {script_name} started.')
    else:
        print(f'⚠️ Script not found: {script_path}')

# Démarrer les scrapers
# run_script("run_top_news_scraper.py")
# run_script("run_all_news_scraper.py")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 TRADING BOT API v2.0 - STARTING")
    print("="*60)
    print(f"✓ Flask app initialized")
    print(f"✓ CORS enabled")
    print(f"✓ WebSocket enabled")
    print(f"✓ Technical indicators: {'Available' if tech_indicators else 'Not available'}")
    print("="*60 + "\n")
    
    # Lancer avec WebSocket support
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)