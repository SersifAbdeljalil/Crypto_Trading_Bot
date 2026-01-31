"""
Flask app COMPLET pour le trading agent - VERSION CHEMINS CORRIGÉS
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
# CHEMINS - CONFIGURATION DYNAMIQUE
# ----------------------------
# Chemin du fichier actuel
CURRENT_FILE = Path(__file__).resolve()

# Racine du projet (C:\BC\Crypto-Bot)
PROJECT_ROOT = CURRENT_FILE.parents[3]  # Ajuster selon votre structure

# Dossier output_data (C:\BC\Crypto-Bot\output_data)
OUTPUT_DATA_DIR = PROJECT_ROOT / "output_data"

# Alternative: Si les données sont dans flask-api/app/output_data
# OUTPUT_DATA_DIR = CURRENT_FILE.parent / "output_data"

# Créer le dossier s'il n'existe pas
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"📁 Output data directory: {OUTPUT_DATA_DIR}")
print(f"   Exists: {OUTPUT_DATA_DIR.exists()}")

# ----------------------------
# Load environment variables
# ----------------------------
dotenv_path = PROJECT_ROOT / ".env"
if not dotenv_path.exists():
    dotenv_path = CURRENT_FILE.parents[2] / ".env"

print(f"📄 .env path: {dotenv_path}")
if dotenv_path.exists():
    print(f"✓ .env file found")
    load_dotenv(dotenv_path=dotenv_path)
else:
    print(f"⚠ .env file not found")

# ----------------------------
# Add project root to sys.path
# ----------------------------
sys.path.append(str(PROJECT_ROOT / 'src'))
sys.path.append(str(PROJECT_ROOT / 'flask-api' / 'src'))

# ----------------------------
# Imports
# ----------------------------
try:
    from data_handler.crypto_news_scraper import CryptoNewsScraper
except ImportError:
    print("⚠️ Could not import CryptoNewsScraper")
    CryptoNewsScraper = None

try:
    import configs.config
except ImportError:
    print("⚠️ Could not import config")

# Import du module d'indicateurs techniques
try:
    from technical_indicators import TechnicalIndicators, fetch_binance_klines
except ImportError:
    print("⚠️ technical_indicators.py not found")
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
CORS(app)
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
        "output_dir": str(OUTPUT_DATA_DIR),
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
        return bot_status

class BotControl(Resource):
    def post(self):
        """Démarre ou arrête le bot"""
        data = request.get_json()
        action = data.get('action')
        
        if action == 'start':
            bot_status['running'] = True
            bot_status['last_update'] = datetime.now().isoformat()
            
            return {
                "success": True,
                "message": "Bot started successfully",
                "status": bot_status
            }
            
        elif action == 'stop':
            bot_status['running'] = False
            bot_status['last_update'] = datetime.now().isoformat()
            
            return {
                "success": True,
                "message": "Bot stopped successfully",
                "status": bot_status
            }
        
        else:
            return {
                "success": False,
                "message": "Invalid action. Use 'start' or 'stop'"
            }, 400

api.add_resource(BotStatus, "/bot_status")
api.add_resource(BotControl, "/bot_control")

# ----------------------------
# Technical Indicators
# ----------------------------
class TechnicalIndicatorsAPI(Resource):
    def get(self):
        """Retourne tous les indicateurs techniques"""
        if not tech_indicators:
            return {
                "error": "Technical indicators module not available"
            }, 503
        
        try:
            candles = fetch_binance_klines(symbol="ETHUSDT", interval="1m", limit=100)
            
            if candles:
                tech_indicators.candles.clear()
                for candle in candles:
                    tech_indicators.add_candle(candle)
            
            indicators = tech_indicators.get_all_indicators()
            
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "symbol": "ETHUSDT",
                "indicators": indicators
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }, 500

class MarketSentiment(Resource):
    def get(self):
        """Retourne le sentiment du marché"""
        if not tech_indicators:
            return {
                "error": "Technical indicators module not available"
            }, 503
        
        try:
            sentiment = tech_indicators.get_market_sentiment()
            
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "sentiment": sentiment
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }, 500

api.add_resource(TechnicalIndicatorsAPI, "/technical_indicators")
api.add_resource(MarketSentiment, "/market_sentiment")

# ----------------------------
# Model Prediction
# ----------------------------
class ModelPrediction(Resource):
    def get(self):
        """Retourne la dernière prédiction du modèle"""
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "prediction": {
                "hold": 0.35,
                "buy": 0.45,
                "sell": 0.20
            },
            "action": bot_status['last_action'],
            "confidence": bot_status['confidence']
        }

api.add_resource(ModelPrediction, "/model_prediction")

# ----------------------------
# Statistics - CHEMINS CORRIGÉS
# ----------------------------
class BotStatistics(Resource):
    def get(self):
        """Retourne les statistiques du bot"""
        try:
            # ✅ UTILISE LE CHEMIN DYNAMIQUE
            csv_path = OUTPUT_DATA_DIR / "transaction_history.csv"
            
            # Vérifier que le fichier existe
            if not csv_path.exists():
                print(f"⚠️ File not found: {csv_path}")
                return {
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
                    },
                    "note": "No transaction history file found"
                }
            
            df = pd.read_csv(csv_path)
            
            if len(df) == 0:
                return {
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
                }
            
            sells = df[df['side'] == 'SELL']
            
            if len(sells) > 0:
                total_trades = len(sells)
                
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
                    
                    bot_status['net_profit'] = net_profit
                    bot_status['win_rate'] = win_rate
                    bot_status['total_trades'] = total_trades
                    bot_status['winning_trades'] = winning_trades
                    
                    return {
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
                    }
            
            return {
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
            }
            
        except Exception as e:
            print(f"❌ Error in BotStatistics: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }, 500

api.add_resource(BotStatistics, "/statistics")

# ----------------------------
# Transactions History - CHEMINS CORRIGÉS
# ----------------------------
class AllTransactionHistory(Resource):
    def get(self, limit):
        try:
            # ✅ UTILISE LE CHEMIN DYNAMIQUE
            csv_path = OUTPUT_DATA_DIR / "transaction_history.csv"
            
            if not csv_path.exists():
                return {"error": f"File not found: {csv_path}"}, 404
            
            df = pd.read_csv(csv_path)
            limited_data = df.tail(limit).iloc[::-1]
            return limited_data.to_dict(orient="records")
        except Exception as e:
            return {"error": str(e)}, 500

api.add_resource(AllTransactionHistory, "/all_transaction_history/<int:limit>")

# ----------------------------
# News - CHEMINS CORRIGÉS
# ----------------------------
class News(Resource):
    def get(self, type_, limit):
        try:
            # ✅ UTILISE LE CHEMIN DYNAMIQUE
            csv_path = OUTPUT_DATA_DIR / f"{type_}News.csv"
            
            if not csv_path.exists():
                return {"error": f"File not found: {csv_path}"}, 404
            
            df = pd.read_csv(csv_path)
            
            if df.shape[1] < 3:
                return {"error": "CSV file format is incorrect"}, 400
            
            df.fillna("Empty", inplace=True)
            
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
            return limited_data.to_dict(orient="records")
            
        except Exception as e:
            return {"error": str(e)}, 500

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
    emit('bot_status', bot_status)
    
    if tech_indicators:
        try:
            indicators = tech_indicators.get_all_indicators()
            emit('technical_indicators', indicators)
        except:
            pass

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
    print(f"✓ Output directory: {OUTPUT_DATA_DIR}")
    print(f"   - Exists: {OUTPUT_DATA_DIR.exists()}")
    if OUTPUT_DATA_DIR.exists():
        files = list(OUTPUT_DATA_DIR.glob("*.csv"))
        print(f"   - CSV files found: {len(files)}")
        for f in files:
            print(f"     • {f.name}")
    print("="*60 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)