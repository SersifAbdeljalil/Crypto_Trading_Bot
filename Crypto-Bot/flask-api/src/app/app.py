"""
Flask API - Trading Bot TEMPS REEL
Integre predictor_engine.py avec WebSocket pour le frontend
"""
from pathlib import Path
import sys
import os
import pandas as pd
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from csv import writer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import json
from datetime import datetime
import threading
import time
import logging

# ─────────────────────────────────────────────
#  CHEMINS
# ─────────────────────────────────────────────
CURRENT_FILE     = Path(__file__).resolve()
PROJECT_ROOT     = CURRENT_FILE.parents[3]          # C:\BC\Crypto-Bot
FLASK_ROOT       = CURRENT_FILE.parents[2]          # C:\BC\Crypto-Bot\flask-api
OUTPUT_DATA_DIR  = PROJECT_ROOT / "output_data"
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Chemin vers le projet RL
RL_PROJECT = Path(r"C:\BC\Reinforcement_Learning\reinforcement_learning_trading_agent")

print(f"Output dir : {OUTPUT_DATA_DIR}")
print(f"RL project : {RL_PROJECT}")

# ─────────────────────────────────────────────
#  ENV
# ─────────────────────────────────────────────
def load_env(env_path):
    if not Path(env_path).exists():
        return
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, val = line.partition('=')
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key.strip(), val)

load_env(PROJECT_ROOT / ".env")
load_env(FLASK_ROOT / ".env")

# ─────────────────────────────────────────────
#  PREDICTOR (depuis RL project)
# ─────────────────────────────────────────────
sys.path.insert(0, str(RL_PROJECT))
predictor_instance = None
predictor_thread   = None
predictor_running  = False

def init_predictor():
    global predictor_instance
    try:
        from predictor_engine import RealTimePredictor

        # Auto-detect best model
        import glob
        folders = sorted(glob.glob(str(RL_PROJECT / "*_Crypto_trader")))
        best_score, best_folder, best_score_str = -float('inf'), None, None

        for folder in folders:
            for wf in glob.glob(os.path.join(folder, "*_Actor.weights.h5")):
                fname     = os.path.basename(wf)
                score_str = fname.split("_Crypto_trader_Actor")[0]
                try:
                    score = float(score_str)
                    if score > best_score:
                        best_score, best_folder, best_score_str = score, folder, score_str
                except ValueError:
                    pass

        if not best_folder:
            print("No model found for predictor.")
            return False

        csv_path = str(RL_PROJECT / "cryptoanalysis_data.csv")
        predictor_instance = RealTimePredictor(
            model_folder   = best_folder,
            model_score    = best_score_str,
            historical_csv = csv_path if os.path.exists(csv_path) else None
        )
        predictor_instance.load_model()
        predictor_instance._warm_up_buffer()
        print(f"Predictor initialized: {os.path.basename(best_folder)} / {best_score_str}")
        return True

    except Exception as e:
        print(f"Predictor init failed: {e}")
        return False

# ─────────────────────────────────────────────
#  TECHNICAL INDICATORS
# ─────────────────────────────────────────────
try:
    from data_handler.technical_indicators import TechnicalIndicators, fetch_binance_klines
    tech_indicators = TechnicalIndicators(symbol="ETHUSDT", interval="1m", lookback=100)
    print("TechnicalIndicators loaded")
except ImportError:
    try:
        # Try local path
        sys.path.insert(0, str(FLASK_ROOT / "src" / "data_handler"))
        from technical_indicators import TechnicalIndicators, fetch_binance_klines
        tech_indicators = TechnicalIndicators(symbol="ETHUSDT", interval="1m", lookback=100)
        print("TechnicalIndicators loaded (local)")
    except ImportError:
        print("TechnicalIndicators not available")
        TechnicalIndicators      = None
        fetch_binance_klines     = None
        tech_indicators          = None

# NLTK
try:
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# ─────────────────────────────────────────────
#  FLASK
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "trading_secret")
api = Api(app)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ─────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────
bot_status = {
    "running":       False,
    "last_action":   "HOLD",
    "confidence":    0.0,
    "net_profit":    0.0,
    "win_rate":      0.0,
    "total_trades":  0,
    "winning_trades":0,
    "last_update":   None,
    "current_price": 0.0,
    "virtual_net_worth": 1000.0,
    "virtual_pnl":   0.0,
    "probabilities": {"HOLD": 33.3, "BUY": 33.3, "SELL": 33.3},
    "signal_strength": "normal",
}

# ─────────────────────────────────────────────
#  PREDICTION LOOP (background thread)
# ─────────────────────────────────────────────
def prediction_loop():
    global predictor_running
    print("Prediction loop started")

    while predictor_running:
        try:
            if predictor_instance is None:
                time.sleep(5)
                continue

            result = predictor_instance.predict_once()

            if result:
                # Update global bot_status
                bot_status["last_action"]      = result["action_label"]
                bot_status["confidence"]        = result["confidence"] / 100.0
                bot_status["current_price"]     = result["price"]
                bot_status["virtual_net_worth"] = result["virtual_net_worth"]
                bot_status["virtual_pnl"]       = result["virtual_pnl"]
                bot_status["probabilities"]     = result["probabilities"]
                bot_status["signal_strength"]   = result["signal_strength"]
                bot_status["last_update"]       = result["timestamp"]

                # Push to all connected WebSocket clients
                socketio.emit('prediction_update', {
                    "bot_status":  bot_status,
                    "prediction":  result,
                    "timestamp":   result["timestamp"],
                })

                # Alert on strong signal
                if result["is_strong_signal"]:
                    socketio.emit('strong_signal', {
                        "action":    result["action_label"],
                        "price":     result["price"],
                        "confidence":result["confidence"],
                    })

            time.sleep(60)   # 1 prediction per minute

        except Exception as e:
            print(f"Prediction loop error: {e}")
            time.sleep(30)

    print("Prediction loop stopped")


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({
        "status":    "online",
        "message":   "Trading Bot Flask API v3.0 - Real-Time Predictor",
        "predictor": "initialized" if predictor_instance else "not available",
        "endpoints": {
            "bot_status":           "/bot_status",
            "bot_control":          "/bot_control (POST)",
            "model_prediction":     "/model_prediction",
            "technical_indicators": "/technical_indicators",
            "market_sentiment":     "/market_sentiment",
            "statistics":           "/statistics",
            "transactions":         "/all_transaction_history/<limit>",
            "news":                 "/news/<type>/<limit>",
            "latest_prediction":    "/latest_prediction",
        }
    })


# ─── Bot Status ──────────────────────────────
class BotStatus(Resource):
    def get(self):
        return bot_status

class BotControl(Resource):
    def post(self):
        global predictor_thread, predictor_running

        data   = request.get_json()
        action = data.get('action')

        if action == 'start':
            if not bot_status['running']:
                # Init predictor if needed
                if predictor_instance is None:
                    init_predictor()

                bot_status['running']     = True
                bot_status['last_update'] = datetime.now().isoformat()
                predictor_running         = True

                predictor_thread = threading.Thread(
                    target=prediction_loop, daemon=True
                )
                predictor_thread.start()

            return {"success": True, "message": "Bot started", "status": bot_status}

        elif action == 'stop':
            bot_status['running'] = False
            predictor_running     = False
            bot_status['last_update'] = datetime.now().isoformat()
            return {"success": True, "message": "Bot stopped", "status": bot_status}

        return {"success": False, "message": "Invalid action. Use start or stop"}, 400

api.add_resource(BotStatus,  "/bot_status")
api.add_resource(BotControl, "/bot_control")


# ─── Latest Prediction ───────────────────────
class LatestPrediction(Resource):
    def get(self):
        if predictor_instance is None:
            return {
                "success":   False,
                "message":   "Predictor not initialized. Start the bot first.",
                "prediction": {
                    "action_label":  "HOLD",
                    "confidence":    0.0,
                    "probabilities": {"HOLD": 33.3, "BUY": 33.3, "SELL": 33.3},
                    "price":         0.0,
                    "virtual_net_worth": 1000.0,
                    "virtual_pnl":   0.0,
                }
            }

        latest = predictor_instance.get_latest_prediction()
        if not latest:
            return {
                "success": False,
                "message": "No prediction yet. Bot is warming up."
            }

        return {
            "success":    True,
            "timestamp":  latest["timestamp"],
            "prediction": latest,
        }

api.add_resource(LatestPrediction, "/latest_prediction")


# ─── Model Prediction (simple format for frontend) ─
class ModelPrediction(Resource):
    def get(self):
        probs = bot_status.get("probabilities", {"HOLD": 33.3, "BUY": 33.3, "SELL": 33.3})
        return {
            "success":    True,
            "timestamp":  datetime.now().isoformat(),
            "prediction": {
                "hold": probs.get("HOLD", 33.3) / 100,
                "buy":  probs.get("BUY",  33.3) / 100,
                "sell": probs.get("SELL", 33.3) / 100,
            },
            "action":     bot_status["last_action"],
            "confidence": bot_status["confidence"],
            "price":      bot_status["current_price"],
            "virtual_net_worth": bot_status["virtual_net_worth"],
            "virtual_pnl":       bot_status["virtual_pnl"],
            "signal_strength":   bot_status["signal_strength"],
        }

api.add_resource(ModelPrediction, "/model_prediction")


# ─── Technical Indicators ────────────────────
class TechnicalIndicatorsAPI(Resource):
    def get(self):
        if not tech_indicators:
            return {"error": "Technical indicators not available"}, 503
        try:
            if fetch_binance_klines:
                candles = fetch_binance_klines(symbol="ETHUSDT", interval="1m", limit=100)
                if candles:
                    tech_indicators.candles.clear()
                    for c in candles:
                        tech_indicators.add_candle(c)
            indicators = tech_indicators.get_all_indicators()
            return {
                "success":    True,
                "timestamp":  datetime.now().isoformat(),
                "symbol":     "ETHUSDT",
                "indicators": indicators,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}, 500

class MarketSentiment(Resource):
    def get(self):
        if not tech_indicators:
            return {"error": "Technical indicators not available"}, 503
        try:
            return {
                "success":   True,
                "timestamp": datetime.now().isoformat(),
                "sentiment": tech_indicators.get_market_sentiment(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}, 500

api.add_resource(TechnicalIndicatorsAPI, "/technical_indicators")
api.add_resource(MarketSentiment,        "/market_sentiment")


# ─── Statistics ──────────────────────────────
class BotStatistics(Resource):
    def get(self):
        try:
            csv_path = OUTPUT_DATA_DIR / "transaction_history.csv"
            if not csv_path.exists():
                return {
                    "success": True,
                    "statistics": {
                        "net_profit": 0.0, "win_rate": 0.0, "total_trades": 0,
                        "winning_trades": 0, "losing_trades": 0,
                        "average_profit": 0.0, "average_loss": 0.0, "profit_factor": 0.0
                    },
                    "note": "No transaction history yet"
                }

            df = pd.read_csv(csv_path)
            if len(df) == 0:
                return {"success": True, "statistics": {
                    "net_profit": 0.0, "win_rate": 0.0, "total_trades": 0,
                    "winning_trades": 0, "losing_trades": 0,
                    "average_profit": 0.0, "average_loss": 0.0, "profit_factor": 0.0
                }}

            sells = df[df['side'] == 'SELL']
            if len(sells) == 0:
                return {"success": True, "statistics": {
                    "net_profit": 0.0, "win_rate": 0.0, "total_trades": 0,
                    "winning_trades": 0, "losing_trades": 0,
                    "average_profit": 0.0, "average_loss": 0.0, "profit_factor": 0.0
                }}

            profits = []
            for _, row in sells.iterrows():
                if row['profits'] != '---' and pd.notna(row['profits']):
                    try:
                        profits.append(float(row['profits']))
                    except:
                        pass

            if not profits:
                return {"success": True, "statistics": {
                    "net_profit": 0.0, "win_rate": 0.0,
                    "total_trades": len(sells), "winning_trades": 0,
                    "losing_trades": 0, "average_profit": 0.0,
                    "average_loss": 0.0, "profit_factor": 0.0
                }}

            total_trades    = len(sells)
            winning_trades  = len([p for p in profits if p > 0])
            losing_trades   = len([p for p in profits if p < 0])
            net_profit      = sum(profits)
            win_rate        = (winning_trades / total_trades) * 100
            wins            = [p for p in profits if p > 0]
            losses          = [p for p in profits if p < 0]
            avg_profit      = sum(wins)   / len(wins)   if wins   else 0
            avg_loss        = sum(losses) / len(losses) if losses else 0
            profit_factor   = abs(sum(wins) / sum(losses)) if losses else 0

            bot_status.update({
                "net_profit":    net_profit,
                "win_rate":      win_rate,
                "total_trades":  total_trades,
                "winning_trades":winning_trades,
            })

            return {
                "success": True,
                "statistics": {
                    "net_profit":     round(net_profit, 2),
                    "win_rate":       round(win_rate, 2),
                    "total_trades":   total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades":  losing_trades,
                    "average_profit": round(avg_profit, 2),
                    "average_loss":   round(avg_loss, 2),
                    "profit_factor":  round(profit_factor, 2),
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}, 500

api.add_resource(BotStatistics, "/statistics")


# ─── Transactions ────────────────────────────
class AllTransactionHistory(Resource):
    def get(self, limit):
        try:
            csv_path = OUTPUT_DATA_DIR / "transaction_history.csv"
            if not csv_path.exists():
                return {"error": "No transaction history yet"}, 404
            df = pd.read_csv(csv_path)
            return df.tail(limit).iloc[::-1].to_dict(orient="records")
        except Exception as e:
            return {"error": str(e)}, 500

api.add_resource(AllTransactionHistory, "/all_transaction_history/<int:limit>")


# ─── News ────────────────────────────────────
class News(Resource):
    def get(self, type_, limit):
        try:
            csv_path = OUTPUT_DATA_DIR / f"{type_}News.csv"
            if not csv_path.exists():
                return {"error": f"File not found: {type_}News.csv"}, 404
            df = pd.read_csv(csv_path)
            if df.shape[1] < 3:
                return {"error": "CSV format incorrect"}, 400
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
                return "Neutral"

            df['sentiment'] = df['article'].apply(get_sentiment)
            return df.tail(limit).iloc[::-1].to_dict(orient="records")
        except Exception as e:
            return {"error": str(e)}, 500

api.add_resource(News, "/news/<string:type_>/<int:limit>")


# ─────────────────────────────────────────────
#  WEBSOCKET EVENTS
# ─────────────────────────────────────────────
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('bot_status', bot_status)
    if predictor_instance:
        latest = predictor_instance.get_latest_prediction()
        if latest:
            emit('prediction_update', {
                "bot_status": bot_status,
                "prediction": latest,
                "timestamp":  latest["timestamp"],
            })

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('request_update')
def handle_request_update():
    emit('bot_status', bot_status)
    if predictor_instance and tech_indicators:
        try:
            emit('technical_indicators', tech_indicators.get_all_indicators())
        except:
            pass

@socketio.on('force_prediction')
def handle_force_prediction():
    """Force an immediate prediction (useful for testing)."""
    if predictor_instance and bot_status['running']:
        try:
            result = predictor_instance.predict_once()
            if result:
                emit('prediction_update', {
                    "bot_status": bot_status,
                    "prediction": result,
                    "timestamp":  result["timestamp"],
                })
        except Exception as e:
            emit('error', {"message": str(e)})
    else:
        emit('error', {"message": "Bot not running or predictor not initialized"})


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TRADING BOT API v3.0 - REAL-TIME PREDICTOR")
    print("=" * 60)
    print(f"  Output dir  : {OUTPUT_DATA_DIR}")
    print(f"  RL project  : {RL_PROJECT}")

    # Pre-load predictor at startup
    print("\n  Loading predictor model...")
    init_predictor()
    print(f"  Predictor   : {'OK' if predictor_instance else 'FAILED'}")
    print(f"  Tech indic. : {'OK' if tech_indicators else 'Not available'}")
    print("=" * 60)
    print("  Endpoints:")
    print("    GET  /bot_status")
    print("    POST /bot_control   {action: start|stop}")
    print("    GET  /model_prediction")
    print("    GET  /latest_prediction")
    print("    GET  /technical_indicators")
    print("    GET  /statistics")
    print("    WS   socket.io events: prediction_update, strong_signal")
    print("=" * 60 + "\n")

    socketio.run(app, host='0.0.0.0', port=5000,
                 debug=False, allow_unsafe_werkzeug=True)