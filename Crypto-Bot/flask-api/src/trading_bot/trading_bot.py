"""
Trading bot with PPO Actor-Critic RL model integration
VERSION CORRIGÉE - Corrections des chemins et synchronisation des features
"""
import websocket
import json
import os
from pathlib import Path
from dotenv import dotenv_values
from csv import writer
import pandas as pd
import smtplib
import numpy as np
import requests
import time

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

print(f"TensorFlow version: {tf.__version__}")

# Load environment variables
env_path = r"C:\BC\Crypto-Bot\flask-api\.env"
if os.path.exists(env_path):
    env = dotenv_values(env_path)
    print(f"✓ Environment file loaded: {env_path}")
    API_KEY = env.get('API_KEY', '')
    API_SECRET = env.get('API_SECRET', '')
    PERSONAL_EMAIL = env.get('PERSONAL_EMAIL', '')
    DEV_EMAIL = env.get('DEV_EMAIL', '')
    EMAIL_PASS = env.get('EMAIL_PASS', '').strip('"')
    TAAPI = env.get('TAAPI', '')
else:
    print(f"✗ Environment file not found: {env_path}")
    raise FileNotFoundError("Environment file not found")

# Model configuration
MODEL_PATH = r"C:\BC\Reinforcement_Learning\reinforcement_learning_trading_agent\2025_12_30_12_22_Crypto_trader"

# Output data paths - CORRECTION DES CHEMINS
OUTPUT_DIR = Path(r"C:\BC\Crypto-Bot\flask-api\app\output_data")
TRANSACTION_HISTORY_PATH = OUTPUT_DIR / "transaction_history.csv"

# Créer le dossier si nécessaire
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialiser le fichier transaction_history.csv s'il n'existe pas
if not TRANSACTION_HISTORY_PATH.exists():
    df = pd.DataFrame(columns=[
        'order_id', 'symbol', 'order_status', 'quantity', 
        'timestamp', 'side', 'price', 'total', 'fee', 
        'net_profit', 'buy_price', 'sell_price'
    ])
    df.to_csv(TRANSACTION_HISTORY_PATH, index=False)
    print(f"✓ Created transaction history file: {TRANSACTION_HISTORY_PATH}")

# Training data normalization parameters
TRAINING_CSV_PATH = r"C:\BC\Reinforcement_Learning\reinforcement_learning_trading_agent\cryptoanalysis_data.csv"

def load_normalization_params():
    """
    Charge les paramètres de normalisation depuis le CSV d'entraînement.
    """
    try:
        df = pd.read_csv(TRAINING_CSV_PATH)
        df = df.rename(columns={'price': 'Close', 'date': 'Date'})
        
        # Retire Close et Date comme dans main.py
        df_for_norm = df.drop(['Close', 'Date'], axis=1)
        
        # Calcule min/max comme dans main.py
        column_maxes = df_for_norm.max()
        df_max = column_maxes.max()
        column_mins = df_for_norm.min()
        df_min = column_mins.min()
        
        print(f"✓ Normalization parameters loaded:")
        print(f"   Min: {df_min}")
        print(f"   Max: {df_max}")
        
        return df_min, df_max
    except Exception as e:
        print(f"⚠ Could not load normalization params: {e}")
        return None, None

DF_MIN, DF_MAX = load_normalization_params()

# Find the best model
def find_best_model(path):
    """Find the model with highest reward score"""
    files = os.listdir(path)
    actor_files = [f for f in files if f.endswith('_Crypto_trader_Actor.weights.h5')]
    
    if not actor_files:
        return None, None
    
    # Cherche le modèle avec le meilleur score
    best_score = -float('inf')
    best_file = None
    
    for f in actor_files:
        try:
            # Extrait le score du nom de fichier
            score_str = f.split('_')[0]
            score = float(score_str)
            if score > best_score:
                best_score = score
                best_file = f
        except:
            # Si pas de score, utilise le fichier par défaut
            if f.startswith('_'):
                best_file = f
    
    if best_file:
        actor_file = best_file
        critic_file = best_file.replace('_Actor.weights.h5', '_Critic.weights.h5')
        print(f"✓ Best model found with score: {best_score}")
        return actor_file, critic_file
    
    return None, None

ACTOR_FILE, CRITIC_FILE = find_best_model(MODEL_PATH)
print(f"Found Actor model: {ACTOR_FILE}")
print(f"Found Critic model: {CRITIC_FILE}")

# Trading configuration
SOCKET = "wss://stream.binance.com:9443/ws/ethusdt@kline_1m"
TRADE_SYMBOL = 'ETHUSDT'
TRADE_QUANTITY = 0.05

# Model parameters
LOOKBACK_WINDOW_SIZE = 50
ACTION_SPACE = 3  # Hold, Buy, Sell
NUM_FEATURES = 20  # 20 features from cryptoanalysis_data.csv

# Les 20 features EXACTES (vérifiez l'ordre dans votre CSV d'entraînement!)
FEATURE_COLUMNS = [
    'receive_count',
    'sent_count',
    'avg_fee',
    'blocksize',
    'btchashrate',
    'OIL',
    'ecr20_transfers',
    'GOLD',
    'searches',
    'hashrate',
    'marketcap',
    'difficulty',
    's&p500',
    'transactionfee',
    'transactions',
    'tweet_count',
    'unique_adresses',
    'VIX',
    'UVYX',
    'Close'
]

# State management
recent_candles = []
MAX_CANDLES = 200

# Global model variable
actor_model = None

# Cache pour les données externes
external_data_cache = {
    'last_update': 0,
    'data': {}
}

# Mode de trading
SIMULATION_MODE = True  # Mettre à False pour activer le trading réel

def create_actor_model(input_shape, action_space, lr=0.00001):
    """
    Recrée l'architecture du modèle Actor.
    """
    X_input = Input(input_shape)
    
    # Shared CNN layers
    X = Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(X_input)
    X = MaxPooling1D(pool_size=2)(X)
    X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(X)
    X = MaxPooling1D(pool_size=2)(X)
    X = Flatten()(X)
    
    # Actor layers
    A = Dense(512, activation="relu")(X)
    A = Dense(256, activation="relu")(A)
    A = Dense(64, activation="relu")(A)
    output = Dense(action_space, activation="softmax")(A)
    
    model = Model(inputs=X_input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr))
    
    return model

def load_actor_model():
    """Load the trained Actor model weights"""
    global actor_model
    
    print("\n" + "="*60)
    print("LOADING PPO ACTOR MODEL")
    print("="*60)
    
    try:
        if not ACTOR_FILE:
            print("✗ No model file found")
            return False
        
        actor_path = os.path.join(MODEL_PATH, ACTOR_FILE)
        
        if not os.path.exists(actor_path):
            print(f"✗ Model file not found: {actor_path}")
            return False
        
        print(f"✓ Found model: {actor_path}")
        
        input_shape = (LOOKBACK_WINDOW_SIZE, NUM_FEATURES)
        print(f"Input shape: {input_shape}")
        
        print("Creating Actor model architecture...")
        actor_model = create_actor_model(input_shape, ACTION_SPACE)
        
        print("Loading weights...")
        actor_model.load_weights(actor_path)
        
        print("✓ Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def send_mail(content):
    """Send notification email."""
    try:
        mail = smtplib.SMTP('smtp.gmail.com', 587)
        mail.ehlo()
        mail.starttls()
        mail.login(DEV_EMAIL, EMAIL_PASS)
        mail.sendmail(DEV_EMAIL, PERSONAL_EMAIL, content)
        mail.close()
        print(f"✓ Email sent: {content}")
    except Exception as e:
        print(f"✗ Failed to send email: {e}")

def get_external_data():
    """
    Récupère les données externes avec cache d'1 heure.
    """
    global external_data_cache
    
    current_time = time.time()
    
    # Si cache encore valide
    if current_time - external_data_cache['last_update'] < 3600:
        return external_data_cache['data']
    
    print("🌐 Fetching external market data...")
    
    try:
        # PLACEHOLDER - Remplacez par vos vraies API calls
        external_data = {
            'OIL': 75.0,
            'GOLD': 2000.0,
            's&p500': 4500.0,
            'VIX': 15.0,
            'UVYX': 10.0,
            'btchashrate': 400e18
        }
        
        external_data_cache['data'] = external_data
        external_data_cache['last_update'] = current_time
        
        print(f"✓ External data updated")
        return external_data
        
    except Exception as e:
        print(f"⚠ Error fetching external data: {e}")
        if external_data_cache['data']:
            return external_data_cache['data']
        else:
            return {
                'OIL': 75.0,
                'GOLD': 2000.0,
                's&p500': 4500.0,
                'VIX': 15.0,
                'UVYX': 10.0,
                'btchashrate': 400e18
            }

def prepare_state(candles_data):
    """
    Prépare l'état pour le modèle avec les 20 features.
    CRITIQUE: L'ordre ET les valeurs doivent correspondre à l'entraînement!
    """
    if len(candles_data) < LOOKBACK_WINDOW_SIZE:
        return None
    
    recent = candles_data[-LOOKBACK_WINDOW_SIZE:]
    df = pd.DataFrame(recent)
    
    # Convertit en float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    external_data = get_external_data()
    
    # Crée les features dans l'ORDRE EXACT du CSV d'entraînement
    features_list = []
    
    for idx, row in df.iterrows():
        # ⚠️ ATTENTION: Cet ordre DOIT correspondre EXACTEMENT à votre CSV
        # Vérifiez les colonnes de cryptoanalysis_data.csv!
        features = [
            external_data.get('receive_count', 1000),      # 0: receive_count
            external_data.get('sent_count', 1000),         # 1: sent_count
            float(row['volume']) * 0.001,                   # 2: avg_fee
            1000000,                                        # 3: blocksize
            external_data.get('btchashrate', 400e18),      # 4: btchashrate
            external_data.get('OIL', 75.0),                # 5: OIL
            500,                                            # 6: ecr20_transfers
            external_data.get('GOLD', 2000.0),             # 7: GOLD
            100,                                            # 8: searches
            300e12,                                         # 9: hashrate
            150e9,                                          # 10: marketcap
            5e12,                                           # 11: difficulty
            external_data.get('s&p500', 4500.0),           # 12: s&p500
            float(row['volume']) * 0.0001,                 # 13: transactionfee
            1000,                                           # 14: transactions
            50,                                             # 15: tweet_count
            50000,                                          # 16: unique_adresses
            external_data.get('VIX', 15.0),                # 17: VIX
            external_data.get('UVYX', 10.0),               # 18: UVYX
            float(row['close'])                             # 19: Close
        ]
        
        features_list.append(features)
    
    state = np.array(features_list)
    
    if state.shape[1] != NUM_FEATURES:
        print(f"⚠ WARNING: Feature mismatch! Expected {NUM_FEATURES}, got {state.shape[1]}")
        return None
    
    # Normalisation avec les paramètres d'entraînement
    if DF_MIN is not None and DF_MAX is not None:
        state = (state - DF_MIN) / (DF_MAX - DF_MIN)
    else:
        # Fallback
        state_min = state.min()
        state_max = state.max()
        if state_max - state_min > 0:
            state = (state - state_min) / (state_max - state_min)
    
    # Reshape: (1, lookback_window, features)
    state = state.reshape(1, LOOKBACK_WINDOW_SIZE, NUM_FEATURES)
    
    return state

def get_model_action(state):
    """
    Obtient l'action du modèle avec seuil de confiance.
    """
    if actor_model is None or state is None:
        return 0
    
    try:
        prediction = actor_model.predict(state, verbose=0)
        confidence = prediction[0]
        action = np.argmax(confidence)
        
        print(f"\n🤖 Model Prediction:")
        print(f"   Hold:  {confidence[0]:.2%}")
        print(f"   Buy:   {confidence[1]:.2%}")
        print(f"   Sell:  {confidence[2]:.2%}")
        
        # Seuil de confiance minimum (ajustez selon vos besoins)
        min_confidence = 0.4
        if confidence[action] < min_confidence:
            print(f"   ⚠️ Confidence too low ({confidence[action]:.2%}), defaulting to HOLD")
            action = 0
        else:
            print(f"   → Action: {['HOLD', 'BUY', 'SELL'][action]} (confidence: {confidence[action]:.2%})")
        
        return action
        
    except Exception as e:
        print(f"✗ Error getting model prediction: {e}")
        return 0

def append_list_as_row(file_name, list_of_elem):
    """Append a list as a row to a CSV file."""
    with open(file_name, 'a+', newline='', encoding='utf-8') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

def order(side, quantity, symbol, order_type='MARKET'):
    """Place an order - simulation ou réel selon SIMULATION_MODE."""
    timestamp = int(time.time() * 1000)
    
    if SIMULATION_MODE:
        print(f"\n💰 ORDER SIMULATION: {side} {quantity} {symbol}")
        
        order_data = [
            f"sim_{timestamp}",
            symbol,
            0,
            quantity,
            timestamp,
            side,
            0,
            0,
            0.075,
            0,
            '---',
            '---'
        ]
        
        try:
            append_list_as_row(TRANSACTION_HISTORY_PATH, order_data)
            print(f"   ✓ Order logged to {TRANSACTION_HISTORY_PATH}")
        except Exception as e:
            print(f"   ✗ Failed to log order: {e}")
        
        return True
    
    else:
        # TRADING RÉEL - Utiliser l'API Binance
        print(f"\n💰 REAL ORDER: {side} {quantity} {symbol}")
        print("   ⚠️ REAL TRADING NOT IMPLEMENTED YET")
        # TODO: Implémenter l'API Binance
        return False

def on_open(ws):
    """Called when WebSocket is opened."""
    print("\n" + "="*60)
    print("WEBSOCKET CONNECTED")
    print("="*60)
    mode = "SIMULATION" if SIMULATION_MODE else "LIVE TRADING"
    send_mail(f'PPO RL trading bot is online ({mode})')
    print(f'✓ PPO RL bot is online - Mode: {mode}')

def on_close(ws, close_status_code, close_msg):
    """Called when WebSocket is closed."""
    print("\n" + "="*60)
    print("WEBSOCKET CLOSED")
    print("="*60)
    send_mail('PPO RL bot stopped.')
    print(f'✗ Connection closed: {close_status_code} - {close_msg}')

def on_error(ws, error):
    """Called when WebSocket encounters an error."""
    print(f"\n✗ WebSocket Error: {error}")

def on_message(ws, message):
    """
    Traite les messages WebSocket et prend des décisions de trading.
    """
    global recent_candles
    
    try:
        json_message = json.loads(message)
        candle = json_message['k']
        is_candle_closed = candle['x']
        
        candle_data = {
            'open': candle['o'],
            'high': candle['h'],
            'low': candle['l'],
            'close': candle['c'],
            'volume': candle['v'],
            'timestamp': candle['t']
        }
        
        # Affichage en temps réel
        print(f"\r💹 Live: Close={candle_data['close']} | "
              f"Vol={float(candle_data['volume']):.2f} | "
              f"Candles={len(recent_candles)}/{LOOKBACK_WINDOW_SIZE}", end='')
        
        if is_candle_closed:
            print()
            print("\n" + "-"*60)
            print(f"📊 CANDLE CLOSED at {candle_data['close']}")
            print("-"*60)
            
            recent_candles.append(candle_data)
            
            if len(recent_candles) > MAX_CANDLES:
                recent_candles.pop(0)
            
            if len(recent_candles) < LOOKBACK_WINDOW_SIZE:
                print(f"⏳ Collecting data... ({len(recent_candles)}/{LOOKBACK_WINDOW_SIZE} candles)")
                return
            
            # Prépare l'état
            state = prepare_state(recent_candles)
            
            if state is not None:
                print(f"✓ State prepared: shape={state.shape}")
                
                # Obtient l'action du modèle
                action = get_model_action(state)
                
                # Lit la dernière position
                try:
                    df = pd.read_csv(TRANSACTION_HISTORY_PATH)
                    if len(df) > 0:
                        last_side = df.iloc[-1]['side']
                    else:
                        last_side = "SELL"
                except Exception as e:
                    print(f"⚠ Could not read transaction history: {e}")
                    last_side = "SELL"
                
                print(f"📍 Current Position: {last_side}")
                
                # Exécute les trades
                if action == 2 and last_side == "BUY":  # SELL
                    print("\n🔴 EXECUTING: SELL")
                    order_succeeded = order('SELL', TRADE_QUANTITY, TRADE_SYMBOL)
                    if order_succeeded:
                        send_mail(f'PPO bot SOLD at {candle_data["close"]}')
                    
                elif action == 1 and last_side == "SELL":  # BUY
                    print("\n🟢 EXECUTING: BUY")
                    order_succeeded = order('BUY', TRADE_QUANTITY, TRADE_SYMBOL)
                    if order_succeeded:
                        send_mail(f'PPO bot BOUGHT at {candle_data["close"]}')
                
                else:  # HOLD
                    print("\n⚪ ACTION: HOLD")
            else:
                print("✗ Failed to prepare state")
            
    except Exception as e:
        print(f"\n✗ Error in on_message: {e}")
        import traceback
        traceback.print_exc()

def bot():
    """Run the PPO RL trading bot."""
    print("\n" + "="*60)
    print("STARTING PPO RL TRADING BOT")
    print("="*60)
    print(f"Model path: {MODEL_PATH}")
    print(f"Actor model: {ACTOR_FILE}")
    print(f"WebSocket: {SOCKET}")
    print(f"Trade symbol: {TRADE_SYMBOL}")
    print(f"Lookback window: {LOOKBACK_WINDOW_SIZE}")
    print(f"Number of features: {NUM_FEATURES}")
    print(f"Transaction history: {TRANSACTION_HISTORY_PATH}")
    print(f"Mode: {'SIMULATION' if SIMULATION_MODE else 'LIVE TRADING'}")
    
    # ⚠️ AVERTISSEMENT SUR LES PERFORMANCES
    print("\n" + "⚠️ "*30)
    print("WARNING: Your model has NEGATIVE reward (-215.69)")
    print("This means the model performed WORSE than random!")
    print("STRONGLY RECOMMEND:")
    print("1. Train longer (more episodes)")
    print("2. Adjust hyperparameters")
    print("3. Verify feature engineering")
    print("4. Test in simulation before live trading")
    print("⚠️ "*30 + "\n")
    
    # Charge le modèle
    model_loaded = load_actor_model()
    
    if not model_loaded:
        print("\n✗ Model loading failed!")
        return
    
    print("\n" + "="*60)
    print("CONNECTING TO BINANCE WEBSOCKET")
    print("="*60)
    
    # Crée le WebSocket
    ws = websocket.WebSocketApp(
        SOCKET,
        on_open=on_open,
        on_close=on_close,
        on_message=on_message,
        on_error=on_error
    )
    
    ws.run_forever()

if __name__ == "__main__":
    bot()