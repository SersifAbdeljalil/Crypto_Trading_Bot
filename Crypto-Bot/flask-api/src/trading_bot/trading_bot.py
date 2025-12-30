"""
Trading bot with PPO Actor-Critic RL model integration
VERSION CORRIGÉE - Utilise les 20 features exactes de cryptoanalysis_data.csv
"""
import websocket
import json
import os
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

# Training data normalization parameters (loaded from cryptoanalysis_data.csv)
# Ces valeurs DOIVENT correspondre à celles utilisées pendant l'entraînement
TRAINING_CSV_PATH = r"C:\BC\Reinforcement_Learning\reinforcement_learning_trading_agent\cryptoanalysis_data.csv"

def load_normalization_params():
    """
    Charge les paramètres de normalisation depuis le CSV d'entraînement.
    CRITIQUE: La normalisation doit être IDENTIQUE à l'entraînement!
    """
    try:
        df = pd.read_csv(TRAINING_CSV_PATH)
        df = df.rename(columns={'price': 'Close', 'date': 'Date'})
        
        # Retire Close et Date comme dans main.py
        df_for_norm = df.drop(['Close', 'Date'], axis=1)
        
        # Calcule min/max comme dans main.py (lignes 23-26)
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
        print("   Using default normalization per-window")
        return None, None

DF_MIN, DF_MAX = load_normalization_params()

# Find the best model (highest reward)
def find_best_model(path):
    """Find the model with highest reward score"""
    files = os.listdir(path)
    actor_files = [f for f in files if f.endswith('_Crypto_trader_Actor.weights.h5')]
    
    if not actor_files:
        return None, None
    
    best_score = -float('inf')
    best_file = None
    
    for f in actor_files:
        try:
            score = float(f.split('_')[0])
            if score > best_score:
                best_score = score
                best_file = f
        except:
            if f.startswith('_'):
                best_file = f
    
    if best_file:
        actor_file = best_file
        critic_file = best_file.replace('_Actor.weights.h5', '_Critic.weights.h5')
        return actor_file, critic_file
    
    return None, None

ACTOR_FILE, CRITIC_FILE = find_best_model(MODEL_PATH)
print(f"Found Actor model: {ACTOR_FILE}")
print(f"Found Critic model: {CRITIC_FILE}")

# Trading configuration
SOCKET = "wss://stream.binance.com:9443/ws/ethusdt@kline_1m"
TRADE_SYMBOL = 'ETHUSDT'
TRADE_QUANTITY = 0.05

# Model parameters (MUST MATCH YOUR TRAINING - from env.py line 58)
LOOKBACK_WINDOW_SIZE = 50
ACTION_SPACE = 3  # Hold, Buy, Sell
NUM_FEATURES = 20  # Exactly 20 features from cryptoanalysis_data.csv

# Les 20 features EXACTES de votre CSV (dans l'ordre)
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
    'Close'  # Ajouté à la fin dans reset() de env.py
]

# State management
recent_candles = []
MAX_CANDLES = 200

# Global model variable
actor_model = None

# Cache pour les données externes (mise à jour toutes les heures)
external_data_cache = {
    'last_update': 0,
    'data': {}
}

def create_actor_model(input_shape, action_space, lr=0.00001):
    """
    Recrée EXACTEMENT l'architecture du modèle Actor.
    Doit correspondre à Shared_Model dans models.py (lignes 79-102)
    """
    X_input = Input(input_shape)
    
    # Shared CNN layers (identique à models.py lignes 88-92)
    X = Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(X_input)
    X = MaxPooling1D(pool_size=2)(X)
    X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(X)
    X = MaxPooling1D(pool_size=2)(X)
    X = Flatten()(X)
    
    # Actor layers (identique à models.py lignes 105-108)
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
        
        print("Model architecture created:")
        actor_model.summary()
        
        print(f"\nLoading weights from: {actor_path}")
        actor_model.load_weights(actor_path)
        
        print("✓ Model loaded successfully!")
        print(f"✓ Using {NUM_FEATURES} features matching training data")
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
    Récupère les données externes (OIL, GOLD, S&P500, VIX, etc.)
    Cache pendant 1 heure pour éviter trop de requêtes API
    """
    global external_data_cache
    
    current_time = time.time()
    
    # Si cache encore valide (moins d'1 heure)
    if current_time - external_data_cache['last_update'] < 3600:
        return external_data_cache['data']
    
    print("🌐 Fetching external market data...")
    
    try:
        # PLACEHOLDER - Remplacez par vos vraies API calls
        # Pour le moment, on utilise des valeurs par défaut
        
        external_data = {
            'OIL': 75.0,           # Prix du pétrole
            'GOLD': 2000.0,        # Prix de l'or
            's&p500': 4500.0,      # S&P 500
            'VIX': 15.0,           # Volatility Index
            'UVYX': 10.0,          # ProShares Ultra VIX
            'btchashrate': 400e18  # Bitcoin hashrate
        }
        
        # TODO: Remplacer par de vraies API calls:
        # - Oil: https://www.alphavantage.co/query?function=WTI...
        # - Gold: https://www.alphavantage.co/query?function=XAU...
        # - S&P500: https://www.alphavantage.co/query?function=SPY...
        # - VIX: https://www.alphavantage.co/query?function=^VIX...
        
        external_data_cache['data'] = external_data
        external_data_cache['last_update'] = current_time
        
        print(f"✓ External data updated: {external_data}")
        return external_data
        
    except Exception as e:
        print(f"⚠ Error fetching external data: {e}")
        # Retourne les dernières données en cache ou des valeurs par défaut
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
    Prépare l'état EXACTEMENT comme dans l'entraînement.
    Utilise les 20 features EXACTES de cryptoanalysis_data.csv
    """
    if len(candles_data) < LOOKBACK_WINDOW_SIZE:
        return None
    
    # Prend les dernières LOOKBACK_WINDOW_SIZE bougies
    recent = candles_data[-LOOKBACK_WINDOW_SIZE:]
    
    # Convertit en DataFrame
    df = pd.DataFrame(recent)
    
    # Convertit en float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    # Récupère les données externes
    external_data = get_external_data()
    
    # Crée les features EXACTEMENT comme dans env.py reset() (lignes 270-290)
    # L'ordre EST CRITIQUE!
    features_list = []
    
    for idx, row in df.iterrows():
        # DOIT correspondre EXACTEMENT à blockchain_data.append() dans env.py
        features = [
            float(row['close']),                    # Close (sera retiré pour normalisation mais ajouté à la fin)
            external_data.get('receive_count', 1000),  # receive_count - À REMPLACER par vraie API
            external_data.get('sent_count', 1000),     # sent_count
            float(row['volume']) * 0.001,              # avg_fee (approximation)
            1000000,                                    # blocksize - À REMPLACER
            external_data.get('btchashrate', 400e18),  # btchashrate
            external_data.get('OIL', 75.0),            # OIL
            500,                                        # ecr20_transfers - À REMPLACER
            external_data.get('GOLD', 2000.0),         # GOLD
            100,                                        # searches (Google Trends) - À REMPLACER
            300e12,                                     # hashrate - À REMPLACER
            150e9,                                      # marketcap - À REMPLACER
            5e12,                                       # difficulty - À REMPLACER
            external_data.get('s&p500', 4500.0),       # s&p500
            float(row['volume']) * 0.0001,             # transactionfee
            1000,                                       # transactions - À REMPLACER
            50,                                         # tweet_count - À REMPLACER
            50000,                                      # unique_adresses - À REMPLACER
            external_data.get('VIX', 15.0),            # VIX
            external_data.get('UVYX', 10.0)            # UVYX
        ]
        
        features_list.append(features)
    
    # Convertit en numpy array
    state = np.array(features_list)
    
    # Vérifie la forme
    if state.shape[1] != NUM_FEATURES:
        print(f"⚠ WARNING: Feature mismatch! Expected {NUM_FEATURES}, got {state.shape[1]}")
        return None
    
    # Normalisation IDENTIQUE à l'entraînement (main.py lignes 23-26)
    if DF_MIN is not None and DF_MAX is not None:
        # Utilise les paramètres d'entraînement
        state = (state - DF_MIN) / (DF_MAX - DF_MIN)
    else:
        # Fallback: normalisation par fenêtre
        state_min = state.min()
        state_max = state.max()
        if state_max - state_min > 0:
            state = (state - state_min) / (state_max - state_min)
    
    # Reshape pour le modèle: (1, lookback_window, features)
    state = state.reshape(1, LOOKBACK_WINDOW_SIZE, NUM_FEATURES)
    
    return state

def get_model_action(state):
    """
    Obtient l'action du modèle Actor.
    Returns: 0 (Hold), 1 (Buy), 2 (Sell)
    """
    if actor_model is None or state is None:
        print("⚠ Model not loaded or state invalid - defaulting to HOLD")
        return 0
    
    try:
        # Prédiction du modèle
        prediction = actor_model.predict(state, verbose=0)
        
        # Action avec la plus haute probabilité
        action = np.argmax(prediction[0])
        
        # Scores de confiance
        confidence = prediction[0]
        
        print(f"\n🤖 Model Prediction:")
        print(f"   Hold:  {confidence[0]:.2%}")
        print(f"   Buy:   {confidence[1]:.2%}")
        print(f"   Sell:  {confidence[2]:.2%}")
        print(f"   → Action: {['HOLD', 'BUY', 'SELL'][action]}")
        
        return action
        
    except Exception as e:
        print(f"✗ Error getting model prediction: {e}")
        import traceback
        traceback.print_exc()
        return 0

def append_list_as_row(file_name, list_of_elem):
    """Append a list as a row to a CSV file."""
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

def order(side, quantity, symbol, order_type='MARKET'):
    """Place an order - currently simulated."""
    print(f"\n💰 ORDER SIMULATION: {side} {quantity} {symbol}")
    print("   ⚠ REAL TRADING DISABLED FOR SAFETY")
    
    # Log l'ordre simulé
    timestamp = int(time.time() * 1000)
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
        append_list_as_row("../transaction_history.csv", order_data)
        print("   ✓ Order logged to transaction_history.csv")
    except Exception as e:
        print(f"   ✗ Failed to log order: {e}")
    
    return True

def on_open(ws):
    """Called when the WebSocket is opened."""
    print("\n" + "="*60)
    print("WEBSOCKET CONNECTED")
    print("="*60)
    send_mail('PPO RL trading bot is online and monitoring.')
    print('✓ PPO RL bot is online')

def on_close(ws, close_status_code, close_msg):
    """Called when the WebSocket is closed."""
    print("\n" + "="*60)
    print("WEBSOCKET CLOSED")
    print("="*60)
    send_mail('Server closed. PPO RL bot stopped.')
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
        
        # Stocke les données de la bougie
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
            
            # Garde seulement les dernières MAX_CANDLES
            if len(recent_candles) > MAX_CANDLES:
                recent_candles.pop(0)
            
            # Attend d'avoir assez de données
            if len(recent_candles) < LOOKBACK_WINDOW_SIZE:
                print(f"⏳ Collecting data... ({len(recent_candles)}/{LOOKBACK_WINDOW_SIZE} candles)")
                return
            
            # Prépare l'état pour le modèle
            state = prepare_state(recent_candles)
            
            if state is not None:
                print(f"✓ State prepared: shape={state.shape}")
                
                # Obtient l'action du modèle
                action = get_model_action(state)
                
                # Lit l'historique des transactions
                try:
                    df = pd.read_csv("../transaction_history.csv")
                    if len(df) > 0:
                        limited_data = df.tail(10).iloc[::-1]
                        last_side = limited_data['side'].iloc[0]
                    else:
                        last_side = "SELL"
                except Exception as e:
                    print(f"⚠ Could not read transaction history: {e}")
                    last_side = "SELL"
                
                print(f"📍 Current Position: {last_side}")
                
                # Exécute les trades selon l'action du modèle
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
    print(f"Features: {FEATURE_COLUMNS}")
    
    # Charge le modèle
    model_loaded = load_actor_model()
    
    if not model_loaded:
        print("\n⚠ WARNING: Model not loaded!")
        print("Please check:")
        print("1. Model path is correct")
        print("2. Model files exist")
        print("3. Input shape matches your training data")
        response = input("\nContinue in monitoring mode? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    print("\n" + "="*60)
    print("CONNECTING TO BINANCE WEBSOCKET")
    print("="*60)
    
    # Crée et lance le WebSocket
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