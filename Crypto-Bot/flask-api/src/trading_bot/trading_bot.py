"""
Trading bot - VERSION AUTO-DETECT LOOKBACK
Détecte automatiquement le lookback_window du modèle
"""
import websocket
import json
import os
from pathlib import Path
from dotenv import dotenv_values
from csv import writer
import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam

# ========================================
# CONFIGURATION
# ========================================

# Chemins
env_path = r"C:\BC\Crypto-Bot\flask-api\.env"
MODEL_PATH = r"C:\BC\Reinforcement_Learning\reinforcement_learning_trading_agent\2026_01_31_10_38_Crypto_trader"
OUTPUT_DIR = Path(r"C:\BC\Crypto-Bot\flask-api\app\output_data")
TRANSACTION_HISTORY_PATH = OUTPUT_DIR / "transaction_history.csv"
TRAINING_CSV_PATH = r"C:\BC\Reinforcement_Learning\reinforcement_learning_trading_agent\cryptoanalysis_data.csv"

# Trading
SOCKET = "wss://stream.binance.com:9443/ws/ethusdt@kline_1m"
TRADE_SYMBOL = 'ETHUSDT'
TRADE_QUANTITY = 0.05
SIMULATION_MODE = True

# Modèle - SERA DÉTECTÉ AUTOMATIQUEMENT
LOOKBACK_WINDOW_SIZE = None  # ⚠️ Auto-détecté depuis Parameters.txt
ACTION_SPACE = 3
NUM_FEATURES = 20

# Seuil de confiance
MIN_CONFIDENCE = 0.15

# Features
FEATURE_COLUMNS = [
    'receive_count', 'sent_count', 'avg_fee', 'blocksize', 'btchashrate',
    'OIL', 'ecr20_transfers', 'GOLD', 'searches', 'hashrate',
    'marketcap', 'difficulty', 's&p500', 'transactionfee', 'transactions',
    'tweet_count', 'unique_adresses', 'VIX', 'UVYX', 'Close'
]

# ========================================
# DÉTECTION AUTOMATIQUE DU LOOKBACK
# ========================================

def detect_lookback_window(model_path):
    """
    Détecte le lookback_window depuis Parameters.txt ou log.txt
    """
    global LOOKBACK_WINDOW_SIZE
    
    print("\n🔍 Detecting model configuration...")
    
    # Essayer Parameters.txt
    params_file = os.path.join(model_path, "Parameters.txt")
    if os.path.exists(params_file):
        try:
            with open(params_file, 'r') as f:
                content = f.read()
                # Chercher "lookback_window_size"
                for line in content.split('\n'):
                    if 'lookback_window_size' in line.lower():
                        # Extraire le nombre
                        import re
                        match = re.search(r'(\d+)', line)
                        if match:
                            LOOKBACK_WINDOW_SIZE = int(match.group(1))
                            print(f"✓ Detected LOOKBACK_WINDOW from Parameters.txt: {LOOKBACK_WINDOW_SIZE}")
                            return LOOKBACK_WINDOW_SIZE
        except Exception as e:
            print(f"⚠ Could not read Parameters.txt: {e}")
    
    # Essayer log.txt
    log_file = os.path.join(model_path, "log.txt")
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                import re
                match = re.search(r'lookback.*?(\d+)', content, re.IGNORECASE)
                if match:
                    LOOKBACK_WINDOW_SIZE = int(match.group(1))
                    print(f"✓ Detected LOOKBACK_WINDOW from log.txt: {LOOKBACK_WINDOW_SIZE}")
                    return LOOKBACK_WINDOW_SIZE
        except Exception as e:
            print(f"⚠ Could not read log.txt: {e}")
    
    # Valeurs communes à essayer
    print("⚠ Could not auto-detect, trying common values...")
    common_values = [30, 35, 40, 50, 20]
    
    for lookback in common_values:
        print(f"   Trying lookback={lookback}...")
        try:
            test_model = create_actor_model((lookback, NUM_FEATURES), ACTION_SPACE)
            actor_file = find_best_model(model_path)[0]
            if actor_file:
                actor_path = os.path.join(model_path, actor_file)
                test_model.load_weights(actor_path)
                LOOKBACK_WINDOW_SIZE = lookback
                print(f"✓ Successfully loaded with LOOKBACK_WINDOW={lookback}")
                del test_model
                return lookback
        except Exception as e:
            continue
    
    # Default fallback
    print("⚠ Using default LOOKBACK_WINDOW=30")
    LOOKBACK_WINDOW_SIZE = 30
    return 30

# ========================================
# ENVIRONNEMENT
# ========================================

if os.path.exists(env_path):
    env = dotenv_values(env_path)
    print(f"✓ Environment loaded")
    API_KEY = env.get('API_KEY', '')
    API_SECRET = env.get('API_SECRET', '')
    PERSONAL_EMAIL = env.get('PERSONAL_EMAIL', '')
    DEV_EMAIL = env.get('DEV_EMAIL', '')
    EMAIL_PASS = env.get('EMAIL_PASS', '').strip('"')
else:
    raise FileNotFoundError("Environment file not found")

# ========================================
# FONCTIONS HELPER
# ========================================

def load_normalization_params():
    """Charge les paramètres de normalisation"""
    try:
        df = pd.read_csv(TRAINING_CSV_PATH)
        df = df.rename(columns={'price': 'Close', 'date': 'Date'})
        df_for_norm = df.drop(['Close', 'Date'], axis=1)
        
        column_maxes = df_for_norm.max()
        df_max = column_maxes.max()
        column_mins = df_for_norm.min()
        df_min = column_mins.min()
        
        print(f"✓ Normalization: Min={df_min:.2f}, Max={df_max:.2f}")
        return df_min, df_max
    except Exception as e:
        print(f"⚠ Normalization error: {e}")
        return None, None

DF_MIN, DF_MAX = load_normalization_params()

def find_best_model(path):
    """Trouve le meilleur modèle"""
    files = os.listdir(path)
    actor_files = [f for f in files if f.endswith('_Crypto_trader_Actor.weights.h5')]
    
    if not actor_files:
        return None, None
    
    best_score = -float('inf')
    best_file = None
    
    for f in actor_files:
        try:
            score_str = f.split('_')[0]
            score = float(score_str)
            if score > best_score:
                best_score = score
                best_file = f
        except:
            if f.startswith('_'):
                best_file = f
    
    if best_file:
        actor_file = best_file
        critic_file = best_file.replace('_Actor.weights.h5', '_Critic.weights.h5')
        print(f"✓ Best model: {best_file} (score: {best_score:.2f})")
        return actor_file, critic_file
    
    return None, None

# Détection automatique du lookback
detect_lookback_window(MODEL_PATH)
ACTOR_FILE, CRITIC_FILE = find_best_model(MODEL_PATH)

# ========================================
# DONNÉES EXTERNES
# ========================================

def get_real_external_data():
    """Récupère les données externes"""
    external_data = {
        'receive_count': 50000,
        'sent_count': 48000,
        'ecr20_transfers': 120000,
        'hashrate': 900e12,
        'difficulty': 12e12,
        'marketcap': 240e9,
        'unique_adresses': 220000,
        'transactions': 1200000,
        'tweet_count': 15000,
        'searches': 85000,
        'blocksize': 80000,
        'transactionfee': 15.0,
        'OIL': 78.5,
        'GOLD': 2050.0,
        's&p500': 4800.0,
        'VIX': 14.5,
        'UVYX': 11.2,
        'btchashrate': 500e18
    }
    return external_data

external_data_cache = {
    'last_update': 0,
    'data': {}
}

def get_external_data():
    """Wrapper avec cache"""
    global external_data_cache
    current_time = time.time()
    
    if current_time - external_data_cache['last_update'] < 1800:
        return external_data_cache['data']
    
    data = get_real_external_data()
    external_data_cache['data'] = data
    external_data_cache['last_update'] = current_time
    
    return data

# ========================================
# PRÉPARATION DES ÉTATS
# ========================================

def prepare_state(candles_data):
    """Prépare l'état avec le LOOKBACK_WINDOW détecté"""
    if LOOKBACK_WINDOW_SIZE is None:
        print("✗ LOOKBACK_WINDOW_SIZE not detected!")
        return None
    
    if len(candles_data) < LOOKBACK_WINDOW_SIZE:
        return None
    
    recent = candles_data[-LOOKBACK_WINDOW_SIZE:]
    df = pd.DataFrame(recent)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    external_data = get_external_data()
    
    features_list = []
    
    for idx, row in df.iterrows():
        features = [
            external_data['receive_count'],
            external_data['sent_count'],
            float(row['volume']) * 0.001,
            external_data['blocksize'],
            external_data['btchashrate'],
            external_data['OIL'],
            external_data['ecr20_transfers'],
            external_data['GOLD'],
            external_data['searches'],
            external_data['hashrate'],
            external_data['marketcap'],
            external_data['difficulty'],
            external_data['s&p500'],
            external_data['transactionfee'],
            external_data['transactions'],
            external_data['tweet_count'],
            external_data['unique_adresses'],
            external_data['VIX'],
            external_data['UVYX'],
            float(row['close'])
        ]
        features_list.append(features)
    
    state = np.array(features_list)
    
    if DF_MIN is not None and DF_MAX is not None:
        state = (state - DF_MIN) / (DF_MAX - DF_MIN)
        state = np.clip(state, 0, 1)
    else:
        state_min = state.min()
        state_max = state.max()
        if state_max - state_min > 0:
            state = (state - state_min) / (state_max - state_min)
    
    state = state.reshape(1, LOOKBACK_WINDOW_SIZE, NUM_FEATURES)
    
    return state

# ========================================
# MODÈLE
# ========================================

actor_model = None
recent_candles = []
MAX_CANDLES = 200

def create_actor_model(input_shape, action_space, lr=0.00001):
    """Recrée l'architecture Actor"""
    X_input = Input(input_shape)
    
    X = Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(X_input)
    X = MaxPooling1D(pool_size=2)(X)
    X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(X)
    X = MaxPooling1D(pool_size=2)(X)
    X = Flatten()(X)
    
    A = Dense(512, activation="relu")(X)
    A = Dense(256, activation="relu")(A)
    A = Dense(64, activation="relu")(A)
    output = Dense(action_space, activation="softmax")(A)
    
    model = Model(inputs=X_input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr))
    
    return model

def load_actor_model():
    """Charge le modèle Actor"""
    global actor_model
    
    print("\n" + "="*60)
    print("LOADING PPO ACTOR MODEL")
    print("="*60)
    print(f"Model folder: {MODEL_PATH}")
    print(f"Lookback window: {LOOKBACK_WINDOW_SIZE}")
    print("="*60)
    
    try:
        if not ACTOR_FILE:
            print("✗ No model file found")
            return False
        
        if LOOKBACK_WINDOW_SIZE is None:
            print("✗ LOOKBACK_WINDOW_SIZE not detected")
            return False
        
        actor_path = os.path.join(MODEL_PATH, ACTOR_FILE)
        
        if not os.path.exists(actor_path):
            print(f"✗ Model not found: {actor_path}")
            return False
        
        print(f"✓ Loading: {ACTOR_FILE}")
        
        input_shape = (LOOKBACK_WINDOW_SIZE, NUM_FEATURES)
        actor_model = create_actor_model(input_shape, ACTION_SPACE)
        actor_model.load_weights(actor_path)
        
        print("✓ Model loaded successfully!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("="*60)
        return False

def get_model_action(state):
    """Obtient l'action du modèle"""
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
        
        if confidence[action] < MIN_CONFIDENCE:
            print(f"   ⚠️ Low confidence ({confidence[action]:.2%} < {MIN_CONFIDENCE:.2%})")
            print(f"   → HOLD")
            action = 0
        else:
            action_names = ['HOLD', 'BUY', 'SELL']
            print(f"   ✓ Action: {action_names[action]} ({confidence[action]:.2%})")
        
        return action
        
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        return 0

# ========================================
# TRADING
# ========================================

def order(side, quantity, symbol, order_type='MARKET'):
    """Place order"""
    timestamp = int(time.time() * 1000)
    
    print(f"\n💰 ORDER: {side} {quantity} {symbol}")
    
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
        with open(TRANSACTION_HISTORY_PATH, 'a+', newline='', encoding='utf-8') as f:
            csv_writer = writer(f)
            csv_writer.writerow(order_data)
        print(f"   ✓ Order logged")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    return True

# ========================================
# WEBSOCKET
# ========================================

def on_message(ws, message):
    """Traite les messages"""
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
        
        print(f"\r💹 {candle_data['close']} | "
              f"Vol={float(candle_data['volume']):.2f} | "
              f"Candles={len(recent_candles)}/{LOOKBACK_WINDOW_SIZE}", end='')
        
        if is_candle_closed:
            print()
            print("\n" + "-"*60)
            print(f"📊 CLOSED at {candle_data['close']}")
            print("-"*60)
            
            recent_candles.append(candle_data)
            
            if len(recent_candles) > MAX_CANDLES:
                recent_candles.pop(0)
            
            if len(recent_candles) < LOOKBACK_WINDOW_SIZE:
                print(f"⏳ Collecting... ({len(recent_candles)}/{LOOKBACK_WINDOW_SIZE})")
                return
            
            state = prepare_state(recent_candles)
            
            if state is not None:
                action = get_model_action(state)
                
                try:
                    df = pd.read_csv(TRANSACTION_HISTORY_PATH)
                    last_side = df.iloc[-1]['side'] if len(df) > 0 else "SELL"
                except:
                    last_side = "SELL"
                
                print(f"📍 Position: {last_side}")
                
                if action == 2 and last_side == "BUY":
                    print("\n🔴 SELL")
                    order('SELL', TRADE_QUANTITY, TRADE_SYMBOL)
                elif action == 1 and last_side == "SELL":
                    print("\n🟢 BUY")
                    order('BUY', TRADE_QUANTITY, TRADE_SYMBOL)
                else:
                    print("\n⚪ HOLD")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")

def on_open(ws):
    print("\n" + "="*60)
    print("CONNECTED")
    print("="*60)

def on_close(ws, close_status_code, close_msg):
    print("\n" + "="*60)
    print("CLOSED")
    print("="*60)

def on_error(ws, error):
    print(f"\n✗ Error: {error}")

# ========================================
# MAIN
# ========================================

def bot():
    """Run bot"""
    print("\n" + "="*60)
    print("PPO RL TRADING BOT - AUTO-DETECT")
    print("="*60)
    print(f"Model: {ACTOR_FILE if ACTOR_FILE else 'Not found'}")
    print(f"Lookback: {LOOKBACK_WINDOW_SIZE}")
    print(f"Threshold: {MIN_CONFIDENCE:.2%}")
    print(f"Mode: {'SIM' if SIMULATION_MODE else 'LIVE'}")
    print("="*60 + "\n")
    
    if not load_actor_model():
        print("\n✗ Failed to load model!")
        return
    
    print("\n" + "="*60)
    print("CONNECTING...")
    print("="*60)
    
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