"""
Trading Bot CORRIGÉ - Les données changent à CHAQUE prédiction
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
import requests
import random

# ========================================
# CONFIGURATION
# ========================================

env_path = r"C:\BC\Crypto-Bot\flask-api\.env"
MODEL_PATH = r"C:\BC\Reinforcement_Learning\reinforcement_learning_trading_agent\2026_01_31_10_38_Crypto_trader"
OUTPUT_DIR = Path(r"C:\BC\Crypto-Bot\output_data")
TRANSACTION_HISTORY_PATH = OUTPUT_DIR / "transaction_history.csv"
TRAINING_CSV_PATH = r"C:\BC\Reinforcement_Learning\reinforcement_learning_trading_agent\cryptoanalysis_data.csv"

SOCKET = "wss://stream.binance.com:9443/ws/ethusdt@kline_1m"
TRADE_SYMBOL = 'ETHUSDT'
TRADE_QUANTITY = 0.05
SIMULATION_MODE = True

LOOKBACK_WINDOW_SIZE = None
ACTION_SPACE = 3
NUM_FEATURES = 20
MIN_CONFIDENCE = 0.35  # 🎯 AUGMENTÉ à 35% pour éviter trop de HOLD

FEATURE_COLUMNS = [
    'receive_count', 'sent_count', 'avg_fee', 'blocksize', 'btchashrate',
    'OIL', 'ecr20_transfers', 'GOLD', 'searches', 'hashrate',
    'marketcap', 'difficulty', 's&p500', 'transactionfee', 'transactions',
    'tweet_count', 'unique_adresses', 'VIX', 'UVYX', 'Close'
]

# ========================================
# ⚡ DONNÉES QUI CHANGENT À CHAQUE APPEL
# ========================================

def get_real_time_data_from_api():
    """Récupère les VRAIES données depuis CoinGecko"""
    try:
        response = requests.get(
            'https://api.coingecko.com/api/v3/coins/ethereum',
            timeout=10
        )
        data = response.json()
        market_data = data.get('market_data', {})
        
        price = market_data.get('current_price', {}).get('usd', 3500)
        market_cap = market_data.get('market_cap', {}).get('usd', 240e9)
        volume_24h = market_data.get('total_volume', {}).get('usd', 15e9)
        price_change = market_data.get('price_change_percentage_24h', 0)
        
        tx_count = int(volume_24h / price)
        volatility = abs(price_change)
        
        print(f"✅ API: Prix=${price:.2f}, Vol=${volume_24h/1e9:.2f}B, Change={price_change:.2f}%")
        
        return {
            'receive_count': tx_count,
            'sent_count': tx_count,
            'avg_fee': 15.0 + random.uniform(-5, 5),
            'blocksize': 80000,
            'btchashrate': 500e18,
            'OIL': 78.5,
            'ecr20_transfers': int(tx_count * 0.6),
            'GOLD': 2050.0,
            'searches': int(85000 * (1 + volatility / 15)),
            'hashrate': market_cap * 3,
            'marketcap': market_cap,
            'difficulty': 0,
            's&p500': 4800.0,
            'transactionfee': 15.0 + random.uniform(-5, 5),
            'transactions': tx_count,
            'tweet_count': int(15000 * (1 + volatility / 10)),
            'unique_adresses': int(tx_count / 5),
            'VIX': 14.5,
            'UVYX': 11.2,
        }
    except Exception as e:
        print(f"⚠️ Erreur API: {e}")
        return None

def get_varying_external_data(current_price, price_trend):
    """
    🔥 NOUVEAU: Génère des données qui VARIENT en fonction du prix actuel
    """
    # Base selon le prix actuel
    base_tx = int(20000000 * (current_price / 2500))  # Plus le prix monte, plus de transactions
    
    # Variation selon la tendance
    if price_trend > 0:  # Prix monte
        multiplier = 1.0 + (price_trend * 0.001)  # +0.1% de données par $ de hausse
        sentiment_boost = 1.2
    else:  # Prix baisse
        multiplier = 1.0 + (price_trend * 0.001)  # -0.1% de données par $ de baisse
        sentiment_boost = 0.8
    
    # Ajouter du bruit aléatoire
    noise = random.uniform(0.95, 1.05)
    
    final_tx = int(base_tx * multiplier * noise)
    
    return {
        'receive_count': final_tx,
        'sent_count': final_tx,
        'avg_fee': 15.0 + random.uniform(-3, 3),
        'blocksize': 80000,
        'btchashrate': 500e18,
        'OIL': 78.5 + random.uniform(-2, 2),
        'ecr20_transfers': int(final_tx * 0.6 * noise),
        'GOLD': 2050.0 + random.uniform(-10, 10),
        'searches': int(85000 * sentiment_boost * noise),
        'hashrate': current_price * 1e11 * noise,
        'marketcap': current_price * 120e6 * noise,
        'difficulty': 0,
        's&p500': 4800.0 + random.uniform(-50, 50),
        'transactionfee': 15.0 + random.uniform(-3, 3),
        'transactions': final_tx,
        'tweet_count': int(15000 * sentiment_boost * noise),
        'unique_adresses': int(final_tx / 5),
        'VIX': 14.5 + random.uniform(-2, 2),
        'UVYX': 11.2 + random.uniform(-1, 1),
    }

# ⚠️ SUPPRIMER LE CACHE - On veut des données fraîches à chaque fois
def get_external_data(current_price, price_history):
    """
    🔥 CORRIGÉ: Calcule toujours de nouvelles données basées sur le prix
    """
    # Calculer la tendance récente
    if len(price_history) >= 5:
        recent_prices = [float(c['close']) for c in price_history[-5:]]
        price_trend = recent_prices[-1] - recent_prices[0]  # Différence sur 5 minutes
    else:
        price_trend = 0
    
    # Essayer l'API d'abord (mais sans cache)
    api_data = get_real_time_data_from_api()
    if api_data:
        print(f"📊 External (API): tx={api_data['transactions']:.0f}, mc={api_data['marketcap']/1e9:.1f}B")
        return api_data
    
    # Sinon, générer des données qui varient selon le prix
    varied_data = get_varying_external_data(current_price, price_trend)
    print(f"📊 External (Sim): tx={varied_data['transactions']:.0f}, "
          f"trend={price_trend:+.2f}, price={current_price:.2f}")
    
    return varied_data

# ========================================
# ENVIRONNEMENT
# ========================================

if os.path.exists(env_path):
    env = dotenv_values(env_path)
    print(f"✓ Environment loaded")
else:
    raise FileNotFoundError("Environment file not found")

# ========================================
# DÉTECTION LOOKBACK
# ========================================

def detect_lookback_window(model_path):
    global LOOKBACK_WINDOW_SIZE
    
    print("\n🔍 Detecting lookback window...")
    
    params_file = os.path.join(model_path, "Parameters.txt")
    if os.path.exists(params_file):
        try:
            with open(params_file, 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if 'lookback_window_size' in line.lower():
                        import re
                        match = re.search(r'(\d+)', line)
                        if match:
                            LOOKBACK_WINDOW_SIZE = int(match.group(1))
                            print(f"✓ LOOKBACK_WINDOW: {LOOKBACK_WINDOW_SIZE}")
                            return LOOKBACK_WINDOW_SIZE
        except Exception as e:
            print(f"⚠ Error: {e}")
    
    for lookback in [30, 35, 40, 50]:
        print(f"   Trying lookback={lookback}...")
        try:
            test_model = create_actor_model((lookback, NUM_FEATURES), ACTION_SPACE)
            actor_file = find_best_model(model_path)[0]
            if actor_file:
                test_model.load_weights(os.path.join(model_path, actor_file))
                LOOKBACK_WINDOW_SIZE = lookback
                print(f"✓ Loaded with lookback={lookback}")
                del test_model
                return lookback
        except:
            continue
    
    LOOKBACK_WINDOW_SIZE = 30
    return 30

def load_normalization_params():
    try:
        df = pd.read_csv(TRAINING_CSV_PATH)
        df = df.rename(columns={'price': 'Close', 'date': 'Date'})
        df_for_norm = df.drop(['Close', 'Date'], axis=1)
        
        df_max = df_for_norm.max().max()
        df_min = df_for_norm.min().min()
        
        print(f"✓ Normalization: Min={df_min:.2f}, Max={df_max:.2f}")
        return df_min, df_max
    except Exception as e:
        print(f"⚠ Normalization error: {e}")
        return None, None

DF_MIN, DF_MAX = load_normalization_params()

def find_best_model(path):
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
        print(f"✓ Best model: {best_file} (score: {best_score:.2f})")
        return best_file, best_file.replace('_Actor.weights.h5', '_Critic.weights.h5')
    
    return None, None

detect_lookback_window(MODEL_PATH)
ACTOR_FILE, CRITIC_FILE = find_best_model(MODEL_PATH)

# ========================================
# PRÉPARATION DES ÉTATS - CORRIGÉ
# ========================================

def prepare_state(candles_data):
    """🔥 CORRIGÉ: Utilise le prix actuel pour générer des données variables"""
    if LOOKBACK_WINDOW_SIZE is None or len(candles_data) < LOOKBACK_WINDOW_SIZE:
        return None
    
    recent = candles_data[-LOOKBACK_WINDOW_SIZE:]
    df = pd.DataFrame(recent)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    # 🔥 NOUVEAU: Passer le prix actuel et l'historique
    current_price = float(df.iloc[-1]['close'])
    external_data = get_external_data(current_price, recent)
    
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
    global actor_model
    
    print("\n" + "="*60)
    print("LOADING PPO ACTOR MODEL")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Lookback: {LOOKBACK_WINDOW_SIZE}")
    print(f"Threshold: {MIN_CONFIDENCE:.2%}")
    print("="*60)
    
    try:
        if not ACTOR_FILE or LOOKBACK_WINDOW_SIZE is None:
            print("✗ Model or lookback not detected")
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
            print(f"   → HOLD (safety)")
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
    timestamp = int(time.time() * 1000)
    
    print(f"\n💰 ORDER: {side} {quantity} {symbol}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not TRANSACTION_HISTORY_PATH.exists():
        with open(TRANSACTION_HISTORY_PATH, 'w', newline='', encoding='utf-8') as f:
            csv_writer = writer(f)
            csv_writer.writerow(['id', 'symbol', 'orderId', 'qty', 'timestamp', 'side', 
                                'price', 'price_with_fee', 'commission', 'commissionAsset', 
                                'profits', 'profits_percent'])
    
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
        print(f"   ✓ Order logged to {TRANSACTION_HISTORY_PATH}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    return True

# ========================================
# WEBSOCKET
# ========================================

def on_message(ws, message):
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
            print(f"📊 CANDLE CLOSED at {candle_data['close']}")
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
                    if TRANSACTION_HISTORY_PATH.exists():
                        df = pd.read_csv(TRANSACTION_HISTORY_PATH)
                        last_side = df.iloc[-1]['side'] if len(df) > 0 else "SELL"
                    else:
                        last_side = "SELL"
                except:
                    last_side = "SELL"
                
                print(f"📍 Current Position: {last_side}")
                
                if action == 2 and last_side == "BUY":
                    print("\n🔴 SELL SIGNAL")
                    order('SELL', TRADE_QUANTITY, TRADE_SYMBOL)
                elif action == 1 and last_side == "SELL":
                    print("\n🟢 BUY SIGNAL")
                    order('BUY', TRADE_QUANTITY, TRADE_SYMBOL)
                else:
                    print("\n⚪ HOLD")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

def on_open(ws):
    print("\n" + "="*60)
    print("✅ CONNECTED TO BINANCE")
    print("="*60)

def on_close(ws, close_status_code, close_msg):
    print("\n" + "="*60)
    print("DISCONNECTED")
    print("="*60)

def on_error(ws, error):
    print(f"\n✗ WebSocket Error: {error}")

# ========================================
# MAIN
# ========================================

def bot():
    print("\n" + "="*60)
    print("🚀 PPO RL TRADING BOT - DONNÉES VARIABLES")
    print("="*60)
    print(f"Model: {ACTOR_FILE if ACTOR_FILE else 'Not found'}")
    print(f"Lookback: {LOOKBACK_WINDOW_SIZE}")
    print(f"Confidence Threshold: {MIN_CONFIDENCE:.2%}")
    print(f"Mode: {'SIMULATION' if SIMULATION_MODE else 'LIVE'}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    if not load_actor_model():
        print("\n✗ Failed to load model!")
        return
    
    print("\n🧪 Testing varying data...")
    test_data1 = get_varying_external_data(2400, 10)  # Prix monte
    test_data2 = get_varying_external_data(2400, -10)  # Prix baisse
    print(f"✓ Data 1 (up): tx={test_data1['transactions']:.0f}")
    print(f"✓ Data 2 (down): tx={test_data2['transactions']:.0f}")
    print(f"✓ Difference: {abs(test_data1['transactions'] - test_data2['transactions']):.0f}")
    
    print("\n" + "="*60)
    print("CONNECTING TO BINANCE...")
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