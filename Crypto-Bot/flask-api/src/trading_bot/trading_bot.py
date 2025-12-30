"""
Trading bot with PPO Actor-Critic RL model integration
Loads the trained CNN model from reinforcement_learning_trading_agent
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
from data_handler.get_hitorical_eth_data import get_data_onStart

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
MODEL_PATH = r"C:\BC\Reinforcement Learning\reinforcement_learning_trading_agent\2025_12_16_13_57_Crypto_trader"

# Find the best model (highest reward)
def find_best_model(path):
    """Find the model with highest reward score"""
    files = os.listdir(path)
    actor_files = [f for f in files if f.endswith('_Crypto_trader_Actor.weights.h5')]
    
    if not actor_files:
        return None, None
    
    # Extract scores from filenames (format: SCORE_Crypto_trader_Actor.weights.h5)
    best_score = -float('inf')
    best_file = None
    
    for f in actor_files:
        try:
            score = float(f.split('_')[0])
            if score > best_score:
                best_score = score
                best_file = f
        except:
            if f.startswith('_'):  # Latest model without score
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

# Model parameters (MUST MATCH YOUR TRAINING)
LOOKBACK_WINDOW_SIZE = 50  # From your training code
ACTION_SPACE = 3  # Hold, Buy, Sell

# State management
recent_candles = []
MAX_CANDLES = 200

# Global model variable
actor_model = None

def create_actor_model(input_shape, action_space, lr=0.00001):
    """
    Recreate the Actor model architecture from your training code.
    This MUST match exactly what you used during training.
    """
    X_input = Input(input_shape)
    
    # Shared CNN layers (from Shared_Model with model="CNN")
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
        
        # Get number of features from a sample calculation
        # Your training normalizes all features except Close and Date
        # Typical features: open, high, low, close, volume + technical indicators
        # From your code: df has multiple columns, Close is separate
        # Let's assume ~20 features (adjust if you know exact number)
        num_features = 20  # ADJUST THIS based on your actual feature count
        
        input_shape = (LOOKBACK_WINDOW_SIZE, num_features)
        print(f"Input shape: {input_shape}")
        
        # Create model architecture
        print("Creating Actor model architecture...")
        actor_model = create_actor_model(input_shape, ACTION_SPACE)
        
        print("Model architecture created:")
        actor_model.summary()
        
        # Load weights
        print(f"\nLoading weights from: {actor_path}")
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

def normalize_data(data, df_min, df_max):
    """
    Normalize data the same way as in training.
    """
    return (data - df_min) / (df_max - df_min)

def prepare_state(candles_data):
    """
    Prepare state exactly as done in training.
    Must match the preprocessing in your training code.
    """
    if len(candles_data) < LOOKBACK_WINDOW_SIZE:
        return None
    
    # Get the last LOOKBACK_WINDOW_SIZE candles
    recent = candles_data[-LOOKBACK_WINDOW_SIZE:]
    
    # Convert to DataFrame (matching training format)
    df = pd.DataFrame(recent)
    
    # Extract features
    # IMPORTANT: Adjust this based on your actual features in cryptoanalysis_data.csv
    # Your training code uses all columns except 'Close' and 'Date'
    features_list = []
    
    for _, row in df.iterrows():
        # Basic OHLCV features
        features = [
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            float(row['volume'])
        ]
        
        # Add more features here if your training data has them
        # Example: RSI, MACD, moving averages, etc.
        # For now, we'll pad with zeros to match expected feature count
        # ADJUST THIS based on your actual feature engineering
        
        features_list.append(features)
    
    # Convert to numpy array
    state = np.array(features_list)
    
    # Normalize (same as training)
    # In training: normalized_df = (df - df_min) / (df_max - df_min)
    # For simplicity, we'll use min-max normalization on the current window
    state_min = state.min()
    state_max = state.max()
    if state_max - state_min > 0:
        state = (state - state_min) / (state_max - state_min)
    
    # Reshape for model: (1, lookback_window, features)
    state = state.reshape(1, LOOKBACK_WINDOW_SIZE, -1)
    
    return state

def get_model_action(state):
    """
    Get trading action from the Actor model.
    Returns: 0 (Hold), 1 (Buy), 2 (Sell)
    """
    if actor_model is None or state is None:
        print("⚠ Model not loaded or state invalid - defaulting to HOLD")
        return 0
    
    try:
        # Get prediction from actor model
        prediction = actor_model.predict(state, verbose=0)
        
        # Get the action with highest probability
        action = np.argmax(prediction[0])
        
        # Get confidence scores
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
    print("   (Real trading disabled)")
    
    # Log the simulated order
    timestamp = int(time.time() * 1000)
    order_data = [
        f"sim_{timestamp}",  # id
        symbol,              # symbol
        0,                   # market_price (to be filled)
        quantity,            # qty
        timestamp,           # timestamp
        side,                # side
        0,                   # cum_market_price
        0,                   # fee
        0.075,              # fee_percent
        0,                   # price_with_fee
        '---',              # profits
        '---'               # profits_percent
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
    get_data_onStart()
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
    Process incoming WebSocket messages and make trading decisions.
    """
    global recent_candles
    
    try:
        json_message = json.loads(message)
        candle = json_message['k']
        is_candle_closed = candle['x']
        
        # Store candle data
        candle_data = {
            'open': candle['o'],
            'high': candle['h'],
            'low': candle['l'],
            'close': candle['c'],
            'volume': candle['v'],
            'timestamp': candle['t']
        }
        
        # Print live updates
        print(f"\r💹 Live: Close={candle_data['close']} | "
              f"Vol={float(candle_data['volume']):.2f} | "
              f"Candles={len(recent_candles)}/{LOOKBACK_WINDOW_SIZE}", end='')
        
        if is_candle_closed:
            print()  # New line
            print("\n" + "-"*60)
            print(f"📊 CANDLE CLOSED at {candle_data['close']}")
            print("-"*60)
            
            recent_candles.append(candle_data)
            
            # Keep only the last MAX_CANDLES
            if len(recent_candles) > MAX_CANDLES:
                recent_candles.pop(0)
            
            # Wait until we have enough data
            if len(recent_candles) < LOOKBACK_WINDOW_SIZE:
                print(f"⏳ Collecting data... ({len(recent_candles)}/{LOOKBACK_WINDOW_SIZE} candles)")
                return
            
            # Prepare state for model
            state = prepare_state(recent_candles)
            
            if state is not None:
                print(f"✓ State prepared: shape={state.shape}")
                
                # Get action from model
                action = get_model_action(state)
                
                # Read transaction history to get last position
                try:
                    df = pd.read_csv("../transaction_history.csv")
                    if len(df) > 0:
                        limited_data = df.tail(10).iloc[::-1]
                        last_side = limited_data['side'].iloc[0]
                    else:
                        last_side = "SELL"  # Default to allow first buy
                except Exception as e:
                    print(f"⚠ Could not read transaction history: {e}")
                    last_side = "SELL"
                
                print(f"📍 Current Position: {last_side}")
                
                # Execute trades based on model action
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
    
    # Load the model
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
    
    # Create and run WebSocket
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