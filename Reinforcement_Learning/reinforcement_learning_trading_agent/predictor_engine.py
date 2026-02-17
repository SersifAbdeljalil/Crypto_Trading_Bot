'''
Real-Time Ethereum Trading Predictor
Uses Binance API with keys from .env file
'''

import numpy as np
import pandas as pd
import os
import time
import logging
import sys
from datetime import datetime
from collections import deque
from typing import Optional, Dict, List
import requests
import hmac
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Load .env manually (no python-dotenv needed)
def load_env(env_path='.env'):
    if not os.path.exists(env_path):
        return
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, val = line.partition('=')
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key.strip(), val)

load_env()

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('realtime_predictions.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
class Config:
    LOOKBACK_WINDOW    = 50
    STATE_SIZE         = (50, 20)
    ACTION_SPACE       = 3
    MODEL_ARCHITECTURE = "CNN"

    INITIAL_BALANCE    = 1000.0
    TRADING_FEE        = 0.001   # Binance standard 0.1%

    BUY_THRESHOLD      = 0.40
    SELL_THRESHOLD     = 0.40
    STRONG_THRESHOLD   = 0.65

    FETCH_INTERVAL     = 60      # seconds
    SYMBOL             = "ETHUSDT"

    FEATURE_COLUMNS = [
        'Close', 'receive_count', 'sent_count', 'avg_fee',
        'blocksize', 'btchashrate', 'OIL', 'ecr20_transfers',
        'GOLD', 'searches', 'hashrate', 'marketcap', 'difficulty',
        's&p500', 'transactionfee', 'transactions', 'tweet_count',
        'unique_adresses', 'VIX', 'UVYX'
    ]


# ─────────────────────────────────────────────
#  BINANCE DATA FETCHER (with SSL fix)
# ─────────────────────────────────────────────
class BinanceDataFetcher:
    '''
    Fetches live ETH data from Binance.
    API key read from .env — never hardcoded.
    '''
    BASE_URL = "https://api.binance.com/api/v3"

    def __init__(self):
        self.api_key    = os.environ.get('API_KEY', '')
        self.api_secret = os.environ.get('API_SECRET', '')
        self.symbol     = Config.SYMBOL

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent':  'ETH-Predictor/1.0',
            'X-MBX-APIKEY': self.api_key
        })

        # SSL fix for Windows
        self.session.verify = True
        try:
            import certifi
            self.session.verify = certifi.where()
        except ImportError:
            pass

        if self.api_key:
            logger.info("Binance API key loaded from .env")
        else:
            logger.warning("No API key found in .env — using public endpoints only")

    def get_current_price(self) -> Optional[float]:
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/ticker/price",
                params={"symbol": self.symbol},
                timeout=10
            )
            resp.raise_for_status()
            return float(resp.json()['price'])
        except Exception as e:
            logger.error(f"Price fetch failed: {e}")
            return self._fallback_price()

    def get_klines(self, interval: str = "1h", limit: int = 100) -> Optional[pd.DataFrame]:
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/klines",
                params={"symbol": self.symbol, "interval": interval, "limit": limit},
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()

            df = pd.DataFrame(data, columns=[
                'timestamp','Open','High','Low','Close','Volume',
                'close_time','quote_volume','trades',
                'taker_base','taker_quote','ignore'
            ])
            df['Date']   = pd.to_datetime(df['timestamp'], unit='ms')
            df['Close']  = df['Close'].astype(float)
            df['Open']   = df['Open'].astype(float)
            df['High']   = df['High'].astype(float)
            df['Low']    = df['Low'].astype(float)
            df['Volume'] = df['Volume'].astype(float)

            logger.info(f"Fetched {len(df)} candles from Binance")
            return df[['Date','Open','High','Low','Close','Volume']]

        except Exception as e:
            logger.error(f"Klines fetch failed: {e}")
            return self._fallback_klines(interval, limit)

    def get_market_stats(self) -> Optional[Dict]:
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/ticker/24hr",
                params={"symbol": self.symbol},
                timeout=10
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Market stats failed: {e}")
            return None

    # ── CoinGecko fallback if Binance SSL fails ──────────────────────
    def _fallback_price(self) -> Optional[float]:
        logger.info("Trying CoinGecko fallback for price ...")
        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "ethereum", "vs_currencies": "usd"},
                timeout=15
            )
            return float(resp.json()['ethereum']['usd'])
        except Exception as e:
            logger.error(f"CoinGecko fallback failed: {e}")
            return None

    def _fallback_klines(self, interval: str, limit: int) -> Optional[pd.DataFrame]:
        logger.info("Trying CoinGecko fallback for OHLC ...")
        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/coins/ethereum/ohlc",
                params={"vs_currency": "usd", "days": "14"},
                timeout=15
            )
            data = resp.json()
            df = pd.DataFrame(data, columns=['timestamp','Open','High','Low','Close'])
            df['Date']   = pd.to_datetime(df['timestamp'], unit='ms')
            df['Volume'] = 0.0
            for col in ['Open','High','Low','Close']:
                df[col] = df[col].astype(float)
            logger.info(f"CoinGecko fallback: {len(df)} candles")
            return df[['Date','Open','High','Low','Close','Volume']].tail(limit)
        except Exception as e:
            logger.error(f"CoinGecko fallback failed: {e}")
            return None


# ─────────────────────────────────────────────
#  FEATURE BUILDER
# ─────────────────────────────────────────────
class FeatureBuilder:
    def __init__(self, historical_csv: Optional[str] = None):
        self.global_min        = 0.0
        self.global_max        = 1.0
        self.feature_baselines = {}

        if historical_csv and os.path.exists(historical_csv):
            self._load_historical(historical_csv)
        else:
            logger.warning("No historical CSV. Using neutral baselines.")
            for col in Config.FEATURE_COLUMNS[1:]:
                self.feature_baselines[col] = 0.5

    def _load_historical(self, csv_path: str):
        try:
            df = pd.read_csv(csv_path, index_col=False)
            df = df.rename(columns={'price': 'Close', 'date': 'Date'})
            df_feat = df.drop(['Close','Date'], axis=1, errors='ignore')
            df_feat = df_feat.clip(lower=df_feat.quantile(0.01),
                                   upper=df_feat.quantile(0.99), axis=1)
            self.global_max = float(df_feat.max().max())
            self.global_min = float(df_feat.min().min())
            last_rows = df_feat.tail(100)
            for col in Config.FEATURE_COLUMNS[1:]:
                if col in last_rows.columns:
                    self.feature_baselines[col] = self._norm(float(last_rows[col].median()))
                else:
                    self.feature_baselines[col] = 0.5
            logger.info(f"Historical CSV loaded: {len(df)} rows")
        except Exception as e:
            logger.warning(f"Could not load CSV: {e}")
            for col in Config.FEATURE_COLUMNS[1:]:
                self.feature_baselines[col] = 0.5

    def _norm(self, value: float) -> float:
        r = self.global_max - self.global_min
        if r == 0:
            return 0.0
        return float(np.clip((value - self.global_min) / r, 0.0, 1.0))

    def bl(self, key: str) -> float:
        return self.feature_baselines.get(key, 0.5)

    def build_feature_vector(self, price_df: pd.DataFrame, market_stats=None) -> Optional[np.ndarray]:
        if price_df is None or len(price_df) < 5:
            return None

        latest    = price_df.iloc[-1]
        close     = float(latest['Close'])
        volume    = float(latest['Volume'])
        vol_sma   = price_df['Volume'].tail(10).mean() + 1e-9
        vol_ratio = volume / vol_sma

        prev      = float(price_df.iloc[-2]['Close']) if len(price_df) >= 2 else close
        chg_1h    = (close - prev) / (prev + 1e-9)
        prev24    = float(price_df.iloc[-24]['Close']) if len(price_df) >= 24 else close
        chg_24h   = (close - prev24) / (prev24 + 1e-9)

        if market_stats:
            mktcap = float(market_stats.get('quoteVolume', close * 120_000_000))
        else:
            mktcap = close * 120_000_000

        raw = {
            'Close':           close,
            'receive_count':   self.bl('receive_count') * (1 + vol_ratio * 0.05),
            'sent_count':      self.bl('sent_count')    * (1 + vol_ratio * 0.05),
            'avg_fee':         self.bl('avg_fee'),
            'blocksize':       self.bl('blocksize'),
            'btchashrate':     self.bl('btchashrate'),
            'OIL':             self.bl('OIL'),
            'ecr20_transfers': self.bl('ecr20_transfers') * (1 + abs(chg_1h)),
            'GOLD':            self.bl('GOLD'),
            'searches':        self.bl('searches') * (1 + abs(chg_24h) * 2),
            'hashrate':        self.bl('hashrate'),
            'marketcap':       mktcap,
            'difficulty':      self.bl('difficulty'),
            's&p500':          self.bl('s&p500'),
            'transactionfee':  self.bl('transactionfee') * (1 + vol_ratio * 0.02),
            'transactions':    self.bl('transactions')   * (1 + vol_ratio * 0.05),
            'tweet_count':     self.bl('tweet_count')    * (1 + abs(chg_24h) * 3),
            'unique_adresses': self.bl('unique_adresses'),
            'VIX':             self.bl('VIX'),
            'UVYX':            self.bl('UVYX'),
        }

        fv = []
        for col in Config.FEATURE_COLUMNS:
            val = raw[col]
            fv.append(float(val) if col == 'Close' else self._norm(val))
        return np.array(fv, dtype=np.float32)


# ─────────────────────────────────────────────
#  REAL-TIME PREDICTOR
# ─────────────────────────────────────────────
class RealTimePredictor:
    ACTION_LABELS = {0: "HOLD", 1: "BUY", 2: "SELL"}

    def __init__(self, model_folder, model_score, model_name="Crypto_trader", historical_csv=None):
        self.model_folder       = model_folder
        self.model_score        = model_score
        self.model_name         = model_name
        self.state_buffer       = deque(maxlen=Config.LOOKBACK_WINDOW)
        self.prediction_history = []
        self.trade_log          = []
        self.fetcher  = BinanceDataFetcher()
        self.builder  = FeatureBuilder(historical_csv=historical_csv)
        self.actor    = None
        self.balance  = Config.INITIAL_BALANCE
        self.crypto_held = 0.0
        self.net_worth   = Config.INITIAL_BALANCE

        logger.info("RealTimePredictor initialized.")
        logger.info(f"  Model: {model_folder} / score={model_score}")

    def load_model(self):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, LSTM

        logger.info("Building Actor network ...")
        X_input = Input(Config.STATE_SIZE)

        if Config.MODEL_ARCHITECTURE == "CNN":
            X = Conv1D(64, kernel_size=6, padding="same", activation="tanh")(X_input)
            X = MaxPooling1D(pool_size=2)(X)
            X = Conv1D(32, kernel_size=3, padding="same", activation="tanh")(X)
            X = MaxPooling1D(pool_size=2)(X)
            X = Flatten()(X)
        elif Config.MODEL_ARCHITECTURE == "LSTM":
            X = LSTM(512, return_sequences=True)(X_input)
            X = LSTM(256)(X)
        else:
            X = Flatten()(X_input)
            X = Dense(512, activation="relu")(X)

        A = Dense(512, activation="relu")(X)
        A = Dense(256, activation="relu")(A)
        A = Dense(64,  activation="relu")(A)
        output = Dense(Config.ACTION_SPACE, activation="softmax")(A)
        self.actor = Model(inputs=X_input, outputs=output)

        weight_path = os.path.join(self.model_folder,
                                   f"{self.model_score}_{self.model_name}_Actor.weights.h5")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weights not found: {weight_path}")

        self.actor.load_weights(weight_path)
        logger.info(f"Model loaded: {weight_path}")

    def _warm_up_buffer(self) -> bool:
        logger.info(f"Warming up buffer ({Config.LOOKBACK_WINDOW} steps) ...")
        price_df     = self.fetcher.get_klines(interval="1h", limit=120)
        market_stats = self.fetcher.get_market_stats()

        if price_df is None or len(price_df) < Config.LOOKBACK_WINDOW:
            logger.error("Not enough candles for warm-up.")
            return False

        for i in range(len(price_df) - Config.LOOKBACK_WINDOW, len(price_df)):
            fv = self.builder.build_feature_vector(price_df.iloc[:i+1], market_stats)
            if fv is not None:
                self.state_buffer.append(fv)

        ready = len(self.state_buffer) >= Config.LOOKBACK_WINDOW
        logger.info(f"Buffer ready: {ready} ({len(self.state_buffer)} steps)")
        return ready

    def _get_state(self):
        if len(self.state_buffer) < Config.LOOKBACK_WINDOW:
            return None
        return np.expand_dims(np.array(list(self.state_buffer), dtype=np.float32), axis=0)

    def predict_once(self) -> Optional[Dict]:
        if self.actor is None:
            self.load_model()

        price_df     = self.fetcher.get_klines(interval="1h", limit=60)
        market_stats = self.fetcher.get_market_stats()
        price        = self.fetcher.get_current_price()

        if price_df is None or price is None:
            logger.error("Failed to fetch live data.")
            return None

        fv = self.builder.build_feature_vector(price_df, market_stats)
        if fv is None:
            return None

        self.state_buffer.append(fv)
        state = self._get_state()
        if state is None:
            logger.warning("Buffer not full yet ...")
            return None

        probs     = self.actor.predict(state, verbose=0)[0]
        hold_prob = float(probs[0])
        buy_prob  = float(probs[1])
        sell_prob = float(probs[2])

        if buy_prob >= Config.BUY_THRESHOLD and buy_prob >= sell_prob:
            action = 1
        elif sell_prob >= Config.SELL_THRESHOLD and sell_prob > buy_prob:
            action = 2
        else:
            action = 0

        confidence = float(probs[action])
        is_strong  = confidence >= Config.STRONG_THRESHOLD

        self._update_portfolio(action, price)

        result = {
            "timestamp":           datetime.now().isoformat(),
            "price":               price,
            "action":              action,
            "action_label":        self.ACTION_LABELS[action],
            "probabilities": {
                "HOLD": round(hold_prob * 100, 2),
                "BUY":  round(buy_prob  * 100, 2),
                "SELL": round(sell_prob * 100, 2),
            },
            "confidence":          round(confidence * 100, 2),
            "signal_strength":     "STRONG" if is_strong else "normal",
            "is_strong_signal":    is_strong,
            "virtual_balance":     round(self.balance, 2),
            "virtual_crypto_held": round(self.crypto_held, 6),
            "virtual_net_worth":   round(self.net_worth, 2),
            "virtual_pnl":         round(self.net_worth - Config.INITIAL_BALANCE, 2),
            "virtual_pnl_pct":     round(
                (self.net_worth - Config.INITIAL_BALANCE) / Config.INITIAL_BALANCE * 100, 2),
        }

        self._print_result(result)
        self._save_result(result)
        self.prediction_history.append(result)
        return result

    def _update_portfolio(self, action, price):
        if action == 1 and self.balance > Config.INITIAL_BALANCE * 0.05:
            bought = (self.balance * (1 - Config.TRADING_FEE)) / price
            self.crypto_held += bought
            self.balance = 0.0
            self.trade_log.append({"time": datetime.now().isoformat(),
                                   "type": "BUY", "price": price, "amount": bought})
        elif action == 2 and self.crypto_held * price > Config.INITIAL_BALANCE * 0.05:
            proceeds = self.crypto_held * price * (1 - Config.TRADING_FEE)
            self.balance += proceeds
            self.crypto_held = 0.0
            self.trade_log.append({"time": datetime.now().isoformat(),
                                   "type": "SELL", "price": price, "proceeds": proceeds})
        self.net_worth = self.balance + self.crypto_held * price

    def _print_result(self, r):
        label  = r['action_label']
        price  = r['price']
        conf   = r['confidence']
        pnl    = r['virtual_pnl']
        pnl_p  = r['virtual_pnl_pct']
        strong = "  *** STRONG ***" if r['is_strong_signal'] else ""
        p      = r['probabilities']
        print("\n" + "=" * 55)
        print(f"  SIGNAL : {label}{strong}")
        print(f"  Price  : ${price:,.2f}")
        print(f"  Conf   : {conf:.1f}%")
        print(f"  BUY={p['BUY']:.1f}%  HOLD={p['HOLD']:.1f}%  SELL={p['SELL']:.1f}%")
        print(f"  P&L    : ${pnl:+,.2f}  ({pnl_p:+.2f}%)")
        print(f"  Worth  : ${r['virtual_net_worth']:,.2f}")
        print("=" * 55)

    def _save_result(self, r):
        try:
            row = {
                "timestamp":        r["timestamp"],
                "price":            r["price"],
                "action":           r["action_label"],
                "confidence_pct":   r["confidence"],
                "prob_buy":         r["probabilities"]["BUY"],
                "prob_hold":        r["probabilities"]["HOLD"],
                "prob_sell":        r["probabilities"]["SELL"],
                "signal_strength":  r["signal_strength"],
                "virtual_net_worth":r["virtual_net_worth"],
                "virtual_pnl":      r["virtual_pnl"],
            }
            log_file = "realtime_prediction_history.csv"
            write_header = not os.path.exists(log_file)
            pd.DataFrame([row]).to_csv(log_file, mode='a', header=write_header, index=False)
        except Exception as e:
            logger.warning(f"CSV save failed: {e}")

    def run(self, interval_seconds=Config.FETCH_INTERVAL, max_iterations=None,
            on_prediction=None, on_strong_signal=None):
        if self.actor is None:
            self.load_model()
        if not self._warm_up_buffer():
            logger.error("Warm-up failed. Check network.")
            return

        logger.info("Starting real-time prediction loop ...")
        logger.info(f"  Interval : {interval_seconds}s  |  Ctrl+C to stop")

        iteration = 0
        while True:
            try:
                result = self.predict_once()
                if result:
                    if on_prediction:
                        on_prediction(result)
                    if on_strong_signal and result['is_strong_signal']:
                        on_strong_signal(result)
                iteration += 1
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Max iterations reached ({max_iterations}).")
                    break
                logger.info(f"Next prediction in {interval_seconds}s ...")
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                logger.info("Stopped by user.")
                self._print_summary()
                break
            except Exception as e:
                logger.error(f"Error: {e}. Retrying ...")
                time.sleep(interval_seconds)

    def _print_summary(self):
        if not self.prediction_history:
            return
        n     = len(self.prediction_history)
        buys  = sum(1 for p in self.prediction_history if p['action_label'] == 'BUY')
        sells = sum(1 for p in self.prediction_history if p['action_label'] == 'SELL')
        holds = n - buys - sells
        pnl   = self.prediction_history[-1]['virtual_pnl']
        print("\n" + "=" * 55)
        print("  SESSION SUMMARY")
        print(f"  Predictions : {n}")
        print(f"  BUY         : {buys}")
        print(f"  SELL        : {sells}")
        print(f"  HOLD        : {holds}")
        print(f"  Virtual P&L : ${pnl:+,.2f}")
        print(f"  Trades      : {len(self.trade_log)}")
        print(f"  CSV log     : realtime_prediction_history.csv")
        print("=" * 55)

    def get_latest_prediction(self):
        return self.prediction_history[-1] if self.prediction_history else None

    def get_prediction_dataframe(self):
        return pd.DataFrame(self.prediction_history) if self.prediction_history else pd.DataFrame()