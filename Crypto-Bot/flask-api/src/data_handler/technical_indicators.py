"""
Module d'indicateurs techniques pour le trading bot
Calcule RSI, MACD, Bollinger Bands, EMA, SMA en temps réel
"""
import pandas as pd
import numpy as np
from collections import deque
import requests

class TechnicalIndicators:
    """Classe pour calculer les indicateurs techniques en temps réel"""
    
    def __init__(self, symbol="ETHUSDT", interval="1m", lookback=100):
        """
        Initialise le calculateur d'indicateurs
        
        Args:
            symbol (str): Paire de trading (ex: ETHUSDT)
            interval (str): Intervalle des bougies (1m, 5m, 15m, 1h, etc.)
            lookback (int): Nombre de bougies historiques à conserver
        """
        self.symbol = symbol
        self.interval = interval
        self.lookback = lookback
        self.candles = deque(maxlen=lookback)
        
    def add_candle(self, candle_data):
        """
        Ajoute une nouvelle bougie à l'historique
        
        Args:
            candle_data (dict): Données de la bougie avec keys: open, high, low, close, volume
        """
        self.candles.append({
            'open': float(candle_data['open']),
            'high': float(candle_data['high']),
            'low': float(candle_data['low']),
            'close': float(candle_data['close']),
            'volume': float(candle_data['volume'])
        })
    
    def get_dataframe(self):
        """Convertit les bougies en DataFrame pandas"""
        if len(self.candles) == 0:
            return pd.DataFrame()
        return pd.DataFrame(list(self.candles))
    
    def calculate_rsi(self, period=14):
        """
        Calcule le RSI (Relative Strength Index)
        
        Args:
            period (int): Période de calcul (défaut: 14)
            
        Returns:
            float: Valeur RSI actuelle (0-100) ou None si pas assez de données
        """
        df = self.get_dataframe()
        if len(df) < period + 1:
            return None
        
        # Calculer les variations de prix
        delta = df['close'].diff()
        
        # Séparer les gains et les pertes
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculer les moyennes mobiles
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculer le RS et le RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1])
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        """
        Calcule le MACD (Moving Average Convergence Divergence)
        
        Args:
            fast (int): Période EMA rapide (défaut: 12)
            slow (int): Période EMA lente (défaut: 26)
            signal (int): Période ligne de signal (défaut: 9)
            
        Returns:
            dict: {'macd': float, 'signal': float, 'histogram': float} ou None
        """
        df = self.get_dataframe()
        if len(df) < slow:
            return None
        
        # Calculer les EMAs
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        # MACD = EMA rapide - EMA lente
        macd_line = ema_fast - ema_slow
        
        # Ligne de signal = EMA du MACD
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Histogramme = MACD - Signal
        histogram = macd_line - signal_line
        
        return {
            'macd': float(macd_line.iloc[-1]),
            'signal': float(signal_line.iloc[-1]),
            'histogram': float(histogram.iloc[-1])
        }
    
    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """
        Calcule les Bandes de Bollinger
        
        Args:
            period (int): Période de la moyenne mobile (défaut: 20)
            std_dev (float): Nombre d'écarts-types (défaut: 2)
            
        Returns:
            dict: {'upper': float, 'middle': float, 'lower': float} ou None
        """
        df = self.get_dataframe()
        if len(df) < period:
            return None
        
        # Calculer la moyenne mobile simple
        sma = df['close'].rolling(window=period).mean()
        
        # Calculer l'écart-type
        std = df['close'].rolling(window=period).std()
        
        # Calculer les bandes
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': float(upper_band.iloc[-1]),
            'middle': float(sma.iloc[-1]),
            'lower': float(lower_band.iloc[-1])
        }
    
    def calculate_ema(self, period=20):
        """
        Calcule l'EMA (Exponential Moving Average)
        
        Args:
            period (int): Période de calcul
            
        Returns:
            float: Valeur EMA actuelle ou None
        """
        df = self.get_dataframe()
        if len(df) < period:
            return None
        
        ema = df['close'].ewm(span=period, adjust=False).mean()
        return float(ema.iloc[-1])
    
    def calculate_sma(self, period=20):
        """
        Calcule la SMA (Simple Moving Average)
        
        Args:
            period (int): Période de calcul
            
        Returns:
            float: Valeur SMA actuelle ou None
        """
        df = self.get_dataframe()
        if len(df) < period:
            return None
        
        sma = df['close'].rolling(window=period).mean()
        return float(sma.iloc[-1])
    
    def calculate_volume_profile(self, bins=10):
        """
        Calcule le profil de volume
        
        Args:
            bins (int): Nombre de niveaux de prix
            
        Returns:
            dict: Distribution du volume par niveau de prix
        """
        df = self.get_dataframe()
        if len(df) < bins:
            return None
        
        # Créer des bins de prix
        price_bins = pd.cut(df['close'], bins=bins)
        
        # Calculer le volume par bin
        volume_profile = df.groupby(price_bins)['volume'].sum()
        
        return {
            'bins': [f"{interval.left:.2f}-{interval.right:.2f}" 
                    for interval in volume_profile.index],
            'volumes': volume_profile.values.tolist()
        }
    
    def get_all_indicators(self):
        """
        Calcule tous les indicateurs en une fois
        
        Returns:
            dict: Dictionnaire avec tous les indicateurs
        """
        current_price = float(self.candles[-1]['close']) if len(self.candles) > 0 else None
        
        indicators = {
            'current_price': current_price,
            'rsi': self.calculate_rsi(14),
            'rsi_overbought': self.calculate_rsi(14) > 70 if self.calculate_rsi(14) else None,
            'rsi_oversold': self.calculate_rsi(14) < 30 if self.calculate_rsi(14) else None,
            'macd': self.calculate_macd(),
            'bollinger_bands': self.calculate_bollinger_bands(),
            'ema_9': self.calculate_ema(9),
            'ema_21': self.calculate_ema(21),
            'ema_50': self.calculate_ema(50),
            'sma_20': self.calculate_sma(20),
            'sma_50': self.calculate_sma(50),
            'sma_200': self.calculate_sma(200),
        }
        
        # Ajouter des signaux de trading
        if indicators['rsi'] and indicators['macd']:
            indicators['signals'] = self._generate_signals(indicators)
        
        return indicators
    
    def _generate_signals(self, indicators):
        """
        Génère des signaux de trading basés sur les indicateurs
        
        Args:
            indicators (dict): Dictionnaire des indicateurs
            
        Returns:
            dict: Signaux de trading
        """
        signals = {
            'buy_signals': [],
            'sell_signals': [],
            'neutral': []
        }
        
        # Signal RSI
        if indicators['rsi']:
            if indicators['rsi'] < 30:
                signals['buy_signals'].append('RSI oversold')
            elif indicators['rsi'] > 70:
                signals['sell_signals'].append('RSI overbought')
        
        # Signal MACD
        macd = indicators.get('macd')
        if macd:
            if macd['histogram'] > 0 and macd['macd'] > macd['signal']:
                signals['buy_signals'].append('MACD bullish')
            elif macd['histogram'] < 0 and macd['macd'] < macd['signal']:
                signals['sell_signals'].append('MACD bearish')
        
        # Signal Bollinger Bands
        bb = indicators.get('bollinger_bands')
        current_price = indicators.get('current_price')
        if bb and current_price:
            if current_price < bb['lower']:
                signals['buy_signals'].append('Price below lower Bollinger Band')
            elif current_price > bb['upper']:
                signals['sell_signals'].append('Price above upper Bollinger Band')
        
        # Signal EMA Crossover
        ema_9 = indicators.get('ema_9')
        ema_21 = indicators.get('ema_21')
        if ema_9 and ema_21:
            if ema_9 > ema_21:
                signals['buy_signals'].append('EMA 9/21 golden cross')
            else:
                signals['sell_signals'].append('EMA 9/21 death cross')
        
        return signals
    
    def get_market_sentiment(self):
        """
        Calcule le sentiment général du marché
        
        Returns:
            dict: Sentiment avec score et raison
        """
        indicators = self.get_all_indicators()
        
        if not indicators['rsi'] or not indicators['macd']:
            return {
                'sentiment': 'NEUTRAL',
                'score': 0,
                'confidence': 0,
                'reason': 'Insufficient data'
            }
        
        score = 0
        reasons = []
        
        # RSI
        rsi = indicators['rsi']
        if rsi < 30:
            score += 2
            reasons.append('RSI oversold (bullish)')
        elif rsi < 40:
            score += 1
            reasons.append('RSI low (moderately bullish)')
        elif rsi > 70:
            score -= 2
            reasons.append('RSI overbought (bearish)')
        elif rsi > 60:
            score -= 1
            reasons.append('RSI high (moderately bearish)')
        
        # MACD
        macd = indicators['macd']
        if macd['histogram'] > 0:
            score += 1
            reasons.append('MACD bullish')
        else:
            score -= 1
            reasons.append('MACD bearish')
        
        # Bollinger Bands
        bb = indicators['bollinger_bands']
        current_price = indicators['current_price']
        if bb and current_price:
            bb_position = (current_price - bb['lower']) / (bb['upper'] - bb['lower'])
            if bb_position < 0.2:
                score += 1
                reasons.append('Price near lower BB (bullish)')
            elif bb_position > 0.8:
                score -= 1
                reasons.append('Price near upper BB (bearish)')
        
        # Déterminer le sentiment
        if score >= 3:
            sentiment = 'VERY BULLISH'
        elif score >= 1:
            sentiment = 'BULLISH'
        elif score <= -3:
            sentiment = 'VERY BEARISH'
        elif score <= -1:
            sentiment = 'BEARISH'
        else:
            sentiment = 'NEUTRAL'
        
        confidence = min(abs(score) * 20, 100)
        
        return {
            'sentiment': sentiment,
            'score': score,
            'confidence': confidence,
            'reasons': reasons
        }


def fetch_binance_klines(symbol="ETHUSDT", interval="1m", limit=100):
    """
    Récupère les bougies historiques depuis Binance
    
    Args:
        symbol (str): Paire de trading
        interval (str): Intervalle (1m, 5m, 15m, 1h, etc.)
        limit (int): Nombre de bougies à récupérer
        
    Returns:
        list: Liste de dictionnaires avec les données des bougies
    """
    url = f"https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        candles = []
        for candle in data:
            candles.append({
                'timestamp': candle[0],
                'open': candle[1],
                'high': candle[2],
                'low': candle[3],
                'close': candle[4],
                'volume': candle[5]
            })
        
        return candles
        
    except Exception as e:
        print(f"Error fetching Binance data: {e}")
        return []


# Test du module
if __name__ == "__main__":
    print("Testing Technical Indicators Module...")
    
    # Créer l'instance
    ti = TechnicalIndicators(symbol="ETHUSDT", interval="1m", lookback=100)
    
    # Récupérer les données historiques
    print("\nFetching historical data from Binance...")
    candles = fetch_binance_klines(symbol="ETHUSDT", interval="1m", limit=100)
    
    if candles:
        print(f"✓ Fetched {len(candles)} candles")
        
        # Ajouter les bougies
        for candle in candles:
            ti.add_candle(candle)
        
        # Calculer les indicateurs
        print("\nCalculating indicators...")
        indicators = ti.get_all_indicators()
        
        print(f"\n📊 Current Price: ${indicators['current_price']:.2f}")
        print(f"📈 RSI (14): {indicators['rsi']:.2f}" if indicators['rsi'] else "RSI: N/A")
        
        if indicators['macd']:
            print(f"📊 MACD: {indicators['macd']['macd']:.2f}")
            print(f"📊 Signal: {indicators['macd']['signal']:.2f}")
            print(f"📊 Histogram: {indicators['macd']['histogram']:.2f}")
        
        if indicators['bollinger_bands']:
            bb = indicators['bollinger_bands']
            print(f"📊 Bollinger Bands:")
            print(f"   Upper: ${bb['upper']:.2f}")
            print(f"   Middle: ${bb['middle']:.2f}")
            print(f"   Lower: ${bb['lower']:.2f}")
        
        # Sentiment du marché
        sentiment = ti.get_market_sentiment()
        print(f"\n🎯 Market Sentiment: {sentiment['sentiment']}")
        print(f"   Score: {sentiment['score']}")
        print(f"   Confidence: {sentiment['confidence']}%")
        print(f"   Reasons: {', '.join(sentiment['reasons'])}")
        
    else:
        print("✗ Failed to fetch data")