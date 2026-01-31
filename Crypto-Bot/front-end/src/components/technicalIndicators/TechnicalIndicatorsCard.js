import React, { useState, useEffect } from 'react'
import styled from 'styled-components'
import axios from 'axios'

const API_URL = "http://localhost:5000"

const Card = styled.div`
  padding: 20px;
  background-color: #ffffff;
  border-radius: 15px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
  margin-bottom: 20px;
`

const Title = styled.h2`
  margin-bottom: 20px;
  color: #333;
  font-size: 1.5rem;
`

const IndicatorGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 15px;
  margin-bottom: 20px;
`

const IndicatorCard = styled.div`
  padding: 15px;
  background-color: #f8f9fa;
  border-radius: 10px;
  border-left: 4px solid ${props => props.color || '#007AFF'};
  transition: transform 0.2s;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  }
`

const IndicatorLabel = styled.div`
  color: #777;
  font-size: 0.85rem;
  margin-bottom: 5px;
  font-weight: 500;
`

const IndicatorValue = styled.div`
  color: #333;
  font-size: 1.3rem;
  font-weight: bold;
`

const SentimentBadge = styled.div`
  display: inline-block;
  padding: 8px 16px;
  border-radius: 20px;
  font-weight: bold;
  font-size: 1.1rem;
  background-color: ${props => {
    if (props.sentiment === 'VERY BULLISH') return '#28a745';
    if (props.sentiment === 'BULLISH') return '#7cb342';
    if (props.sentiment === 'VERY BEARISH') return '#dc3545';
    if (props.sentiment === 'BEARISH') return '#e57373';
    return '#6c757d';
  }};
  color: white;
  margin: 10px 0;
`

const SignalChip = styled.div`
  display: inline-block;
  padding: 6px 12px;
  margin: 5px;
  border-radius: 15px;
  font-size: 0.85rem;
  background-color: ${props => props.type === 'buy' ? '#d4edda' : 
                               props.type === 'sell' ? '#f8d7da' : '#e2e3e5'};
  color: ${props => props.type === 'buy' ? '#155724' : 
                    props.type === 'sell' ? '#721c24' : '#383d41'};
  border: 1px solid ${props => props.type === 'buy' ? '#c3e6cb' : 
                               props.type === 'sell' ? '#f5c6cb' : '#d6d8db'};
`

const LoadingSpinner = styled.div`
  border: 4px solid #f3f3f3;
  border-top: 4px solid #007AFF;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;
  margin: 20px auto;

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`

const TechnicalIndicatorsCard = () => {
  const [indicators, setIndicators] = useState(null)
  const [sentiment, setSentiment] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchIndicators = async () => {
    try {
      const [indicatorsRes, sentimentRes] = await Promise.all([
        axios.get(`${API_URL}/technical_indicators`),
        axios.get(`${API_URL}/market_sentiment`)
      ])

      if (indicatorsRes.data.success) {
        setIndicators(indicatorsRes.data.indicators)
      }

      if (sentimentRes.data.success) {
        setSentiment(sentimentRes.data.sentiment)
      }

      setLoading(false)
      setError(null)
    } catch (err) {
      console.error('Error fetching indicators:', err)
      setError('Failed to fetch technical indicators. Make sure the API is running.')
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchIndicators()

    // Update every 30 seconds
    const interval = setInterval(fetchIndicators, 30000)

    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <Card>
        <Title>Technical Indicators</Title>
        <LoadingSpinner />
        <div style={{ textAlign: 'center', color: '#777' }}>Loading indicators...</div>
      </Card>
    )
  }

  if (error) {
    return (
      <Card>
        <Title>Technical Indicators</Title>
        <div style={{ 
          padding: '20px', 
          backgroundColor: '#f8d7da', 
          borderRadius: '8px',
          color: '#721c24',
          textAlign: 'center'
        }}>
          {error}
        </div>
      </Card>
    )
  }

  if (!indicators) {
    return null
  }

  return (
    <Card>
      <Title>Technical Indicators</Title>

      {/* Market Sentiment */}
      {sentiment && (
        <div style={{ marginBottom: '20px', textAlign: 'center' }}>
          <SentimentBadge sentiment={sentiment.sentiment}>
            {sentiment.sentiment}
          </SentimentBadge>
          <div style={{ color: '#777', fontSize: '0.9rem', marginTop: '5px' }}>
            Confidence: {sentiment.confidence}%
          </div>
          <div style={{ marginTop: '10px' }}>
            {sentiment.reasons && sentiment.reasons.map((reason, idx) => (
              <div key={idx} style={{ 
                color: '#555', 
                fontSize: '0.85rem',
                margin: '3px 0'
              }}>
                • {reason}
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ 
        backgroundColor: '#e0e0e0', 
        width: '100%', 
        height: 2, 
        margin: '20px 0' 
      }}></div>

      {/* Current Price */}
      <div style={{ textAlign: 'center', marginBottom: '20px' }}>
        <div style={{ color: '#777', fontSize: '1rem' }}>Current Price</div>
        <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#007AFF' }}>
          ${indicators.current_price?.toFixed(2) || 'N/A'}
        </div>
      </div>

      {/* Indicators Grid */}
      <IndicatorGrid>
        {/* RSI */}
        {indicators.rsi && (
          <IndicatorCard color={
            indicators.rsi > 70 ? '#dc3545' : 
            indicators.rsi < 30 ? '#28a745' : '#007AFF'
          }>
            <IndicatorLabel>RSI (14)</IndicatorLabel>
            <IndicatorValue>{indicators.rsi.toFixed(2)}</IndicatorValue>
            {indicators.rsi_overbought && <div style={{ color: '#dc3545', fontSize: '0.75rem' }}>Overbought</div>}
            {indicators.rsi_oversold && <div style={{ color: '#28a745', fontSize: '0.75rem' }}>Oversold</div>}
          </IndicatorCard>
        )}

        {/* MACD */}
        {indicators.macd && (
          <>
            <IndicatorCard color="#ff9800">
              <IndicatorLabel>MACD</IndicatorLabel>
              <IndicatorValue>{indicators.macd.macd.toFixed(2)}</IndicatorValue>
            </IndicatorCard>
            <IndicatorCard color="#9c27b0">
              <IndicatorLabel>Signal</IndicatorLabel>
              <IndicatorValue>{indicators.macd.signal.toFixed(2)}</IndicatorValue>
            </IndicatorCard>
            <IndicatorCard color={indicators.macd.histogram > 0 ? '#28a745' : '#dc3545'}>
              <IndicatorLabel>Histogram</IndicatorLabel>
              <IndicatorValue>{indicators.macd.histogram.toFixed(2)}</IndicatorValue>
            </IndicatorCard>
          </>
        )}

        {/* EMAs */}
        {indicators.ema_9 && (
          <IndicatorCard color="#2196f3">
            <IndicatorLabel>EMA 9</IndicatorLabel>
            <IndicatorValue>${indicators.ema_9.toFixed(2)}</IndicatorValue>
          </IndicatorCard>
        )}
        {indicators.ema_21 && (
          <IndicatorCard color="#03a9f4">
            <IndicatorLabel>EMA 21</IndicatorLabel>
            <IndicatorValue>${indicators.ema_21.toFixed(2)}</IndicatorValue>
          </IndicatorCard>
        )}
        {indicators.ema_50 && (
          <IndicatorCard color="#00bcd4">
            <IndicatorLabel>EMA 50</IndicatorLabel>
            <IndicatorValue>${indicators.ema_50.toFixed(2)}</IndicatorValue>
          </IndicatorCard>
        )}

        {/* SMAs */}
        {indicators.sma_20 && (
          <IndicatorCard color="#4caf50">
            <IndicatorLabel>SMA 20</IndicatorLabel>
            <IndicatorValue>${indicators.sma_20.toFixed(2)}</IndicatorValue>
          </IndicatorCard>
        )}
        {indicators.sma_50 && (
          <IndicatorCard color="#8bc34a">
            <IndicatorLabel>SMA 50</IndicatorLabel>
            <IndicatorValue>${indicators.sma_50.toFixed(2)}</IndicatorValue>
          </IndicatorCard>
        )}
      </IndicatorGrid>

      {/* Bollinger Bands */}
      {indicators.bollinger_bands && (
        <div style={{ marginTop: '20px' }}>
          <h3 style={{ color: '#333', fontSize: '1.1rem', marginBottom: '10px' }}>
            Bollinger Bands
          </h3>
          <div style={{ display: 'flex', justifyContent: 'space-around', gap: '10px' }}>
            <div style={{ textAlign: 'center', flex: 1 }}>
              <div style={{ color: '#777', fontSize: '0.8rem' }}>Upper</div>
              <div style={{ fontWeight: 'bold', color: '#dc3545' }}>
                ${indicators.bollinger_bands.upper.toFixed(2)}
              </div>
            </div>
            <div style={{ textAlign: 'center', flex: 1 }}>
              <div style={{ color: '#777', fontSize: '0.8rem' }}>Middle</div>
              <div style={{ fontWeight: 'bold', color: '#007AFF' }}>
                ${indicators.bollinger_bands.middle.toFixed(2)}
              </div>
            </div>
            <div style={{ textAlign: 'center', flex: 1 }}>
              <div style={{ color: '#777', fontSize: '0.8rem' }}>Lower</div>
              <div style={{ fontWeight: 'bold', color: '#28a745' }}>
                ${indicators.bollinger_bands.lower.toFixed(2)}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Trading Signals */}
      {indicators.signals && (
        <div style={{ marginTop: '20px' }}>
          <h3 style={{ color: '#333', fontSize: '1.1rem', marginBottom: '10px' }}>
            Trading Signals
          </h3>
          
          {indicators.signals.buy_signals && indicators.signals.buy_signals.length > 0 && (
            <div style={{ marginBottom: '10px' }}>
              <div style={{ color: '#28a745', fontWeight: 'bold', marginBottom: '5px' }}>
                🟢 Buy Signals:
              </div>
              {indicators.signals.buy_signals.map((signal, idx) => (
                <SignalChip key={idx} type="buy">{signal}</SignalChip>
              ))}
            </div>
          )}

          {indicators.signals.sell_signals && indicators.signals.sell_signals.length > 0 && (
            <div style={{ marginBottom: '10px' }}>
              <div style={{ color: '#dc3545', fontWeight: 'bold', marginBottom: '5px' }}>
                🔴 Sell Signals:
              </div>
              {indicators.signals.sell_signals.map((signal, idx) => (
                <SignalChip key={idx} type="sell">{signal}</SignalChip>
              ))}
            </div>
          )}

          {(!indicators.signals.buy_signals || indicators.signals.buy_signals.length === 0) &&
           (!indicators.signals.sell_signals || indicators.signals.sell_signals.length === 0) && (
            <div style={{ textAlign: 'center', color: '#777', padding: '10px' }}>
              No strong signals detected
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

export default TechnicalIndicatorsCard