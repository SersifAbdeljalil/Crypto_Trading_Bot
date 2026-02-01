import React, { useState, useEffect } from 'react'
import styled from 'styled-components'
import axios from 'axios'

const API_URL = "http://localhost:5000"

// Ethereum-inspired color palette
const COLORS = {
  ethereumPurple: '#627EEA',
  ethereumDark: '#1C1C3C',
  ethereumLight: '#8A92B2',
  blockchainBlue: '#00D4FF',
  cryptoGreen: '#10B981',
  cryptoRed: '#EF4444',
  goldAccent: '#F59E0B',
  darkBg: '#0F0F23',
  cardBg: '#1A1B3A',
  surface: '#252641',
  border: '#2D2E4E',
  textPrimary: '#E5E7EB',
  textSecondary: '#9CA3AF',
}

const Card = styled.div`
  padding: 24px;
  background-color: ${COLORS.cardBg};
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
  margin-bottom: 20px;
  border: 1px solid ${COLORS.border};
`

const Title = styled.h2`
  margin-bottom: 24px;
  color: ${COLORS.textPrimary};
  font-size: 1.8rem;
  background: linear-gradient(135deg, ${COLORS.ethereumPurple}, ${COLORS.blockchainBlue});
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
`

const IndicatorGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
  margin-bottom: 20px;
`

const IndicatorCard = styled.div`
  padding: 16px;
  background-color: ${COLORS.surface};
  border-radius: 12px;
  border-left: 4px solid ${props => props.color || COLORS.ethereumPurple};
  border: 1px solid ${COLORS.border};
  border-left: 4px solid ${props => props.color || COLORS.ethereumPurple};
  transition: transform 0.2s, box-shadow 0.2s;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
  }
`

const IndicatorLabel = styled.div`
  color: ${COLORS.textSecondary};
  font-size: 0.85rem;
  margin-bottom: 8px;
  font-weight: 500;
`

const IndicatorValue = styled.div`
  color: ${COLORS.textPrimary};
  font-size: 1.4rem;
  font-weight: bold;
`

const SentimentBadge = styled.div`
  display: inline-block;
  padding: 10px 20px;
  border-radius: 24px;
  font-weight: bold;
  font-size: 1.1rem;
  background: ${props => {
    if (props.sentiment === 'VERY BULLISH') return `linear-gradient(135deg, ${COLORS.cryptoGreen}, #059669)`;
    if (props.sentiment === 'BULLISH') return `linear-gradient(135deg, #10B981, #34D399)`;
    if (props.sentiment === 'VERY BEARISH') return `linear-gradient(135deg, ${COLORS.cryptoRed}, #DC2626)`;
    if (props.sentiment === 'BEARISH') return `linear-gradient(135deg, #EF4444, #F87171)`;
    return COLORS.surface;
  }};
  color: white;
  margin: 10px 0;
  box-shadow: ${props => {
    if (props.sentiment.includes('BULLISH')) return `0 4px 16px rgba(16, 185, 129, 0.3)`;
    if (props.sentiment.includes('BEARISH')) return `0 4px 16px rgba(239, 68, 68, 0.3)`;
    return 'none';
  }};
`

const SignalChip = styled.div`
  display: inline-block;
  padding: 8px 14px;
  margin: 5px;
  border-radius: 20px;
  font-size: 0.85rem;
  background-color: ${props => props.type === 'buy' ? 'rgba(16, 185, 129, 0.15)' : 
                               props.type === 'sell' ? 'rgba(239, 68, 68, 0.15)' : COLORS.surface};
  color: ${props => props.type === 'buy' ? COLORS.cryptoGreen : 
                    props.type === 'sell' ? COLORS.cryptoRed : COLORS.textSecondary};
  border: 1px solid ${props => props.type === 'buy' ? COLORS.cryptoGreen : 
                               props.type === 'sell' ? COLORS.cryptoRed : COLORS.border};
`

const LoadingSpinner = styled.div`
  border: 4px solid ${COLORS.surface};
  border-top: 4px solid ${COLORS.ethereumPurple};
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
    const interval = setInterval(fetchIndicators, 30000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <Card>
        <Title>Technical Indicators</Title>
        <LoadingSpinner />
        <div style={{ textAlign: 'center', color: COLORS.textSecondary }}>Loading indicators...</div>
      </Card>
    )
  }

  if (error) {
    return (
      <Card>
        <Title>Technical Indicators</Title>
        <div style={{ 
          padding: '20px', 
          backgroundColor: 'rgba(239, 68, 68, 0.1)', 
          borderRadius: '12px',
          color: COLORS.cryptoRed,
          textAlign: 'center',
          border: `1px solid ${COLORS.cryptoRed}`
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

      {sentiment && (
        <div style={{ marginBottom: '24px', textAlign: 'center' }}>
          <SentimentBadge sentiment={sentiment.sentiment}>
            {sentiment.sentiment}
          </SentimentBadge>
          <div style={{ color: COLORS.textSecondary, fontSize: '0.95rem', marginTop: '8px' }}>
            Confidence: {sentiment.confidence}%
          </div>
          <div style={{ marginTop: '12px' }}>
            {sentiment.reasons && sentiment.reasons.map((reason, idx) => (
              <div key={idx} style={{ 
                color: COLORS.textSecondary, 
                fontSize: '0.85rem',
                margin: '4px 0'
              }}>
                • {reason}
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ 
        backgroundColor: COLORS.border, 
        width: '100%', 
        height: 1, 
        margin: '24px 0' 
      }}></div>

      <div style={{ textAlign: 'center', marginBottom: '24px' }}>
        <div style={{ color: COLORS.textSecondary, fontSize: '1rem', marginBottom: '8px' }}>Current Price</div>
        <div style={{ 
          fontSize: '2.5rem', 
          fontWeight: 'bold', 
          background: `linear-gradient(135deg, ${COLORS.ethereumPurple}, ${COLORS.blockchainBlue})`,
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent'
        }}>
          ${indicators.current_price?.toFixed(2) || 'N/A'}
        </div>
      </div>

      <IndicatorGrid>
        {indicators.rsi && (
          <IndicatorCard color={
            indicators.rsi > 70 ? COLORS.cryptoRed : 
            indicators.rsi < 30 ? COLORS.cryptoGreen : COLORS.ethereumPurple
          }>
            <IndicatorLabel>RSI (14)</IndicatorLabel>
            <IndicatorValue>{indicators.rsi.toFixed(2)}</IndicatorValue>
            {indicators.rsi_overbought && <div style={{ color: COLORS.cryptoRed, fontSize: '0.75rem', marginTop: '4px' }}>Overbought</div>}
            {indicators.rsi_oversold && <div style={{ color: COLORS.cryptoGreen, fontSize: '0.75rem', marginTop: '4px' }}>Oversold</div>}
          </IndicatorCard>
        )}

        {indicators.macd && (
          <>
            <IndicatorCard color={COLORS.goldAccent}>
              <IndicatorLabel>MACD</IndicatorLabel>
              <IndicatorValue>{indicators.macd.macd.toFixed(2)}</IndicatorValue>
            </IndicatorCard>
            <IndicatorCard color={COLORS.ethereumLight}>
              <IndicatorLabel>Signal</IndicatorLabel>
              <IndicatorValue>{indicators.macd.signal.toFixed(2)}</IndicatorValue>
            </IndicatorCard>
            <IndicatorCard color={indicators.macd.histogram > 0 ? COLORS.cryptoGreen : COLORS.cryptoRed}>
              <IndicatorLabel>Histogram</IndicatorLabel>
              <IndicatorValue>{indicators.macd.histogram.toFixed(2)}</IndicatorValue>
            </IndicatorCard>
          </>
        )}

        {indicators.ema_9 && (
          <IndicatorCard color={COLORS.blockchainBlue}>
            <IndicatorLabel>EMA 9</IndicatorLabel>
            <IndicatorValue>${indicators.ema_9.toFixed(2)}</IndicatorValue>
          </IndicatorCard>
        )}
        {indicators.ema_21 && (
          <IndicatorCard color={COLORS.blockchainBlue}>
            <IndicatorLabel>EMA 21</IndicatorLabel>
            <IndicatorValue>${indicators.ema_21.toFixed(2)}</IndicatorValue>
          </IndicatorCard>
        )}
        {indicators.ema_50 && (
          <IndicatorCard color={COLORS.blockchainBlue}>
            <IndicatorLabel>EMA 50</IndicatorLabel>
            <IndicatorValue>${indicators.ema_50.toFixed(2)}</IndicatorValue>
          </IndicatorCard>
        )}

        {indicators.sma_20 && (
          <IndicatorCard color={COLORS.ethereumPurple}>
            <IndicatorLabel>SMA 20</IndicatorLabel>
            <IndicatorValue>${indicators.sma_20.toFixed(2)}</IndicatorValue>
          </IndicatorCard>
        )}
        {indicators.sma_50 && (
          <IndicatorCard color={COLORS.ethereumPurple}>
            <IndicatorLabel>SMA 50</IndicatorLabel>
            <IndicatorValue>${indicators.sma_50.toFixed(2)}</IndicatorValue>
          </IndicatorCard>
        )}
      </IndicatorGrid>

      {indicators.bollinger_bands && (
        <div style={{ marginTop: '24px' }}>
          <h3 style={{ 
            color: COLORS.textPrimary, 
            fontSize: '1.2rem', 
            marginBottom: '16px',
            background: `linear-gradient(135deg, ${COLORS.ethereumPurple}, ${COLORS.blockchainBlue})`,
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent'
          }}>
            Bollinger Bands
          </h3>
          <div style={{ display: 'flex', justifyContent: 'space-around', gap: '12px' }}>
            <div style={{ 
              textAlign: 'center', 
              flex: 1,
              padding: '16px',
              backgroundColor: COLORS.surface,
              borderRadius: '12px',
              border: `1px solid ${COLORS.border}`
            }}>
              <div style={{ color: COLORS.textSecondary, fontSize: '0.85rem', marginBottom: '8px' }}>Upper</div>
              <div style={{ fontWeight: 'bold', color: COLORS.cryptoRed, fontSize: '1.2rem' }}>
                ${indicators.bollinger_bands.upper.toFixed(2)}
              </div>
            </div>
            <div style={{ 
              textAlign: 'center', 
              flex: 1,
              padding: '16px',
              backgroundColor: COLORS.surface,
              borderRadius: '12px',
              border: `1px solid ${COLORS.border}`
            }}>
              <div style={{ color: COLORS.textSecondary, fontSize: '0.85rem', marginBottom: '8px' }}>Middle</div>
              <div style={{ fontWeight: 'bold', color: COLORS.ethereumPurple, fontSize: '1.2rem' }}>
                ${indicators.bollinger_bands.middle.toFixed(2)}
              </div>
            </div>
            <div style={{ 
              textAlign: 'center', 
              flex: 1,
              padding: '16px',
              backgroundColor: COLORS.surface,
              borderRadius: '12px',
              border: `1px solid ${COLORS.border}`
            }}>
              <div style={{ color: COLORS.textSecondary, fontSize: '0.85rem', marginBottom: '8px' }}>Lower</div>
              <div style={{ fontWeight: 'bold', color: COLORS.cryptoGreen, fontSize: '1.2rem' }}>
                ${indicators.bollinger_bands.lower.toFixed(2)}
              </div>
            </div>
          </div>
        </div>
      )}

      {indicators.signals && (
        <div style={{ marginTop: '24px' }}>
          <h3 style={{ 
            color: COLORS.textPrimary, 
            fontSize: '1.2rem', 
            marginBottom: '16px',
            background: `linear-gradient(135deg, ${COLORS.ethereumPurple}, ${COLORS.blockchainBlue})`,
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent'
          }}>
            Trading Signals
          </h3>
          
          {indicators.signals.buy_signals && indicators.signals.buy_signals.length > 0 && (
            <div style={{ marginBottom: '12px' }}>
              <div style={{ color: COLORS.cryptoGreen, fontWeight: 'bold', marginBottom: '8px' }}>
                🟢 Buy Signals:
              </div>
              {indicators.signals.buy_signals.map((signal, idx) => (
                <SignalChip key={idx} type="buy">{signal}</SignalChip>
              ))}
            </div>
          )}

          {indicators.signals.sell_signals && indicators.signals.sell_signals.length > 0 && (
            <div style={{ marginBottom: '12px' }}>
              <div style={{ color: COLORS.cryptoRed, fontWeight: 'bold', marginBottom: '8px' }}>
                🔴 Sell Signals:
              </div>
              {indicators.signals.sell_signals.map((signal, idx) => (
                <SignalChip key={idx} type="sell">{signal}</SignalChip>
              ))}
            </div>
          )}

          {(!indicators.signals.buy_signals || indicators.signals.buy_signals.length === 0) &&
           (!indicators.signals.sell_signals || indicators.signals.sell_signals.length === 0) && (
            <div style={{ 
              textAlign: 'center', 
              color: COLORS.textSecondary, 
              padding: '16px',
              backgroundColor: COLORS.surface,
              borderRadius: '12px',
              border: `1px solid ${COLORS.border}`
            }}>
              No strong signals detected
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

export default TechnicalIndicatorsCard