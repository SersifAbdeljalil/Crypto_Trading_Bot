import React, { useState, useEffect } from "react"
import * as Icon from "react-cryptocoins"
import moment from "moment"
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

const cryptoIcons = [
  { symbol: "ETH", icon: <Icon.Eth color={COLORS.ethereumPurple} /> },
  { symbol: "LTC", icon: <Icon.Ltc color={COLORS.blockchainBlue} /> },
  { symbol: "XRP", icon: <Icon.Xrp color={COLORS.blockchainBlue} /> },
  { symbol: "BTC", icon: <Icon.Btc color={COLORS.goldAccent} /> },
  { symbol: "USDT", icon: <Icon.Usdt color={COLORS.cryptoGreen} /> },
  { symbol: "ADA", icon: <Icon.Ada color={COLORS.ethereumPurple} /> },
]

const ResponsiveDiv = styled.div`
  padding: 10px;
  height: 100%;

  @media (max-width: 1500px) {
    padding: 5px;
    width: 700px;
  }
`

const StatusIndicator = styled.div`
  display: flex;
  align-items: center;
  padding: 12px 16px;
  background-color: ${props => props.running ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)'};
  border: 1px solid ${props => props.running ? COLORS.cryptoGreen : COLORS.cryptoRed};
  border-radius: 12px;
  margin-bottom: 15px;
  
  &::before {
    content: '';
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background-color: ${props => props.running ? COLORS.cryptoGreen : COLORS.cryptoRed};
    margin-right: 12px;
    animation: ${props => props.running ? 'pulse 2s infinite' : 'none'};
    box-shadow: ${props => props.running ? `0 0 10px ${COLORS.cryptoGreen}` : 'none'};
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
`

const ErrorBanner = styled.div`
  padding: 12px 16px;
  background-color: rgba(239, 68, 68, 0.1);
  border: 1px solid ${COLORS.cryptoRed};
  border-radius: 10px;
  color: ${COLORS.cryptoRed};
  margin-bottom: 15px;
  text-align: center;
  font-size: 0.9rem;
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

const RightSideSection = () => {
  const [symbol, setSymbol] = useState("ETHUSDT")
  const [botStatus, setBotStatus] = useState({
    running: false,
    last_action: "HOLD",
    confidence: 0,
    net_profit: 0,
    win_rate: 0,
    total_trades: 0
  })
  const [statistics, setStatistics] = useState({
    net_profit: 0,
    win_rate: 0,
    total_trades: 0,
    winning_trades: 0,
    losing_trades: 0,
    average_profit: 0,
    average_loss: 0,
    profit_factor: 0
  })
  const [prediction, setPrediction] = useState({
    hold: 0,
    buy: 0,
    sell: 0
  })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [apiOnline, setApiOnline] = useState(false)

  const fetchBotStatus = async () => {
    try {
      const response = await axios.get(`${API_URL}/bot_status`, { timeout: 5000 })
      setBotStatus(response.data)
      setApiOnline(true)
      setError(null)
    } catch (error) {
      console.error('Error fetching bot status:', error)
      setApiOnline(false)
      if (!error.response) {
        setError('API serveur non accessible. Vérifiez que Flask est démarré.')
      }
    }
  }

  const fetchStatistics = async () => {
    try {
      const response = await axios.get(`${API_URL}/statistics`, { timeout: 5000 })
      if (response.data.success) {
        setStatistics(response.data.statistics)
        setError(null)
      } else {
        console.error('Statistics request failed:', response.data)
      }
    } catch (error) {
      console.error('Error fetching statistics:', error)
      if (error.response?.status === 500) {
        console.warn('Statistics endpoint error (peut-être pas de données encore)')
      }
    }
  }

  const fetchPrediction = async () => {
    try {
      const response = await axios.get(`${API_URL}/model_prediction`, { timeout: 5000 })
      if (response.data.success) {
        setPrediction(response.data.prediction)
        setError(null)
      }
    } catch (error) {
      console.error('Error fetching prediction:', error)
    }
  }

  const controlBot = async (action) => {
    try {
      const response = await axios.post(`${API_URL}/bot_control`, { action }, { timeout: 5000 })
      if (response.data.success) {
        setBotStatus(response.data.status)
        alert(response.data.message)
        setError(null)
      }
    } catch (error) {
      console.error('Error controlling bot:', error)
      alert('Échec du contrôle du bot. Vérifiez que l\'API est en ligne.')
      setError('Impossible de contrôler le bot')
    }
  }

  const checkApiHealth = async () => {
    try {
      const response = await axios.get(`${API_URL}/`, { timeout: 3000 })
      if (response.data.status === 'online') {
        setApiOnline(true)
        setError(null)
        return true
      }
    } catch (error) {
      setApiOnline(false)
      setError('API Flask non accessible')
      return false
    }
  }

  useEffect(() => {
    const initializeData = async () => {
      setLoading(true)
      
      const isHealthy = await checkApiHealth()
      
      if (isHealthy) {
        await Promise.all([
          fetchBotStatus(),
          fetchStatistics(),
          fetchPrediction()
        ])
      }
      
      setLoading(false)
    }
    
    initializeData()

    const interval = setInterval(async () => {
      if (apiOnline) {
        fetchBotStatus()
        fetchStatistics()
        fetchPrediction()
      } else {
        await checkApiHealth()
      }
    }, 5000)

    return () => clearInterval(interval)
  }, [apiOnline])

  if (loading) {
    return (
      <ResponsiveDiv>
        <div style={{
          backgroundColor: COLORS.cardBg,
          borderRadius: "16px",
          padding: "40px",
          textAlign: "center",
          border: `1px solid ${COLORS.border}`
        }}>
          <LoadingSpinner />
          <p style={{ color: COLORS.textSecondary, marginTop: "20px" }}>
            Chargement des données du bot...
          </p>
        </div>
      </ResponsiveDiv>
    )
  }

  return (
    <ResponsiveDiv>
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          backgroundColor: COLORS.cardBg,
          borderRadius: "16px",
          padding: "24px",
          boxShadow: `0 8px 32px rgba(0, 0, 0, 0.4)`,
          marginBottom: "20px",
          border: `1px solid ${COLORS.border}`
        }}
      >
        <p
          style={{
            color: COLORS.textPrimary,
            marginTop: "0",
            fontSize: "1.8rem",
            fontWeight: "bold",
            fontFamily: "Arial, sans-serif",
            background: `linear-gradient(135deg, ${COLORS.ethereumPurple}, ${COLORS.blockchainBlue})`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          Contrôle du Bot
        </p>
        
        {error && <ErrorBanner>⚠️ {error}</ErrorBanner>}

        {!apiOnline && (
          <ErrorBanner>
            🔴 Serveur Flask non accessible - Vérifiez que le backend est démarré
          </ErrorBanner>
        )}
        
        <StatusIndicator running={botStatus.running}>
          <span style={{ fontWeight: 'bold', fontFamily: "Arial, sans-serif", color: COLORS.textPrimary }}>
            Statut: {botStatus.running ? 'EN MARCHE' : 'ARRÊTÉ'}
          </span>
        </StatusIndicator>

        <div
          style={{
            backgroundColor: COLORS.border,
            width: "100%",
            height: 1,
            marginBottom: "24px",
          }}
        ></div>

        <div
          style={{
            width: "100%",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}
        >
          <select
            style={{
              backgroundColor: COLORS.surface,
              border: `2px solid ${COLORS.border}`,
              color: COLORS.textPrimary,
              width: "50%",
              fontSize: "1rem",
              borderRadius: "10px",
              padding: "10px 12px",
              cursor: "pointer",
              fontFamily: "Arial, sans-serif",
              marginBottom: "20px"
            }}
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            disabled={!apiOnline}
          >
            <option value="BTCUSDT">BTC/USD</option>
            <option value="ETHUSDT">ETH/USD</option>
            <option value="XRPUSDT">XRP/USD</option>
          </select>

          <div
            style={{
              flexDirection: "row",
              display: "flex",
              justifyContent: "space-between",
              width: "80%",
              marginBottom: "24px",
              gap: "12px"
            }}
          >
            <button
              onClick={() => controlBot('start')}
              disabled={botStatus.running || !apiOnline}
              style={{
                background: (botStatus.running || !apiOnline) 
                  ? COLORS.surface 
                  : `linear-gradient(135deg, ${COLORS.cryptoGreen}, #059669)`,
                borderRadius: "10px",
                width: "48%",
                padding: "14px",
                color: (botStatus.running || !apiOnline) ? COLORS.textSecondary : "white",
                border: "none",
                cursor: (botStatus.running || !apiOnline) ? "not-allowed" : "pointer",
                fontWeight: "bold",
                fontFamily: "Arial, sans-serif",
                fontSize: "1rem",
                boxShadow: (botStatus.running || !apiOnline) ? "none" : `0 4px 16px rgba(16, 185, 129, 0.3)`
              }}
            >
              {botStatus.running ? "En marche..." : "🚀 Démarrer"}
            </button>
            <button
              onClick={() => controlBot('stop')}
              disabled={!botStatus.running || !apiOnline}
              style={{
                background: (!botStatus.running || !apiOnline) 
                  ? COLORS.surface 
                  : `linear-gradient(135deg, ${COLORS.cryptoRed}, #DC2626)`,
                borderRadius: "10px",
                width: "48%",
                padding: "14px",
                color: (!botStatus.running || !apiOnline) ? COLORS.textSecondary : "white",
                border: "none",
                cursor: (!botStatus.running || !apiOnline) ? "not-allowed" : "pointer",
                fontWeight: "bold",
                fontFamily: "Arial, sans-serif",
                fontSize: "1rem",
                boxShadow: (!botStatus.running || !apiOnline) ? "none" : `0 4px 16px rgba(239, 68, 68, 0.3)`
              }}
            >
              ⏹️ Arrêter
            </button>
          </div>

          <div style={{
            width: "100%",
            padding: "16px",
            backgroundColor: COLORS.surface,
            border: `1px solid ${COLORS.border}`,
            borderRadius: "12px",
            marginBottom: "20px",
            textAlign: "center"
          }}>
            <div style={{ color: COLORS.textSecondary, fontSize: "0.9rem", marginBottom: "8px" }}>
              Dernière Action
            </div>
            <div style={{
              fontSize: "1.5rem",
              fontWeight: "bold",
              color: botStatus.last_action === 'BUY' ? COLORS.cryptoGreen : 
                     botStatus.last_action === 'SELL' ? COLORS.cryptoRed : COLORS.textSecondary
            }}>
              {botStatus.last_action}
            </div>
            <div style={{ color: COLORS.textSecondary, fontSize: "0.85rem", marginTop: "8px" }}>
              Confiance: {(botStatus.confidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        <div
          style={{
            backgroundColor: COLORS.border,
            width: "100%",
            height: 1,
            margin: "20px 0",
          }}
        ></div>

        <div
          style={{
            color: COLORS.textPrimary,
            display: "flex",
            flexDirection: "column",
            width: "100%",
            fontFamily: "Arial, sans-serif",
          }}
        >
          <h3 style={{ 
            margin: "10px 0 20px 0", 
            textAlign: "center",
            background: `linear-gradient(135deg, ${COLORS.ethereumPurple}, ${COLORS.blockchainBlue})`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}>
            Statistiques de Performance
          </h3>
          
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", padding: "10px" }}>
            <StatCard 
              label="Profit Net" 
              value={`$${statistics.net_profit.toFixed(2)}`}
              positive={statistics.net_profit > 0}
            />
            <StatCard 
              label="Taux de Réussite" 
              value={`${statistics.win_rate.toFixed(1)}%`}
              positive={statistics.win_rate > 50}
            />
            <StatCard 
              label="Total Trades" 
              value={statistics.total_trades}
            />
            <StatCard 
              label="Trades Gagnants" 
              value={statistics.winning_trades}
              positive={true}
            />
            <StatCard 
              label="Profit Moyen" 
              value={`$${statistics.average_profit.toFixed(2)}`}
              positive={statistics.average_profit > 0}
            />
            <StatCard 
              label="Facteur de Profit" 
              value={statistics.profit_factor.toFixed(2)}
              positive={statistics.profit_factor > 1}
            />
          </div>
        </div>

        <div
          style={{
            backgroundColor: COLORS.border,
            width: "100%",
            height: 1,
            margin: "20px 0",
          }}
        ></div>

        <div style={{ width: "100%", padding: "10px" }}>
          <h3 style={{ 
            margin: "10px 0 20px 0", 
            textAlign: "center", 
            color: COLORS.textPrimary,
            background: `linear-gradient(135deg, ${COLORS.ethereumPurple}, ${COLORS.blockchainBlue})`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}>
            Prédiction du Modèle
          </h3>
          
          <div style={{ marginTop: "15px" }}>
            <PredictionBar label="HOLD" value={prediction.hold} color={COLORS.textSecondary} />
            <PredictionBar label="BUY" value={prediction.buy} color={COLORS.cryptoGreen} />
            <PredictionBar label="SELL" value={prediction.sell} color={COLORS.cryptoRed} />
          </div>
        </div>
      </div>
    </ResponsiveDiv>
  )
}

const StatCard = ({ label, value, positive }) => (
  <div
    style={{
      padding: "16px",
      backgroundColor: COLORS.surface,
      border: `1px solid ${COLORS.border}`,
      borderRadius: "12px",
      textAlign: "center",
    }}
  >
    <div style={{ color: COLORS.textSecondary, fontSize: "0.85rem", marginBottom: "8px" }}>
      {label}
    </div>
    <div
      style={{
        fontSize: "1.3rem",
        fontWeight: "bold",
        color: positive !== undefined 
          ? (positive ? COLORS.cryptoGreen : COLORS.cryptoRed)
          : COLORS.textPrimary
      }}
    >
      {value}
    </div>
  </div>
)

const PredictionBar = ({ label, value, color }) => (
  <div style={{ marginBottom: "16px" }}>
    <div style={{ 
      display: "flex", 
      justifyContent: "space-between", 
      marginBottom: "8px",
      fontSize: "0.95rem"
    }}>
      <span style={{ fontWeight: "bold", color: COLORS.textPrimary }}>{label}</span>
      <span style={{ color: COLORS.textSecondary }}>{(value * 100).toFixed(1)}%</span>
    </div>
    <div style={{
      width: "100%",
      height: "24px",
      backgroundColor: COLORS.surface,
      border: `1px solid ${COLORS.border}`,
      borderRadius: "12px",
      overflow: "hidden"
    }}>
      <div style={{
        width: `${value * 100}%`,
        height: "100%",
        background: `linear-gradient(90deg, ${color}, ${color}dd)`,
        transition: "width 0.3s ease",
        boxShadow: `0 0 12px ${color}66`
      }}></div>
    </div>
  </div>
)

export default RightSideSection