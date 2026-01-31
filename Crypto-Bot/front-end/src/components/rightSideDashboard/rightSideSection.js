import React, { useState, useEffect } from "react"
import * as Icon from "react-cryptocoins"
import moment from "moment"
import styled from 'styled-components'
import axios from 'axios'

const API_URL = "http://localhost:5000"

const cryptoIcons = [
  { symbol: "ETH", icon: <Icon.Eth color="black" /> },
  { symbol: "LTC", icon: <Icon.Ltc color="black" /> },
  { symbol: "XRP", icon: <Icon.Xrp color="black" /> },
  { symbol: "BTC", icon: <Icon.Btc color="black" /> },
  { symbol: "USDT", icon: <Icon.Usdt color="black" /> },
  { symbol: "ADA", icon: <Icon.Ada color="black" /> },
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
  padding: 10px;
  background-color: ${props => props.running ? '#d4edda' : '#f8d7da'};
  border-radius: 8px;
  margin-bottom: 10px;
  
  &::before {
    content: '';
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background-color: ${props => props.running ? '#28a745' : '#dc3545'};
    margin-right: 10px;
    animation: ${props => props.running ? 'pulse 2s infinite' : 'none'};
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
`

const ErrorBanner = styled.div`
  padding: 10px;
  background-color: #f8d7da;
  border: 1px solid #f5c6cb;
  border-radius: 8px;
  color: #721c24;
  margin-bottom: 15px;
  text-align: center;
  font-size: 0.9rem;
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

  // Fetch bot status
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

  // Fetch statistics - MAINTENANT CORRIGÉ ✅
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
      // Ne pas afficher d'erreur si c'est juste les stats qui manquent
      if (error.response?.status === 500) {
        console.warn('Statistics endpoint error (peut-être pas de données encore)')
      }
    }
  }

  // Fetch model prediction
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

  // Control bot (start/stop)
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

  // Check API health
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
    // Initial health check
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

    // Poll every 5 seconds
    const interval = setInterval(async () => {
      if (apiOnline) {
        fetchBotStatus()
        fetchStatistics()
        fetchPrediction()
      } else {
        // Retry connection
        await checkApiHealth()
      }
    }, 5000)

    return () => clearInterval(interval)
  }, [apiOnline])

  if (loading) {
    return (
      <ResponsiveDiv>
        <div style={{
          backgroundColor: "#fff",
          borderRadius: "10px",
          padding: "40px",
          textAlign: "center"
        }}>
          <LoadingSpinner />
          <p style={{ color: "#777", marginTop: "20px" }}>
            Chargement des données du bot...
          </p>
        </div>
      </ResponsiveDiv>
    )
  }

  return (
    <ResponsiveDiv>
      {/* Bot Status Card */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          backgroundColor: "#fff",
          borderRadius: "10px",
          padding: "20px",
          boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
          marginBottom: "20px",
        }}
      >
        <p
          style={{
            color: "#333",
            marginTop: "0",
            fontSize: "1.5rem",
            fontWeight: "bold",
            fontFamily: "Arial, sans-serif",
          }}
        >
          Contrôle du Bot
        </p>
        
        {/* Error Banner */}
        {error && (
          <ErrorBanner>
            ⚠️ {error}
          </ErrorBanner>
        )}

        {/* API Status */}
        {!apiOnline && (
          <ErrorBanner>
            🔴 Serveur Flask non accessible - Vérifiez que le backend est démarré
          </ErrorBanner>
        )}
        
        <StatusIndicator running={botStatus.running}>
          <span style={{ fontWeight: 'bold', fontFamily: "Arial, sans-serif" }}>
            Statut: {botStatus.running ? 'EN MARCHE' : 'ARRÊTÉ'}
          </span>
        </StatusIndicator>

        <div
          style={{
            backgroundColor: "#e0e0e0",
            width: "100%",
            height: 2,
            marginBottom: "20px",
          }}
        ></div>

        {/* User Control Section */}
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
              backgroundColor: "#fff",
              border: "1px solid #ddd",
              color: "#333",
              width: "50%",
              fontSize: "1rem",
              borderRadius: "5px",
              padding: "8px",
              cursor: "pointer",
              fontFamily: "Arial, sans-serif",
              marginBottom: "15px"
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
              marginBottom: "20px",
            }}
          >
            <button
              onClick={() => controlBot('start')}
              disabled={botStatus.running || !apiOnline}
              style={{
                backgroundColor: (botStatus.running || !apiOnline) ? "#ccc" : "#28a745",
                borderRadius: "8px",
                width: "48%",
                padding: "12px",
                color: "white",
                border: "none",
                cursor: (botStatus.running || !apiOnline) ? "not-allowed" : "pointer",
                fontWeight: "bold",
                fontFamily: "Arial, sans-serif",
                fontSize: "1rem"
              }}
            >
              {botStatus.running ? "En marche..." : "Démarrer"}
            </button>
            <button
              onClick={() => controlBot('stop')}
              disabled={!botStatus.running || !apiOnline}
              style={{
                backgroundColor: (!botStatus.running || !apiOnline) ? "#ccc" : "#dc3545",
                borderRadius: "8px",
                width: "48%",
                padding: "12px",
                color: "white",
                border: "none",
                cursor: (!botStatus.running || !apiOnline) ? "not-allowed" : "pointer",
                fontWeight: "bold",
                fontFamily: "Arial, sans-serif",
                fontSize: "1rem"
              }}
            >
              Arrêter
            </button>
          </div>

          {/* Last Action */}
          <div style={{
            width: "100%",
            padding: "10px",
            backgroundColor: "#f8f9fa",
            borderRadius: "8px",
            marginBottom: "15px",
            textAlign: "center"
          }}>
            <div style={{ color: "#777", fontSize: "0.9rem", marginBottom: "5px" }}>
              Dernière Action
            </div>
            <div style={{
              fontSize: "1.3rem",
              fontWeight: "bold",
              color: botStatus.last_action === 'BUY' ? '#28a745' : 
                     botStatus.last_action === 'SELL' ? '#dc3545' : '#6c757d'
            }}>
              {botStatus.last_action}
            </div>
            <div style={{ color: "#777", fontSize: "0.8rem", marginTop: "5px" }}>
              Confiance: {(botStatus.confidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        <div
          style={{
            backgroundColor: "#e0e0e0",
            width: "100%",
            height: 2,
            margin: "15px 0",
          }}
        ></div>

        {/* Bot Statistics Section */}
        <div
          style={{
            color: "#333",
            display: "flex",
            flexDirection: "column",
            width: "100%",
            fontFamily: "Arial, sans-serif",
          }}
        >
          <h3 style={{ margin: "10px 0", textAlign: "center" }}>Statistiques de Performance</h3>
          
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "15px", padding: "10px" }}>
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
            backgroundColor: "#e0e0e0",
            width: "100%",
            height: 2,
            margin: "15px 0",
          }}
        ></div>

        {/* Model Prediction */}
        <div style={{ width: "100%", padding: "10px" }}>
          <h3 style={{ margin: "10px 0", textAlign: "center", color: "#333" }}>
            Prédiction du Modèle
          </h3>
          
          <div style={{ marginTop: "15px" }}>
            <PredictionBar label="HOLD" value={prediction.hold} color="#6c757d" />
            <PredictionBar label="BUY" value={prediction.buy} color="#28a745" />
            <PredictionBar label="SELL" value={prediction.sell} color="#dc3545" />
          </div>
        </div>
      </div>
    </ResponsiveDiv>
  )
}

// Helper Components
const StatCard = ({ label, value, positive }) => (
  <div
    style={{
      padding: "15px",
      backgroundColor: "#f8f9fa",
      borderRadius: "8px",
      textAlign: "center",
      border: "1px solid #e0e0e0"
    }}
  >
    <div style={{ color: "#777", fontSize: "0.85rem", marginBottom: "5px" }}>
      {label}
    </div>
    <div
      style={{
        fontSize: "1.2rem",
        fontWeight: "bold",
        color: positive !== undefined 
          ? (positive ? "#28a745" : "#dc3545")
          : "#333"
      }}
    >
      {value}
    </div>
  </div>
)

const PredictionBar = ({ label, value, color }) => (
  <div style={{ marginBottom: "12px" }}>
    <div style={{ 
      display: "flex", 
      justifyContent: "space-between", 
      marginBottom: "5px",
      fontSize: "0.9rem"
    }}>
      <span style={{ fontWeight: "bold" }}>{label}</span>
      <span>{(value * 100).toFixed(1)}%</span>
    </div>
    <div style={{
      width: "100%",
      height: "20px",
      backgroundColor: "#e0e0e0",
      borderRadius: "10px",
      overflow: "hidden"
    }}>
      <div style={{
        width: `${value * 100}%`,
        height: "100%",
        backgroundColor: color,
        transition: "width 0.3s ease"
      }}></div>
    </div>
  </div>
)

export default RightSideSection