import React, { useState, useEffect, useRef } from "react"
import styled, { keyframes, css } from "styled-components"
import axios from "axios"
import { io } from "socket.io-client"

const API_URL = "http://localhost:5000"

const COLORS = {
  ethereumPurple: '#627EEA',
  blockchainBlue: '#00D4FF',
  cryptoGreen:    '#10B981',
  cryptoRed:      '#EF4444',
  goldAccent:     '#F59E0B',
  darkBg:         '#0F0F23',
  cardBg:         '#1A1B3A',
  surface:        '#252641',
  border:         '#2D2E4E',
  textPrimary:    '#E5E7EB',
  textSecondary:  '#9CA3AF',
}

// ─── Keyframes ─────────────────────────────────────────────────
const pulse = keyframes`
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.6; transform: scale(0.9); }
`
const spin = keyframes`
  0%   { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
`
const slideIn = keyframes`
  from { opacity: 0; transform: translateY(-8px); }
  to   { opacity: 1; transform: translateY(0); }
`
const glowAnim = keyframes`
  0%, 100% { box-shadow: 0 0 8px rgba(98,126,234,0.6); }
  50%       { box-shadow: 0 0 24px rgba(98,126,234,0.9); }
`

// ─── css helpers (required for keyframe interpolation in v4+) ──
const pulseAnimation = css`
  animation: ${pulse} 2s infinite;
`
const glowAnimation = css`
  animation: ${glowAnim} 1.5s infinite;
`
const slideInAnimation = css`
  animation: ${slideIn} 0.3s ease;
`

// ─── Styled Components ─────────────────────────────────────────
const Wrap = styled.div`
  padding: 10px;
  height: 100%;
  @media (max-width: 1500px) { padding: 5px; width: 700px; }
`

const Card = styled.div`
  background-color: ${COLORS.cardBg};
  border: 1px solid ${COLORS.border};
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  margin-bottom: 20px;
`

const GradTitle = styled.h2`
  font-size: 1.6rem;
  font-weight: bold;
  margin: 0 0 20px 0;
  background: linear-gradient(135deg, ${COLORS.ethereumPurple}, ${COLORS.blockchainBlue});
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-align: center;
`

const Divider = styled.div`
  width: 100%; height: 1px;
  background: ${COLORS.border};
  margin: 20px 0;
`

// FIX: use css helper instead of raw keyframe interpolation
const StatusDot = styled.div`
  width: 12px; height: 12px;
  border-radius: 50%;
  background: ${p => p.on ? COLORS.cryptoGreen : COLORS.cryptoRed};
  box-shadow: ${p => p.on ? `0 0 10px ${COLORS.cryptoGreen}` : 'none'};
  ${p => p.on && pulseAnimation}
`

const StatusBar = styled.div`
  display: flex; align-items: center; gap: 12px;
  padding: 12px 16px;
  border-radius: 12px;
  border: 1px solid ${p => p.on ? COLORS.cryptoGreen : COLORS.cryptoRed};
  background: ${p => p.on ? 'rgba(16,185,129,0.08)' : 'rgba(239,68,68,0.08)'};
  margin-bottom: 16px;
  font-weight: bold;
  color: ${COLORS.textPrimary};
`

const Btn = styled.button`
  flex: 1; padding: 14px;
  border: none; border-radius: 10px;
  font-weight: bold; font-size: 1rem;
  transition: all 0.2s;
  background: ${p => p.disabled
    ? COLORS.surface
    : p.danger
      ? `linear-gradient(135deg, ${COLORS.cryptoRed}, #DC2626)`
      : `linear-gradient(135deg, ${COLORS.cryptoGreen}, #059669)`};
  color: ${p => p.disabled ? COLORS.textSecondary : 'white'};
  cursor: ${p => p.disabled ? 'not-allowed' : 'pointer'};
  box-shadow: ${p => p.disabled ? 'none'
    : p.danger
      ? `0 4px 16px rgba(239,68,68,0.3)`
      : `0 4px 16px rgba(16,185,129,0.3)`};
`

// FIX: use css helper for conditional glow + slideIn
const SignalBox = styled.div`
  display: flex; flex-direction: column; align-items: center;
  padding: 20px;
  border-radius: 14px;
  border: 2px solid ${p =>
    p.action === 'BUY'  ? COLORS.cryptoGreen :
    p.action === 'SELL' ? COLORS.cryptoRed   :
    COLORS.border};
  background: ${p =>
    p.action === 'BUY'  ? 'rgba(16,185,129,0.08)' :
    p.action === 'SELL' ? 'rgba(239,68,68,0.08)'  :
    COLORS.surface};
  margin-bottom: 16px;
  ${slideInAnimation}
  ${p => p.strong && glowAnimation}
`

const ActionLabel = styled.div`
  font-size: 2.2rem; font-weight: bold;
  color: ${p =>
    p.action === 'BUY'  ? COLORS.cryptoGreen  :
    p.action === 'SELL' ? COLORS.cryptoRed     :
    COLORS.textSecondary};
`

const Spinner = styled.div`
  width: 40px; height: 40px;
  border: 4px solid ${COLORS.surface};
  border-top: 4px solid ${COLORS.ethereumPurple};
  border-radius: 50%;
  animation: ${spin} 1s linear infinite;
  margin: 20px auto;
`

const ErrorBanner = styled.div`
  padding: 12px; border-radius: 10px;
  background: rgba(239,68,68,0.1);
  border: 1px solid ${COLORS.cryptoRed};
  color: ${COLORS.cryptoRed};
  text-align: center; font-size: 0.9rem;
  margin-bottom: 12px;
`

const LiveBadge = styled.span`
  display: inline-block;
  padding: 3px 10px; border-radius: 20px;
  font-size: 0.75rem; font-weight: bold;
  background: rgba(16,185,129,0.15);
  border: 1px solid ${COLORS.cryptoGreen};
  color: ${COLORS.cryptoGreen};
  margin-left: 8px; vertical-align: middle;
`

// ─── Sub-components ─────────────────────────────────────────────
const ProbBar = ({ label, value, color }) => (
  <div style={{ marginBottom: 14 }}>
    <div style={{ display: 'flex', justifyContent: 'space-between',
                  marginBottom: 6, fontSize: '0.9rem' }}>
      <span style={{ fontWeight: 'bold', color: COLORS.textPrimary }}>{label}</span>
      <span style={{ color: COLORS.textSecondary }}>{Number(value || 0).toFixed(1)}%</span>
    </div>
    <div style={{
      width: '100%', height: 22, background: COLORS.surface,
      borderRadius: 11, overflow: 'hidden', border: `1px solid ${COLORS.border}`
    }}>
      <div style={{
        width: `${Math.min(value || 0, 100)}%`, height: '100%',
        background: `linear-gradient(90deg, ${color}, ${color}cc)`,
        transition: 'width 0.5s ease',
        boxShadow: `0 0 10px ${color}55`,
      }} />
    </div>
  </div>
)

const StatCard = ({ label, value, positive }) => (
  <div style={{
    padding: 16, background: COLORS.surface,
    border: `1px solid ${COLORS.border}`,
    borderRadius: 12, textAlign: 'center',
  }}>
    <div style={{ color: COLORS.textSecondary, fontSize: '0.82rem', marginBottom: 8 }}>
      {label}
    </div>
    <div style={{
      fontSize: '1.2rem', fontWeight: 'bold',
      color: positive !== undefined
        ? (positive ? COLORS.cryptoGreen : COLORS.cryptoRed)
        : COLORS.textPrimary,
    }}>
      {value}
    </div>
  </div>
)

// ─── Main Component ─────────────────────────────────────────────
const RightSideSection = () => {
  const socketRef = useRef(null)

  const [botStatus, setBotStatus] = useState({
    running: false, last_action: 'HOLD',
    confidence: 0, current_price: 0,
    virtual_net_worth: 1000, virtual_pnl: 0,
    signal_strength: 'normal',
  })
  const [probabilities, setProbabilities] = useState({
    HOLD: 33.3, BUY: 33.3, SELL: 33.3
  })
  const [statistics, setStatistics] = useState({
    net_profit: 0, win_rate: 0, total_trades: 0,
    winning_trades: 0, losing_trades: 0,
    average_profit: 0, average_loss: 0, profit_factor: 0,
  })
  const [loading,     setLoading]     = useState(true)
  const [error,       setError]       = useState(null)
  const [apiOnline,   setApiOnline]   = useState(false)
  const [wsConnected, setWsConnected] = useState(false)
  const [lastUpdate,  setLastUpdate]  = useState(null)
  const [strongAlert, setStrongAlert] = useState(null)

  // ── WebSocket ────────────────────────────────
  useEffect(() => {
    const socket = io(API_URL, { transports: ['websocket', 'polling'] })
    socketRef.current = socket

    socket.on('connect',    () => { setWsConnected(true);  socket.emit('request_update') })
    socket.on('disconnect', () =>   setWsConnected(false))

    socket.on('bot_status', data => {
      setBotStatus(prev => ({ ...prev, ...data }))
    })

    socket.on('prediction_update', data => {
      if (data.bot_status)               setBotStatus(prev => ({ ...prev, ...data.bot_status }))
      if (data.prediction?.probabilities) setProbabilities(data.prediction.probabilities)
      setLastUpdate(new Date().toLocaleTimeString())
    })

    socket.on('strong_signal', data => {
      setStrongAlert(data)
      setTimeout(() => setStrongAlert(null), 8000)
    })

    return () => socket.disconnect()
  }, [])

  // ── REST polling ─────────────────────────────
  useEffect(() => {
    const init = async () => {
      setLoading(true)
      try {
        const res = await axios.get(`${API_URL}/`, { timeout: 4000 })
        if (res.data.status === 'online') setApiOnline(true)
      } catch {
        setError('Flask API not accessible')
      }
      await Promise.all([fetchStatus(), fetchStats()])
      setLoading(false)
    }
    init()
    const iv = setInterval(() => { fetchStatus(); fetchStats() }, 10000)
    return () => clearInterval(iv)
  }, [])

  const fetchStatus = async () => {
    try {
      const res = await axios.get(`${API_URL}/bot_status`, { timeout: 5000 })
      setBotStatus(prev => ({ ...prev, ...res.data }))
      setApiOnline(true); setError(null)
    } catch {
      setApiOnline(false)
      setError('Flask not accessible')
    }
  }

  const fetchStats = async () => {
    try {
      const res = await axios.get(`${API_URL}/statistics`, { timeout: 5000 })
      if (res.data.success) setStatistics(res.data.statistics)
    } catch {}
  }

  const controlBot = async (action) => {
    try {
      const res = await axios.post(`${API_URL}/bot_control`, { action }, { timeout: 8000 })
      if (res.data.success) {
        setBotStatus(prev => ({ ...prev, ...res.data.status }))
        setError(null)
      }
    } catch {
      setError('Cannot control bot — check Flask API')
    }
  }

  // ── Render ──────────────────────────────────
  if (loading) return (
    <Wrap>
      <Card style={{ textAlign: 'center' }}>
        <Spinner />
        <p style={{ color: COLORS.textSecondary }}>Loading bot data...</p>
      </Card>
    </Wrap>
  )

  const action   = botStatus.last_action || 'HOLD'
  const isStrong = botStatus.signal_strength === 'STRONG'

  return (
    <Wrap>

      {/* Strong Signal Popup */}
      {strongAlert && (
        <div style={{
          position: 'fixed', top: 20, right: 20, zIndex: 9999,
          padding: '16px 24px', borderRadius: 14,
          background: strongAlert.action === 'BUY'
            ? `linear-gradient(135deg, ${COLORS.cryptoGreen}, #059669)`
            : `linear-gradient(135deg, ${COLORS.cryptoRed}, #DC2626)`,
          color: 'white', fontWeight: 'bold', fontSize: '1.1rem',
          boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        }}>
          STRONG {strongAlert.action} @ ${Number(strongAlert.price || 0).toFixed(2)}
          <div style={{ fontSize: '0.85rem', fontWeight: 'normal', marginTop: 4 }}>
            {Number(strongAlert.confidence || 0).toFixed(1)}% confidence
          </div>
        </div>
      )}

      {/* Bot Control Card */}
      <Card>
        <GradTitle>
          Bot Control
          {wsConnected && <LiveBadge>LIVE</LiveBadge>}
        </GradTitle>

        {error      && <ErrorBanner>{error}</ErrorBanner>}
        {!apiOnline && <ErrorBanner>Flask server not accessible</ErrorBanner>}

        <StatusBar on={botStatus.running}>
          <StatusDot on={botStatus.running} />
          {botStatus.running ? 'RUNNING' : 'STOPPED'}
        </StatusBar>

        <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
          <Btn
            disabled={botStatus.running || !apiOnline}
            onClick={() => controlBot('start')}
          >
            {botStatus.running ? 'Running...' : 'Start Bot'}
          </Btn>
          <Btn
            danger
            disabled={!botStatus.running || !apiOnline}
            onClick={() => controlBot('stop')}
          >
            Stop Bot
          </Btn>
        </div>

        {/* Signal Box */}
        <SignalBox action={action} strong={isStrong}>
          <div style={{ color: COLORS.textSecondary, fontSize: '0.85rem', marginBottom: 8 }}>
            AI Signal
            {isStrong && (
              <span style={{
                marginLeft: 8, padding: '2px 10px', borderRadius: 20,
                background: COLORS.goldAccent, color: '#000',
                fontSize: '0.75rem', fontWeight: 'bold',
              }}>
                STRONG
              </span>
            )}
          </div>

          <ActionLabel action={action}>{action}</ActionLabel>

          <div style={{ color: COLORS.textSecondary, marginTop: 8, fontSize: '0.9rem' }}>
            Confidence: {(Number(botStatus.confidence || 0) * 100).toFixed(1)}%
          </div>

          {botStatus.current_price > 0 && (
            <div style={{ color: COLORS.textPrimary, marginTop: 4, fontSize: '0.9rem' }}>
              ETH: ${Number(botStatus.current_price).toFixed(2)}
            </div>
          )}

          {lastUpdate && (
            <div style={{ color: COLORS.textSecondary, marginTop: 4, fontSize: '0.75rem' }}>
              Updated: {lastUpdate}
            </div>
          )}
        </SignalBox>

        {/* Probability Bars */}
        <div style={{ marginTop: 16 }}>
          <ProbBar label="BUY"  value={probabilities.BUY  || 0} color={COLORS.cryptoGreen}    />
          <ProbBar label="HOLD" value={probabilities.HOLD || 0} color={COLORS.ethereumPurple}  />
          <ProbBar label="SELL" value={probabilities.SELL || 0} color={COLORS.cryptoRed}        />
        </div>

        <Divider />

        {/* Virtual P&L */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <StatCard
            label="Virtual Net Worth"
            value={`$${Number(botStatus.virtual_net_worth || 1000).toFixed(2)}`}
          />
          <StatCard
            label="Virtual P&L"
            value={`${(botStatus.virtual_pnl || 0) >= 0 ? '+' : ''}$${Number(botStatus.virtual_pnl || 0).toFixed(2)}`}
            positive={(botStatus.virtual_pnl || 0) >= 0}
          />
        </div>
      </Card>

      {/* Performance Card */}
      <Card>
        <GradTitle>Performance</GradTitle>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <StatCard label="Net Profit"    value={`$${Number(statistics.net_profit || 0).toFixed(2)}`}    positive={statistics.net_profit > 0}   />
          <StatCard label="Win Rate"      value={`${Number(statistics.win_rate || 0).toFixed(1)}%`}      positive={statistics.win_rate > 50}    />
          <StatCard label="Total Trades"  value={statistics.total_trades  || 0} />
          <StatCard label="Winners"       value={statistics.winning_trades || 0} positive />
          <StatCard label="Avg Profit"    value={`$${Number(statistics.average_profit || 0).toFixed(2)}`} positive={statistics.average_profit > 0} />
          <StatCard label="Profit Factor" value={Number(statistics.profit_factor || 0).toFixed(2)}       positive={statistics.profit_factor > 1}  />
        </div>
      </Card>

    </Wrap>
  )
}

export default RightSideSection