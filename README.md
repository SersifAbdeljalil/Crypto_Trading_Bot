# Ethereum Automated Trading Agent

<div align="center">

![Ethereum Trading](https://img.shields.io/badge/Crypto-Ethereum-blue)
![RL Algorithm](https://img.shields.io/badge/Algorithm-PPO-green)
![Python](https://img.shields.io/badge/Python-3.8+-yellow)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)
![Status](https://img.shields.io/badge/Status-Backtesting_Complete-brightgreen)

**Reinforcement Learning agent trained on historical Ethereum price data with backtesting analysis**

[About](#about-the-project) • [Backtesting Results](#backtesting-results) • [Installation](#installation) • [Extensions](#team-extensions)

</div>

---

> **Note**: To see all images referenced in this README, ensure the following files are placed at the project root:
> - `frontend.png` - Interface screenshot
> - `train_trades_plot_episode_*.png` - Backtesting charts (in the RL project folder)

---

## Table of Contents

- [About the Project](#about-the-project)
- [Core Project Scope](#core-project-scope)
- [Backtesting Results](#backtesting-results)
- [Team Extensions](#team-extensions)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)

---

## About the Project

This project implements a **PPO (Proximal Policy Optimization) agent** trained exclusively on **historical Ethereum price data (2017-2018)**. The agent was developed for **backtesting purposes only**, without deployment in real-time trading conditions.

### Project Objectives (Core Requirement)

✅ **Completed:**
- Develop an RL agent capable of learning Ethereum trading strategies
- Train the agent on historical data (2017-2018)
- Backtest the strategy and evaluate performance
- Visualize trading decisions and profitability

### What This Project IS

- ✅ A **research and educational project** on reinforcement learning for algorithmic trading
- ✅ A **backtesting framework** to evaluate PPO agent performance on historical data
- ✅ A **proof-of-concept** demonstrating RL applications in cryptocurrency trading

### What This Project IS NOT

- ❌ **NOT** a real-time trading system deployed in production
- ❌ **NOT** tested or validated on live market data
- ❌ **NOT** financial advice or a guaranteed profitable strategy

---

## Core Project Scope

### Academic Requirements Met

The core project fulfills the following academic objectives:

1. **Reinforcement Learning Implementation**
   - PPO (Proximal Policy Optimization) algorithm
   - Actor-Critic architecture with CNN layers
   - Custom Gym environment for trading simulation

2. **Historical Data Training**
   - Training dataset: Ethereum price data (2017-2018)
   - 20+ features including on-chain metrics, market data, and sentiment indicators
   - Sliding window approach (30-50 candles)

3. **Backtesting & Evaluation**
   - Comprehensive backtesting on historical episodes
   - Performance metrics: profit %, Sharpe ratio, win rate
   - Visualization of trading decisions and portfolio evolution

### PPO Agent Architecture

```
Ethereum Trading Agent (Core)
│
├── Reinforcement Learning Model
│   ├── PPO Agent (Actor-Critic)
│   ├── Custom Gym Environment
│   ├── Training Pipeline
│   └── Backtesting Engine
│
└── Data Pipeline
    ├── Historical Data (CSV)
    ├── Feature Engineering
    └── Normalization & Preprocessing
```

**Model Components:**
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Architecture**: CNN + Dense Layers
- **State**: Sliding window of 30-50 candles with 20 features
- **Actions**: 3 possible actions (HOLD, BUY, SELL)
- **Reward**: Based on profit and transaction fees

---

## Backtesting Results

### Performance on Historical Data (2017-2018)

The agent was trained and tested exclusively on historical data with the following results:

| Episode | Initial Net Worth | Final Net Worth | Profit | No. Transactions |
|---------|------------------|-----------------|--------|------------------|
| 0       | $10,000          | $12,707         | +27.07% | 8                |
| 268     | $10,000          | $58,233         | +482%   | 45+              |
| Best    | $10,000          | $105,397        | +953%   | 60+              |

### Performance Visualizations

#### Examples of Successful Trading Episodes

**Episode 268 - Profit of +482%**

![Backtest Episode 268](./Reinforcement_Learning/reinforcement_learning_trading_agent/train_trades_plot_episode_268.png)

*The agent successfully identified optimal entry and exit points during the 2017-2018 bubble, generating a profit of +482%.*

**Episode 26 - Optimal Trading**

![Backtest Episode 26](./Reinforcement_Learning/reinforcement_learning_trading_agent/train_trades_plot_episode_26.png)

*Example of trading with precise timing on price movements.*

**Learning Progression**

| Episode | Trading Image |
|---------|---------------|
| 0       | ![Episode 0](./Reinforcement_Learning/reinforcement_learning_trading_agent/train_trades_plot_episode_0.png) |
| 10      | ![Episode 10](./Reinforcement_Learning/reinforcement_learning_trading_agent/train_trades_plot_episode_10.png) |
| 268     | ![Episode 268](./Reinforcement_Learning/reinforcement_learning_trading_agent/train_trades_plot_episode_268.png) |
| 323     | ![Episode 323](./Reinforcement_Learning/reinforcement_learning_trading_agent/train_trades_plot_episode_323.png) |

**Performance Indicators** (Backtesting Only):
- Success rate: ~65-70% on backtesting data
- Maximum profit observed: +953% on one episode
- Average number of trades per episode: 15-30
- Sharpe Ratio: 1.8-2.2 (depending on episode)

### Complete Gallery of Backtesting Results

You can view all progression charts in the folder:
```
reinforcement_learning_trading_agent/
├── train_trades_plot_episode_0.png
├── train_trades_plot_episode_1.png
├── train_trades_plot_episode_10.png
├── train_trades_plot_episode_268.png
├── train_trades_plot_episode_323.png
└── ... (30+ episodes available)
```

**Key Episodes to Review**:
- **Episode 0**: First execution, initial learning
- **Episodes 10-15**: Progressive improvement of strategies
- **Episode 268**: Exceptional performance (+482% profit)
- **Episode 323**: Advanced optimized strategy

---

## Team Extensions

> **Note**: The following components were developed by our team as **extensions beyond the core academic requirements**. These features explore real-time integration possibilities but are **NOT part of the validated academic deliverable**.

### Additional Architecture (Team Contribution)

Our team extended the base project with the following components for exploration purposes:

```
Extended System Architecture (Team Addition)
│
├── Flask API Backend (Team Extension)
│   ├── Real-time Data Collection from Binance
│   ├── Crypto News Scraper (CoinDesk, CoinTelegraph)
│   ├── Technical Indicators Calculator (RSI, MACD, Bollinger Bands)
│   └── Trading Bot Controller (Experimental)
│
└── React/Gatsby Frontend (Team Extension)
    ├── TradingView Charts Integration
    ├── Performance Dashboard
    ├── Transaction History Display
    └── Real-time Bot Status Indicators
```

### Extended Features (Experimental)

#### 1. Flask API Backend
- **Real-time data collection** from Binance via WebSocket
- **Crypto news scraping** from multiple sources for sentiment analysis
- **Technical indicator calculation** (RSI, MACD, Bollinger Bands, etc.)
- **Trading order management** interface (simulation mode)

#### 2. React/Gatsby Frontend
- **Price chart visualization** using TradingView widgets
- **Transaction history** display and tracking
- **Real-time performance monitoring** dashboard
- **Bot control interface** (Start/Stop, Mode Selection)

**Interface Screenshot:**

![Frontend Dashboard](./frontend.png)

*Experimental interface showing TradingView charts, transaction history, crypto news, and bot controls.*

### ⚠️ Important Disclaimer for Extensions

**The Flask API and Frontend components are:**
- ✅ Functional for demonstration and learning purposes
- ✅ Useful for understanding real-time trading system architecture
- ❌ **NOT validated for real-time trading** performance
- ❌ **NOT tested with live market data** beyond basic connectivity
- ❌ **NOT recommended for actual trading** without extensive additional validation

**Known Limitations:**
- Distribution shift between training data (2017-2018) and current market conditions
- Agent tends to predict HOLD in real-time scenarios (see Known Issues section)
- No live trading validation or performance metrics

---

## Technologies Used

### Core Project (Backtesting)
- **TensorFlow/Keras**: RL model framework
- **NumPy/Pandas**: Data manipulation and analysis
- **Gym**: Custom RL training environment
- **Matplotlib**: Backtesting visualization

### Team Extensions
- **Flask**: REST API backend
- **WebSocket**: Real-time Binance connection
- **Python-Binance**: API client for Binance
- **Beautiful Soup**: News scraping
- **React**: Frontend UI library
- **Gatsby**: Static site generator
- **TradingView Widgets**: Chart integration
- **Chart.js**: Custom visualizations

---

## Installation

### Prerequisites

- Python 3.8+
- Node.js 14+ (for frontend extensions only)
- npm or yarn (for frontend extensions only)

### Core Project Setup (Backtesting Only)

```bash
# Clone the repository
git clone https://github.com/your-username/ethereum-trading-agent.git
cd ethereum-trading-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install core dependencies
cd reinforcement_learning_trading_agent
pip install -r requirements.txt
```

### Extended Features Setup (Optional)

#### Flask API Backend

```bash
cd flask-api
pip install -r requirements.txt

# Create environment configuration
cp .env.example .env
# Edit .env with your settings (optional for backtesting)
```

#### React Frontend

```bash
cd front-end
npm install
# or
yarn install
```

---

## Usage

### Core Project: Backtesting

#### 1. Train the PPO Agent

```bash
cd reinforcement_learning_trading_agent
python main.py --mode train --episodes 500 --batch-size 500
```

#### 2. Run Backtesting

```bash
python main.py --mode test --episodes 10 --model-path ./2026_01_31_10_38_Crypto_trader
```

This will generate:
- Trading performance charts
- Episode statistics
- Profit/loss analysis

### Extended Features (Optional)

#### 1. Launch Flask API (Experimental)

```bash
cd flask-api/src
python app/app.py
```

API accessible at `http://localhost:5000`

#### 2. Launch Trading Bot (Simulation Mode)

```bash
cd flask-api/src
python trading_bot/trading_bot.py
```

⚠️ **Always use SIMULATION_MODE=True in .env**

#### 3. Launch Frontend Dashboard

```bash
cd front-end
npm run develop
# or
yarn develop
```

Interface accessible at `http://localhost:8000`

---

## Known Issues

### Main Issue: Real-Time Performance Gap

**Description**: While the agent performs well on backtesting (historical data), it does not perform effectively when connected to real-time data streams.

**Symptoms in Real-Time Mode**:
- HOLD probability > 60-80%
- BUY/SELL probability < 20%
- No transactions executed for long periods

**Root Causes**:

1. **Distribution Shift**
   - Agent trained on 2017-2018 data
   - Current market patterns differ significantly
   - Model hasn't been retrained on recent data

2. **Static Features**
   - External features (Google Trends, VIX) don't vary enough in real-time
   - Model doesn't perceive significant state changes

3. **Normalization Issues**
   - Normalized values may not reflect current market volatility
   - Bias towards historical mean values

**This is why the project remains in backtesting-only scope.**

---

## Technical Approach

### Feature Engineering

The model uses 20 features per timestamp:

**On-Chain Features** (9):
1. Receive Count
2. Sent Count
3. Unique Addresses
4. Transactions
5. Transaction Fees
6. ERC20 Transfers
7. Hash Rate
8. Block Size
9. Mining Difficulty

**Market Features** (3):
10. ETH Close Price
11. Trading Volume
12. Market Cap

**Macroeconomic Features** (7):
13. Bitcoin Hash Rate
14. Bitcoin Price
15. S&P 500
16. Gold Price
17. Oil Price
18. VIX (Volatility Index)
19. UVYX (Volatility ETF)

**Sentiment Features** (2):
20. Google Trends ("Ethereum" searches)
21. Tweet Count (Twitter mentions)

### Model Architecture

**Shared Model (Actor-Critic)**:

```python
Input: (lookback_window, 20 features) → Shape: (30-50, 20)
    ↓
Conv1D(64 filters, kernel=6) + MaxPooling
    ↓
Conv1D(32 filters, kernel=3) + MaxPooling
    ↓
Flatten
    ↓
    ├── Actor Branch                 ├── Critic Branch
    │   Dense(512) → ReLU           │   Dense(512) → ReLU
    │   Dense(256) → ReLU           │   Dense(256) → ReLU
    │   Dense(64) → ReLU            │   Dense(64) → ReLU
    │   Dense(3) → Softmax          │   Dense(1) → Value
    │   [HOLD, BUY, SELL]           │
```

### PPO Algorithm

Proximal Policy Optimization combines:
- **A2C** (Actor-Critic): Two separate networks for policy and value
- **Trust Region**: Limits policy updates for stability
- **Clipping**: Prevents overly large changes

**Loss Functions**:

```python
# Actor Loss
L_CLIP = min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)

# Critic Loss
L_VALUE = MSE(V_predicted - V_target)

# Entropy Bonus (exploration)
L_ENTROPY = -β * Σ(π * log(π))

# Total Loss
L_TOTAL = L_CLIP - L_VALUE + L_ENTROPY
```

---

## Project Structure

```
ethereum-trading-agent/
│
├── reinforcement_learning_trading_agent/    # CORE PROJECT
│   ├── env.py                    # Custom Gym environment
│   ├── models.py                 # PPO model definitions
│   ├── main.py                   # Training & backtesting script
│   ├── utils.py                  # Utility functions
│   ├── cryptoanalysis_data.csv   # Training data
│   └── 2026_01_31_10_38_Crypto_trader/  # Trained models
│       ├── *_Actor.weights.h5
│       └── *_Critic.weights.h5
│
├── flask-api/                               # TEAM EXTENSION
│   ├── src/
│   │   ├── app/
│   │   │   ├── app.py                    # Main Flask API
│   │   │   ├── config.py                 # Configuration
│   │   │   └── run_trading_bot.py        # Bot launcher
│   │   ├── data_handler/
│   │   │   ├── crypto_news_scraper.py    # News scraping
│   │   │   ├── technical_indicators.py   # Indicators
│   │   │   └── get_historical_eth_data.py
│   │   └── trading_bot/
│   │       └── trading_bot.py            # Trading bot (experimental)
│   └── requirements.txt
│
├── front-end/                               # TEAM EXTENSION
│   ├── src/
│   │   ├── components/
│   │   │   ├── leftSideDashboard/        # Transaction history
│   │   │   ├── middleDashboard/          # TradingView charts
│   │   │   ├── rightSideDashboard/       # Controls & status
│   │   │   └── technicalIndicators/      # Technical indicators
│   │   ├── pages/
│   │   │   └── index.js                  # Main page
│   │   └── images/
│   ├── gatsby-config.js
│   └── package.json
│
├── output_data/
│   ├── transaction_history.csv           # Trade history
│   ├── cryptoanalysis_data.csv           # Aggregated data
│   └── ETH_hourly_data.csv               # Hourly prices
│
├── progression plots/                     # Backtesting results
│   └── train_trades_plot_episode_*.png
│
└── README.md                              # This file
```

---

## Evaluation Metrics

### Backtesting Metrics (Core Project)

The model is evaluated on historical data using:

- **Episode Reward**: Cumulative reward per episode
- **Net Worth**: Final portfolio value vs initial
- **Number of Trades**: Transaction frequency
- **Profit %**: Return on investment
- **Sharpe Ratio**: Risk-adjusted returns
- **Win Rate**: Percentage of profitable trades

### Results Summary

| Metric                | Best Performance     | Average Performance |
|-----------------------|----------------------|---------------------|
| Profit (single episode)| +953%               | +150% to +400%      |
| Sharpe Ratio          | 2.2                  | 1.8-2.0             |
| Win Rate              | 70%                  | 65-70%              |
| Trades per Episode    | 60+                  | 15-30               |

**Important**: These metrics are from backtesting only. Real-time performance is not validated.

---

## Additional Documentation

### Configuration Files

- `requirements.txt`: Python dependencies
- `Parameters.txt`: Training hyperparameters
- `.env`: Environment variables (for extensions)

### Jupyter Notebooks

- `backtesting_combinations.ipynb`: Strategy comparisons
- `backtesting_prophet.ipynb`: Comparison with Facebook Prophet
- `exploratory_data_analysis.Rmd`: Data analysis

### Reports

- `Trading Bot Manuscript version 1.05.pdf`: Complete academic documentation

---

## Development Team

This project was developed by:
- **Abdeljalil Sersif** - RL Model Development & Backtesting
- **Yassin Jador** - Flask API & Frontend Integration

---

## Disclaimer

### ⚠️ CRITICAL WARNINGS

**This project is STRICTLY for educational and research purposes.**

1. **Backtesting Only**: The core validated project is limited to backtesting on historical data (2017-2018).

2. **No Real-Time Validation**: The extended features (Flask API, Frontend) are experimental and NOT validated for live trading.

3. **Financial Risk**: Cryptocurrency trading involves significant risk of financial loss.
   - **Never trade with money you cannot afford to lose**
   - **Past performance does not guarantee future results**
   - **The agent has NOT been tested in real market conditions**

4. **Not Financial Advice**: This project does not constitute financial advice. Consult a licensed financial advisor before making investment decisions.

5. **No Warranty**: The developers provide this software "as is" without warranties of any kind and are not responsible for any financial losses.

### Recommended Use

✅ **Appropriate Uses:**
- Learning about reinforcement learning algorithms
- Understanding PPO implementation
- Studying backtesting methodologies
- Academic research on algorithmic trading

❌ **Inappropriate Uses:**
- Real money trading without extensive additional validation
- Production deployment without proper risk management
- Assuming backtesting results will transfer to live markets

---

## Acknowledgments

This project was inspired by:
- [PPO agent trained on Ethereum price data; backtesting only.]([text](https://github.com/roblen001/reinforcement_learning_trading_agent)) by pythonlessons
- The OpenAI Gym community
- Proximal Policy Optimization researchers (Schulman et al., 2017)

Special thanks to all open-source contributors who made this project possible!

---

## Academic Context

**Project Type**: Reinforcement Learning Research Project  
**Scope**: PPO agent trained and evaluated on historical Ethereum price data  
**Validation**: Backtesting only, no real-time deployment  
**Extensions**: Flask API and React frontend added by team for exploration

---

<div align="center">

**If this project was useful for your learning, don't forget to give it a star!**

Developed by Abdeljalil Sersif & Yassin Jador

</div>