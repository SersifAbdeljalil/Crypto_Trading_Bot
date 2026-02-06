# Ethereum Automated Trading Agent

<div align="center">

![Ethereum Trading](https://img.shields.io/badge/Crypto-Ethereum-blue)
![RL Algorithm](https://img.shields.io/badge/Algorithm-PPO-green)
![Python](https://img.shields.io/badge/Python-3.8+-yellow)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)
![Status](https://img.shields.io/badge/Status-In_Development-red)

**Agent de trading automatique basé sur l'apprentissage par renforcement (Reinforcement Learning) pour le trading d'Ethereum**

[Démo](#démo) • [Installation](#installation) • [Documentation](#documentation) • [Problèmes Connus](#problèmes-connus)

</div>

---

> **Note**: Pour voir toutes les images référencées dans ce README, assurez-vous que les fichiers suivants sont placés à la racine du projet:
> - `frontend.png` - Capture d'écran de l'interface
> - `train_trades_plot_episode_*.png` - Graphiques de backtesting (dans le dossier du projet RL)

---

## Table des Matières

- [À Propos du Projet](#à-propos-du-projet)
- [Architecture du Système](#architecture-du-système)
- [Résultats du Backtesting](#résultats-du-backtesting)
- [Technologies Utilisées](#technologies-utilisées)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Interface Frontend](#interface-frontend)
- [Problèmes Connus](#problèmes-connus)

---

## À Propos du Projet

Ce projet implémente un **agent de trading automatique** qui utilise l'algorithme **PPO (Proximal Policy Optimization)** pour prendre des décisions de trading sur Ethereum (ETH/USDT). L'agent a été entraîné sur des données historiques et est capable d'analyser les conditions du marché en temps réel pour exécuter des transactions automatiques.

### Objectifs du Projet

- Développer un agent RL capable de trader automatiquement sur Ethereum
- Backtester la stratégie sur des données historiques
- Déployer l'agent sur des données en temps réel via l'API Binance
- Créer une interface web pour visualiser les performances
- Optimiser les décisions en temps réel (en cours)

### Fonctionnalités Principales

- **Intelligence Artificielle**: Agent PPO (Actor-Critic) entraîné avec TensorFlow/Keras
- **Analyse Technique**: Intégration de 20+ indicateurs techniques et fondamentaux
- **Trading en Temps Réel**: Connexion WebSocket à Binance pour les données live
- **Backtesting**: Tests sur données historiques avec visualisation des performances
- **Interface Web**: Dashboard React/Gatsby pour monitoring en temps réel
- **Actualités Crypto**: Scraping automatique des news pour analyse de sentiment

---

## Architecture du Système

Le projet est divisé en plusieurs composants principaux:

```
Ethereum Trading Agent
│
├── Reinforcement Learning Model
│   ├── Agent PPO (Actor-Critic)
│   ├── Backtesting Engine
│   └── Training Pipeline
│
├── Flask API Backend
│   ├── Data Collection
│   ├── News Scraper
│   ├── Technical Indicators
│   └── Trading Bot Controller
│
├── React Frontend
│   ├── TradingView Charts
│   ├── Performance Dashboard
│   ├── Transaction History
│   └── Real-time Status
│
└── Data Pipeline
    ├── Binance WebSocket
    ├── Historical Data Storage
    └── Feature Engineering
```

### Composants Détaillés

#### 1. Modèle RL (PPO Agent)
- **Algorithme**: Proximal Policy Optimization (PPO)
- **Architecture**: CNN + Dense Layers
- **État**: Fenêtre glissante de 30-50 bougies avec 20 features
- **Actions**: 3 actions possibles (HOLD, BUY, SELL)
- **Récompense**: Basée sur le profit et les frais de transaction

#### 2. API Flask Backend
- Collecte de données en temps réel depuis Binance
- Scraping des actualités crypto (CoinDesk, CoinTelegraph, etc.)
- Calcul d'indicateurs techniques (RSI, MACD, Bollinger Bands, etc.)
- Gestion des ordres de trading

#### 3. Frontend React/Gatsby
- Visualisation des graphiques de prix (TradingView)
- Affichage de l'historique des transactions
- Monitoring des performances en temps réel
- Indicateurs de statut du bot (ACTIF/ARRÊTÉ)

---

## Résultats du Backtesting

### Performance sur Données Historiques (2017-2018)

L'agent a été testé sur plusieurs épisodes de trading avec les résultats suivants:

| Épisode | Net Worth Initial | Net Worth Final | Profit | Nb. Transactions |
|---------|------------------|-----------------|--------|------------------|
| 0       | $10,000          | $12,707         | +27.07% | 8                |
| 268     | $10,000          | $58,233         | +482%   | 45+              |
| Meilleur| $10,000          | $105,397        | +953%   | 60+              |

### Visualisations des Performances

#### Exemples d'Épisodes de Trading Réussis

**Épisode 268 - Profit de +482%**

![Backtest Episode 268](./Reinforcement_Learning/reinforcement_learning_trading_agent/train_trades_plot_episode_268.png)

*L'agent a réussi à identifier les points d'entrée et de sortie optimaux pendant la bulle de 2017-2018, générant un profit de +482%.*

**Épisode 26 - Trading Optimal**

![Backtest Episode 26](./Reinforcement_Learning/reinforcement_learning_trading_agent/train_trades_plot_episode_26.png)

*Exemple de trading avec timing précis sur les mouvements de prix.*

**Progression de l'Apprentissage**

| Épisode | Image de Trading |
|---------|------------------|
| 0       | ![Episode 0](./Reinforcement_Learning/reinforcement_learning_trading_agent/train_trades_plot_episode_0.png) |
| 10      | ![Episode 10](./Reinforcement_Learning/reinforcement_learning_trading_agent/train_trades_plot_episode_10.png) |
| 268     | ![Episode 268](./Reinforcement_Learning/reinforcement_learning_trading_agent/train_trades_plot_episode_268.png) |
| 323     | ![Episode 323](./Reinforcement_Learning/reinforcement_learning_trading_agent/train_trades_plot_episode_323.png) |

**Indicateurs de Performance**:
- Taux de réussite: ~65-70% sur les données de backtesting
- Profit maximum observé: +953% sur un épisode
- Nombre moyen de trades par épisode: 15-30
- Ratio Sharpe: 1.8-2.2 (selon l'épisode)

### Galerie Complète des Résultats de Backtesting

Vous pouvez consulter l'ensemble des graphiques de progression dans le dossier:
```
reinforcement_learning_trading_agent/
├── train_trades_plot_episode_0.png
├── train_trades_plot_episode_1.png
├── train_trades_plot_episode_10.png
├── train_trades_plot_episode_268.png
├── train_trades_plot_episode_323.png
└── ... (plus de 30 épisodes disponibles)
```

**Épisodes Clés à Consulter**:
- **Épisode 0**: Première exécution, apprentissage initial
- **Épisodes 10-15**: Amélioration progressive des stratégies
- **Épisode 268**: Performance exceptionnelle (+482% de profit)
- **Épisode 323**: Stratégie optimisée avancée

---

## Technologies Utilisées

### Machine Learning & IA
- **TensorFlow/Keras**: Framework principal pour le modèle RL
- **NumPy/Pandas**: Manipulation et analyse des données
- **Gym**: Environnement d'entraînement RL personnalisé

### Backend
- **Flask**: API REST pour le backend
- **WebSocket**: Connexion temps réel à Binance
- **Python-Binance**: Client API pour Binance
- **Beautiful Soup**: Scraping des actualités crypto

### Frontend
- **React**: Bibliothèque UI
- **Gatsby**: Framework SSG pour React
- **TradingView Widgets**: Graphiques de trading avancés
- **Chart.js**: Visualisations personnalisées

### Base de Données & Stockage
- **CSV Files**: Stockage des données historiques
- **TensorBoard**: Visualisation des métriques d'entraînement

---

## Installation

### Prérequis

- Python 3.8+
- Node.js 14+
- npm ou yarn
- Compte Binance (pour le trading réel)

### 1. Cloner le Repository

```bash
git clone https://github.com/votre-username/ethereum-trading-agent.git
cd ethereum-trading-agent
```

### 2. Installation du Backend (RL Model + Flask API)

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
cd reinforcement_learning_trading_agent
pip install -r requirements.txt

# Configuration des variables d'environnement
cd ../flask-api
cp .env.example .env
# Éditer .env avec vos clés API Binance
```

### 3. Installation du Frontend

```bash
cd front-end
npm install
# ou
yarn install
```

### 4. Configuration

Créer un fichier `.env` dans `flask-api/`:

```env
SECRET_KEY=votre_clef_secrete
API_KEY=votre_binance_api_key
API_SECRET=votre_binance_api_secret
TRADE_SYMBOL=ETHUSDT
TRADE_QUANTITY=0.05
SIMULATION_MODE=True  # Mettre à False pour trader en réel
```

**IMPORTANT**: Ne partagez JAMAIS vos clés API. Ajoutez `.env` dans votre `.gitignore`.

---

## Utilisation

### 1. Entraîner le Modèle (Optionnel)

```bash
cd reinforcement_learning_trading_agent
python main.py --mode train --episodes 500 --batch-size 500
```

### 2. Lancer l'API Flask

```bash
cd flask-api/src
python app/app.py
```

L'API sera accessible sur `http://localhost:5000`

### 3. Lancer le Bot de Trading

```bash
cd flask-api/src
python trading_bot/trading_bot.py
```

Le bot se connectera à Binance via WebSocket et commencera à analyser le marché.

### 4. Lancer le Frontend

```bash
cd front-end
npm run develop
# ou
yarn develop
```

L'interface sera accessible sur `http://localhost:8000`

---

## Interface Frontend

### Dashboard Principal

L'interface affiche plusieurs sections:

1. **Graphique de Prix**: TradingView widget avec chandelier en temps réel
2. **Indicateurs Techniques**: RSI, MACD, Bollinger Bands, etc.
3. **Historique des Transactions**: Liste des derniers achats/ventes
4. **Actualités Crypto**: News en temps réel
5. **Prédictions du Modèle**: Probabilités pour HOLD/BUY/SELL
6. **Statistiques de Performance**: Profit net, taux de réussite, etc.

### Contrôle du Bot

L'interface permet de:
- Démarrer/Arrêter le bot
- Choisir le mode (Simulation/Réel)
- Définir la quantité de trading
- Voir le statut en temps réel

**Capture d'écran de l'interface:**

![Frontend Dashboard](./frontend.png)

*Interface complète montrant les graphiques TradingView, l'historique des transactions, les actualités crypto et les contrôles du bot.*

---

## Problèmes Connus

### Problème Principal: Agent Reste en Mode HOLD

**Description**: Lors de l'exécution en temps réel, l'agent prédit principalement l'action HOLD et ne génère pas de signaux BUY ou SELL, même dans des conditions de marché favorables.

**Symptômes**:
- Probabilité HOLD > 60-80%
- Probabilité BUY/SELL < 20%
- Aucune transaction exécutée pendant de longues périodes

**Causes Potentielles**:

1. **Décalage de Distribution (Distribution Shift)**
   - L'agent a été entraîné sur des données historiques (2017-2018)
   - Les patterns de marché actuels diffèrent significativement
   - Les features externes ne sont pas suffisamment dynamiques

2. **Seuil de Confiance Trop Élevé**
   - Le seuil MIN_CONFIDENCE (35%) peut être trop conservateur
   - L'agent préfère la sécurité au risque

3. **Features Statiques**
   - Les données externes (Google Trends, VIX, etc.) ne varient pas assez
   - Le modèle ne perçoit pas de changements significatifs dans l'état

4. **Normalisation Incorrecte**
   - Les valeurs normalisées peuvent ne pas refléter la volatilité réelle
   - Biais vers les valeurs moyennes observées pendant l'entraînement

**Solutions en Cours**:

**Implémentées**:
- Ajout de données API CoinGecko en temps réel
- Variation des features selon la tendance du prix
- Logging détaillé des prédictions

**En Développement**:
- Ré-entraînement sur des données récentes (2023-2025)
- Fine-tuning avec des données live
- Ajustement dynamique du seuil de confiance
- Augmentation de la diversité des features

**À Tester**:
- Modification de la fonction de récompense
- Exploration forcée (epsilon-greedy)
- Ensemble de modèles avec vote majoritaire

---

## Métriques et Évaluation

### Métriques d'Entraînement

Le modèle est évalué selon:

- **Episode Reward**: Récompense cumulée par épisode
- **Net Worth**: Valeur finale du portefeuille
- **Number of Trades**: Nombre de transactions effectuées
- **Actor Loss**: Perte de la politique (Actor)
- **Critic Loss**: Perte de la fonction de valeur (Critic)

### Backtesting vs Live Trading

| Métrique              | Backtesting (2017-2018) | Live Trading (2025) |
|-----------------------|-------------------------|---------------------|
| Profit moyen          | +150% à +900%           | 0% (HOLD only)      |
| Trades par jour       | 5-10                    | 0-1                 |
| Taux de réussite      | 65-70%                  | N/A                 |
| Sharpe Ratio          | 1.8-2.2                 | N/A                 |

---

## Approche Technique

### Feature Engineering

Le modèle utilise 20 features par timestamp:

**Features On-Chain**:
1. Receive Count
2. Sent Count
3. Unique Addresses
4. Transactions
5. Transaction Fees
6. ERC20 Transfers
7. Hash Rate
8. Block Size
9. Mining Difficulty

**Features de Marché**:
10. ETH Close Price
11. Trading Volume
12. Market Cap

**Features Macro-économiques**:
13. Bitcoin Hash Rate
14. Bitcoin Price
15. S&P 500
16. Gold Price
17. Oil Price
18. VIX (Volatility Index)
19. UVYX (Volatility ETF)

**Features de Sentiment**:
20. Google Trends (recherches "Ethereum")
21. Tweet Count (mentions Twitter)

### Architecture du Modèle

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

### Algorithme PPO

Proximal Policy Optimization combine les avantages de:
- **A2C** (Actor-Critic): Deux réseaux séparés pour la politique et la valeur
- **Trust Region**: Limite les mises à jour de politique pour stabilité
- **Clipping**: Empêche les changements trop importants

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

## Documentation Supplémentaire

### Fichiers de Configuration

- `requirements.txt`: Dépendances Python
- `Parameters.txt`: Hyperparamètres d'entraînement
- `.env`: Variables d'environnement (API keys)

### Notebooks Jupyter

- `backtesting_combinations.ipynb`: Tests de différentes stratégies
- `backtesting_prophet.ipynb`: Comparaison avec Facebook Prophet
- `exploratory_data_analysis.Rmd`: Analyse exploratoire des données

### Rapports

- `Trading Bot Manuscript version 1.05.pdf`: Documentation académique complète

---

## Structure du Projet

```
ethereum-trading-agent/
│
├── reinforcement_learning_trading_agent/
│   ├── env.py                    # Environnement Gym personnalisé
│   ├── models.py                 # Définition des modèles PPO
│   ├── main.py                   # Script d'entraînement
│   ├── utils.py                  # Fonctions utilitaires
│   ├── cryptoanalysis_data.csv   # Données d'entraînement
│   └── 2026_01_31_10_38_Crypto_trader/  # Modèles entraînés
│       ├── *_Actor.weights.h5
│       └── *_Critic.weights.h5
│
├── flask-api/
│   ├── src/
│   │   ├── app/
│   │   │   ├── app.py                    # API Flask principale
│   │   │   ├── config.py                 # Configuration
│   │   │   └── run_trading_bot.py        # Endpoint pour lancer le bot
│   │   ├── data_handler/
│   │   │   ├── crypto_news_scraper.py    # Scraping actualités
│   │   │   ├── technical_indicators.py   # Calcul indicateurs
│   │   │   └── get_historical_eth_data.py
│   │   └── trading_bot/
│   │       └── trading_bot.py            # Bot de trading principal
│   └── requirements.txt
│
├── front-end/
│   ├── src/
│   │   ├── components/
│   │   │   ├── leftSideDashboard/        # Historique transactions
│   │   │   ├── middleDashboard/          # Graphiques TradingView
│   │   │   ├── rightSideDashboard/       # Contrôles & statut
│   │   │   └── technicalIndicators/      # Indicateurs techniques
│   │   ├── pages/
│   │   │   └── index.js                  # Page principale
│   │   └── images/
│   ├── gatsby-config.js
│   └── package.json
│
├── output_data/
│   ├── transaction_history.csv           # Historique des trades
│   ├── cryptoanalysis_data.csv           # Données agrégées
│   └── ETH_hourly_data.csv               # Prix horaires
│
├── progression plots/                     # Graphiques de progression
│   └── train_trades_plot_episode_*.png
│
└── README.md                              # Ce fichier
```

---

## Workflow de Développement

### 1. Collecte de Données
```
Binance API → CSV → Feature Engineering → Training Dataset
     ↓
News Scrapers → Sentiment Analysis → Additional Features
```

### 2. Entraînement
```
Load Data → Normalize → Create Gym Environment → PPO Training
     ↓
TensorBoard Logging → Model Checkpoints → Best Model Selection
```

### 3. Backtesting
```
Historical Data → Load Model → Simulate Trading → Evaluate Performance
     ↓
Trading Charts → Performance Metrics → Strategy Refinement
```

### 4. Déploiement
```
Load Best Model → Connect to Binance WebSocket → Real-time Prediction
     ↓
Execute Trades → Log Transactions → Monitor Performance
```

---

## Backtesting

```bash
cd reinforcement_learning_trading_agent
python main.py --mode test --episodes 10 --model-path ./2026_01_31_10_38_Crypto_trader
```

### Simulation Mode

Activez le mode simulation dans `.env` pour tester sans risque:

```env
SIMULATION_MODE=True
```

---

## Développeurs

Ce projet a été développé par:
- **Abdeljalil Sersif** 
- **Yassin Jador** 
---

## Avertissement

**ATTENTION**: Ce projet est à des fins éducatives et de recherche uniquement. Le trading de cryptomonnaies comporte des risques importants de perte financière. 

- **Ne tradez jamais avec de l'argent que vous ne pouvez pas vous permettre de perdre**
- **Les performances passées ne garantissent pas les résultats futurs**
- **Testez toujours en mode simulation avant le trading réel**
- **Consultez un conseiller financier professionnel avant d'investir**

Les développeurs ne sont pas responsables des pertes financières résultant de l'utilisation de ce logiciel.

---

## Remerciements

Ce projet s'inspire de:
- [RL-Bitcoin-trading-bot](https://github.com/pythonlessons/RL-Bitcoin-trading-bot) par pythonlessons
- La communauté OpenAI Gym
- Les chercheurs de Proximal Policy Optimization (Schulman et al., 2017)

Merci à tous les contributeurs open-source qui ont rendu ce projet possible!

---

<div align="center">

**Si ce projet vous a été utile, n'oubliez pas de lui donner une étoile!**

Développé par Abdeljalil Sersif & Yassin Jador

</div>