'''
Run Real-Time ETH Trading Predictor
=====================================
Entry point pour lancer les prédictions en temps réel.

Usage:
    python run_realtime.py                     # mode continu
    python run_realtime.py --once              # une seule prédiction
    python run_realtime.py --backtest          # test sur données historiques
    python run_realtime.py --list-models       # liste les modèles disponibles
'''

import argparse
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime

from predictor_engine import RealTimePredictor, Config


# ─────────────────────────────────────────────
#  UTILITAIRES
# ─────────────────────────────────────────────

def list_available_models(base_dir: str = "."):
    '''Affiche tous les modèles sauvegardés disponibles.'''
    folders = sorted(glob.glob(os.path.join(base_dir, "*_Crypto_trader")))
    if not folders:
        print("❌ Aucun modèle trouvé dans le répertoire courant.")
        return []

    print("\n" + "═"*70)
    print("  📁 MODÈLES DISPONIBLES")
    print("═"*70)

    model_list = []
    for folder in folders:
        weight_files = glob.glob(os.path.join(folder, "*_Actor.weights.h5"))
        if not weight_files:
            continue

        print(f"\n  📂 {os.path.basename(folder)}")

        # Lire Parameters.txt si disponible
        param_file = os.path.join(folder, "Parameters.txt")
        if os.path.exists(param_file):
            with open(param_file) as f:
                for line in f.readlines()[:4]:
                    print(f"      {line.rstrip()}")

        # Lister les checkpoints avec leurs scores
        scores = []
        for wf in sorted(weight_files):
            fname = os.path.basename(wf)
            score = fname.split("_Crypto_trader_Actor")[0]
            if score:
                scores.append(score)
                print(f"      💾 Score: {score}")
        model_list.append({"folder": folder, "scores": scores})

    print("═"*70 + "\n")
    return model_list


def pick_best_model(base_dir: str = ".") -> tuple:
    '''
    Trouve automatiquement le meilleur modèle (score le plus élevé).
    Returns: (folder_path, score_str)
    '''
    folders = sorted(glob.glob(os.path.join(base_dir, "*_Crypto_trader")))
    best_score  = -float('inf')
    best_folder = None
    best_score_str = None

    for folder in folders:
        weight_files = glob.glob(os.path.join(folder, "*_Actor.weights.h5"))
        for wf in weight_files:
            fname = os.path.basename(wf)
            score_str = fname.split("_Crypto_trader_Actor")[0]
            try:
                score = float(score_str)
                if score > best_score:
                    best_score     = score
                    best_folder    = folder
                    best_score_str = score_str
            except ValueError:
                pass

    return best_folder, best_score_str


def run_backtest_on_csv(
    predictor: RealTimePredictor,
    csv_path:  str,
    n_last:    int = 200
):
    '''
    Teste le modèle sur les N dernières lignes du CSV historique.
    Simule le comportement temps-réel sans appel API.
    '''
    print(f"\n🧪 BACKTEST MODE — {csv_path} (dernières {n_last} lignes)\n")

    df = pd.read_csv(csv_path, index_col=False)
    df = df.rename(columns={'price': 'Close', 'date': 'Date'})

    # Normalisation (même pipeline que training)
    close_col = df['Close'].copy()
    dates_col = df['Date'].copy()
    df_feat   = df.drop(['Close', 'Date'], axis=1, errors='ignore')
    col_max   = df_feat.max().max()
    col_min   = df_feat.min().min()
    df_norm   = (df_feat - col_min) / (col_max - col_min)
    df_norm['Close'] = close_col
    df_norm['Date']  = dates_col

    results = []
    df_slice = df_norm.tail(n_last + Config.LOOKBACK_WINDOW)

    for i in range(Config.LOOKBACK_WINDOW, len(df_slice)):
        window = df_slice.iloc[i - Config.LOOKBACK_WINDOW:i]

        # Build feature matrix (LOOKBACK × 20) from historical rows
        feature_rows = []
        for _, row in window.iterrows():
            fv = []
            for col in Config.FEATURE_COLUMNS:
                fv.append(float(row.get(col, 0.5)))
            feature_rows.append(fv)

        state = np.array(feature_rows, dtype=np.float32)
        state = np.expand_dims(state, axis=0)  # (1, 30, 20)

        # Run model inference
        probs  = predictor.actor.predict(state, verbose=0)[0]
        action = int(np.argmax(probs))

        current_row  = df_slice.iloc[i]
        current_price= float(current_row['Close'])
        current_date = str(current_row.get('Date', i))

        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        results.append({
            "date":       current_date,
            "price":      current_price,
            "action":     action_map[action],
            "prob_buy":   round(float(probs[1]) * 100, 2),
            "prob_hold":  round(float(probs[0]) * 100, 2),
            "prob_sell":  round(float(probs[2]) * 100, 2),
            "confidence": round(float(probs[action]) * 100, 2),
        })

    df_results = pd.DataFrame(results)

    # Stats
    n          = len(df_results)
    buy_count  = (df_results['action'] == 'BUY').sum()
    sell_count = (df_results['action'] == 'SELL').sum()
    hold_count = (df_results['action'] == 'HOLD').sum()
    avg_conf   = df_results['confidence'].mean()

    print("═"*60)
    print("   RÉSULTATS BACKTEST")
    print(f"  Période:       {df_results['date'].iloc[0]} → {df_results['date'].iloc[-1]}")
    print(f"  Steps testés:  {n}")
    print(f"  BUY signals:   {buy_count} ({buy_count/n*100:.1f}%)")
    print(f"  SELL signals:  {sell_count} ({sell_count/n*100:.1f}%)")
    print(f"  HOLD signals:  {hold_count} ({hold_count/n*100:.1f}%)")
    print(f"  Confiance moy: {avg_conf:.1f}%")
    print("═"*60)

    out_file = "backtest_results.csv"
    df_results.to_csv(out_file, index=False)
    print(f"  Résultats sauvegardés: {out_file}\n")
    print(df_results.tail(20).to_string(index=False))


# ─────────────────────────────────────────────
#  CALLBACKS D'ALERTE
# ─────────────────────────────────────────────

def alert_strong_signal(result: dict):
    '''Callback appelé quand un signal fort est détecté.'''
    label = result['action_label']
    price = result['price']
    conf  = result['confidence']
    print(f"\n ALERTE SIGNAL FORT: {label} @ ${price:,.2f} ({conf:.1f}% confiance) 🔥🔥\n")
    # → Vous pouvez ajouter: email, Discord webhook, SMS, etc.


def log_all_predictions(result: dict):
    '''Callback appelé à chaque prédiction.'''
    # Exemple: envoyer à une base de données, webhook, etc.
    pass


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Ethereum Trading Predictor"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Dossier du modèle (ex: '2026_01_22_14_12_Crypto_trader')"
    )
    parser.add_argument(
        "--score",
        type=str,
        default=None,
        help="Score du checkpoint (ex: '105397.90')"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="cryptoanalysis_data.csv",
        help="Chemin vers cryptoanalysis_data.csv"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Intervalle entre prédictions en secondes (default: 60)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Faire une seule prédiction puis quitter"
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Mode backtest sur données CSV historiques"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Lister les modèles disponibles"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Nombre max de prédictions (None = infini)"
    )

    args = parser.parse_args()

    # ── List models ────────────────────────────────────────────────────
    if args.list_models:
        list_available_models()
        return

    # ── Auto-select best model if not specified ────────────────────────
    folder = args.folder
    score  = args.score

    if folder is None or score is None:
        print(" Recherche du meilleur modèle disponible...")
        auto_folder, auto_score = pick_best_model()
        if auto_folder is None:
            print(" Aucun modèle trouvé. Spécifiez --folder et --score.")
            list_available_models()
            return
        folder = auto_folder
        score  = auto_score
        print(f"Modèle sélectionné: {os.path.basename(folder)}")
        print(f"   Score: {score}\n")

    # ── Initialize predictor ───────────────────────────────────────────
    predictor = RealTimePredictor(
        model_folder   = folder,
        model_score    = score,
        historical_csv = args.csv if os.path.exists(args.csv) else None
    )

    # ── Backtest mode ──────────────────────────────────────────────────
    if args.backtest:
        predictor.load_model()
        csv_path = args.csv
        if not os.path.exists(csv_path):
            print(f"❌ CSV non trouvé: {csv_path}")
            return
        run_backtest_on_csv(predictor, csv_path, n_last=200)
        return

    # ── Single prediction ──────────────────────────────────────────────
    if args.once:
        predictor.load_model()
        result = predictor.predict_once()
        if result:
            print(f"\nRésultat: {result['action_label']} @ ${result['price']:,.2f}")
            print(f"   Confiance: {result['confidence']}%")
            print(f"   Probabilités: {result['probabilities']}")
        return

    # ── Continuous loop ────────────────────────────────────────────────
    predictor.run(
        interval_seconds  = args.interval,
        max_iterations    = args.max_iter,
        on_prediction     = log_all_predictions,
        on_strong_signal  = alert_strong_signal
    )


if __name__ == "__main__":
    main()