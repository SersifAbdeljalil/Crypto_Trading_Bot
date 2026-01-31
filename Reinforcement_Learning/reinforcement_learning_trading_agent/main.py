'''Entraînement PPO ULTRA-RAPIDE (30-60 minutes max)
Author: Roberto Lentini (version optimisée vitesse)
'''
import pandas as pd
from env import EthereumEnv, CustomAgent
from models import train_agent, test_agent
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":

    # ========================================
    # ⚡ CONFIGURATION ULTRA-RAPIDE
    # ========================================
    
    LOOKBACK_WINDOW = 30           # ⚡ Réduit de 50 → 30 (moins de données à traiter)
    TEST_WINDOW = 300              # ⚡ Réduit de 500 → 300
    
    TRAIN_EPISODES = 300           # ⚡ DRASTIQUEMENT réduit: 2000 → 300
    BATCH_SIZE = 128               # ⚡ Réduit de 500 → 128 (updates plus fréquents)
    LEARNING_RATE = 0.0001         # ⚡ Augmenté pour apprentissage plus rapide
    EPOCHS = 5                     # ⚡ Réduit de 15 → 5 (suffisant avec bon LR)
    
    # ========================================
    # CHARGEMENT DES DONNÉES
    # ========================================
    
    print("="*60)
    print("⚡ CONFIGURATION ULTRA-RAPIDE (30-60 min)")
    print("="*60)
    print(f"Episodes: {TRAIN_EPISODES} (au lieu de 2000)")
    print(f"Learning rate: {LEARNING_RATE} (apprentissage rapide)")
    print(f"Epochs per update: {EPOCHS} (efficace)")
    print(f"Batch size: {BATCH_SIZE} (équilibré)")
    print(f"Lookback: {LOOKBACK_WINDOW} (optimisé)")
    print("="*60 + "\n")
    
    df = pd.read_csv('cryptoanalysis_data.csv', index_col=False)
    df = df.rename(columns={'price': 'Close', 'date': 'Date'})
    
    # Séparation Close et Date
    Close = list(df['Close'])
    df = df.drop(['Close'], axis=1)
    Date = df['Date']
    df = df.drop(['Date'], axis=1)
    
    # Normalisation globale
    column_maxes = df.max()
    df_max = column_maxes.max()
    column_mins = df.min()
    df_min = column_mins.min()
    
    normalized_df = (df - df_min) / (df_max - df_min)
    normalized_df['Close'] = Close
    normalized_df['Date'] = Date
    df = normalized_df
    
    # Split train/test
    train_df = df[:-TEST_WINDOW - LOOKBACK_WINDOW]
    test_df = df[-TEST_WINDOW - LOOKBACK_WINDOW:]
    
    print(f"📊 Données chargées:")
    print(f"   Total: {len(df)} points")
    print(f"   Train: {len(train_df)} points")
    print(f"   Test: {len(test_df)} points")
    print(f"   Features: {df.shape[1]}\n")
    
    # ========================================
    # CRÉATION DE L'AGENT OPTIMISÉ VITESSE
    # ========================================
    
    print("🤖 Création de l'agent PPO rapide...")
    agent = CustomAgent(
        lookback_window_size=LOOKBACK_WINDOW,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        optimizer=Adam,
        batch_size=64,              # Bon compromis vitesse/stabilité
        model="CNN"                 # CNN = rapide et efficace
    )
    
    print("🏋️ Création de l'environnement...")
    train_env = EthereumEnv(
        train_df, 
        lookback_window_size=LOOKBACK_WINDOW
    )
    
    # ========================================
    # ENTRAÎNEMENT RAPIDE
    # ========================================
    
    print("\n" + "="*60)
    print("⚡ DÉMARRAGE DE L'ENTRAÎNEMENT RAPIDE")
    print("="*60)
    print("✅ Durée estimée: 30-60 minutes")
    print("✅ 300 épisodes au lieu de 2000")
    print("✅ Lookback réduit pour plus de vitesse")
    print("✅ Learning rate augmenté pour convergence rapide")
    print("\n💡 ASTUCE:")
    print("   - Lancez: tensorboard --logdir=logs")
    print("   - Surveillez la courbe de récompense")
    print("   - Arrêtez si récompense positive stable")
    print("="*60 + "\n")
    
    train_agent(
        train_env, 
        agent, 
        visualize=False,              # False = beaucoup plus rapide
        train_episodes=TRAIN_EPISODES,
        training_batch_size=BATCH_SIZE
    )
    
    print("\n" + "="*60)
    print("✅ ENTRAÎNEMENT TERMINÉ EN TEMPS RECORD!")
    print("="*60)
    print("📁 Vérifiez le dossier créé: YYYY_MM_DD_HH_MM_Crypto_trader")
    print("🎯 Cherchez un modèle avec récompense POSITIVE")
    print("="*60 + "\n")
    
    # ========================================
    # TEST RAPIDE (décommenter pour tester)
    # ========================================
    
    # print("🧪 Test rapide du modèle...")
    # test_env = EthereumEnv(test_df, lookback_window_size=LOOKBACK_WINDOW)
    # 
    # test_agent(
    #     test_env, 
    #     agent, 
    #     visualize=True, 
    #     test_episodes=5,  # Réduit pour test rapide
    #     folder="VOTRE_DOSSIER_ICI",
    #     name="VOTRE_MEILLEUR_MODELE",
    #     comment="Test rapide"
    # )