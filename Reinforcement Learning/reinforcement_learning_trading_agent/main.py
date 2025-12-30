'''File which is run to see the agent in action.
    Modified from: https://github.com/pythonlessons/RL-Bitcoin-trading-bot
    Author: Roberto Lentini
    Email: roberto.lentini@mail.utoronto.ca
'''
import pandas as pd
from pandas.core.frame import DataFrame
from env import EthereumEnv, CustomAgent
from models import Random_games, train_agent, test_agent
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn import preprocessing

if __name__ == "__main__":
    # Read the dataset
    df = pd.read_csv('cryptoanalysis_data.csv', index_col=False)
    
    # Rename columns for consistency
    df = df.rename(columns={'price': 'Close'})
    df = df.rename(columns={'date': 'Date'})
    
    # Extract the 'Close' column and remove it from the dataframe
    Close = list(df['Close'])
    df = df.drop(['Close'], axis=1)
    
    # Extract the 'Date' column and remove it from the dataframe
    Date = df['Date']
    df = df.drop(['Date'], axis=1)
    
    # Normalize the data
    column_maxes = df.max()
    df_max = column_maxes.max()
    column_mins = df.min()
    df_min = column_mins.min()
    normalized_df = (df - df_min) / (df_max - df_min)
    normalized_df['Close'] = Close
    normalized_df['Date'] = Date
    df = normalized_df
    
    # Define the lookback window size and test window size
    lookback_window_size = 50
    test_window = 500
    
    # Split the dataframe into training and testing sets
    train_df = df[:-test_window - lookback_window_size]
    test_df = df[-test_window - lookback_window_size:]
    
    print(f"📊 Dataset loaded successfully!")
    print(f"   Total data points: {len(df)}")
    print(f"   Training data: {len(train_df)} points")
    print(f"   Testing data: {len(test_df)} points")
    print(f"   Features: {df.shape[1]}")
    
    # =============================================================================
    # SECTION 1: ENTRAÎNEMENT (DÉCOMMENTEZ CETTE SECTION)
    # =============================================================================
    
    print("\n🤖 Creating agent...")
    agent = CustomAgent(
        lookback_window_size=lookback_window_size,
        lr=0.00001,
        epochs=10,
        optimizer=Adam,
        batch_size=64,
        model="CNN"
    )
    
    print("🏋️ Creating training environment...")
    train_env = EthereumEnv(
        train_df, 
        lookback_window_size=lookback_window_size
    )
    
    print("\n🚀 Starting training...")
    print("   This may take several hours depending on train_episodes")
    print("   Open TensorBoard at http://localhost:6006 to monitor progress")
    print("   Press CTRL+C to stop training\n")
    
    # OPTION A: Test rapide (50 épisodes, ~10-15 minutes)
    train_agent(
        train_env, 
        agent, 
        visualize=False,           # True pour voir la progression
        train_episodes=50,        # Petit nombre pour tester
        training_batch_size=500
    )
    
    # OPTION B: Entraînement sérieux (décommentez pour production)
    # train_agent(
    #     train_env, 
    #     agent, 
    #     visualize=False,         # False = plus rapide
    #     train_episodes=20000,    # Entraînement complet
    #     training_batch_size=500
    # )
    
    print("\n✅ Training completed!")
    print("   Check the folder created (format: YYYY_MM_DD_HH_MM_Crypto_trader)")
    print("   Best models are saved as: REWARD_Crypto_trader_Actor.h5 and _Critic.h5")
    
    # =============================================================================
    # SECTION 2: TEST (COMMENTEZ SI VOUS VOULEZ JUSTE ENTRAÎNER)
    # =============================================================================
    
    # print("\n🧪 Testing the trained agent...")
    # test_env = EthereumEnv(
    #     test_df, 
    #     lookback_window_size=lookback_window_size
    # )
    
    # # Utilisez le dossier et le nom du modèle que l'entraînement a créé
    # test_agent(
    #     test_env, 
    #     agent, 
    #     visualize=True, 
    #     test_episodes=1,
    #     folder="2022_01_18_10_40_Crypto_trader",  # CHANGEZ avec votre dossier
    #     name="122580.55_Crypto_trader",            # CHANGEZ avec votre meilleur modèle
    #     comment=""
    # )
    
    # =============================================================================
    # SECTION 3: JEUX ALÉATOIRES (pour comparaison baseline)
    # =============================================================================
    
    # print("\n🎲 Running random agent for comparison...")
    # Random_games(
    #     train_env, 
    #     visualize=True,
    #     train_episodes=10
    # )