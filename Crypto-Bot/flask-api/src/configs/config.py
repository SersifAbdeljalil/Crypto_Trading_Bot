from dotenv import dotenv_values
import os

path = "../../.env"

if os.path.exists(path):
    env = dotenv_values(path)
    print(f"The path {path} exists.")
    
    # Liste des clés requises
    required_keys = [
        'API_KEY',
        'API_SECRET',
        'TEST_API_KEY',
        'TEST_SECRET_KEY',
        'PERSONAL_EMAIL',
        'DEV_EMAIL',
        'EMAIL_PASS',
        'TAAPI'
    ]
    
    # Vérifier les clés manquantes
    missing_keys = [key for key in required_keys if key not in env or not env[key]]
    
    if missing_keys:
        print(f"\n❌ ERROR: Missing required environment variables in .env file:")
        for key in missing_keys:
            print(f"   - {key}")
        print(f"\nPlease add these variables to your {path} file.\n")
        raise KeyError(f"Missing required environment variables: {', '.join(missing_keys)}")
    
    # Charger les variables seulement si toutes sont présentes
    API_KEY = env['API_KEY']
    API_SECRET = env['API_SECRET']
    TEST_API_KEY = env['TEST_API_KEY']
    TEST_SECRET_KEY = env['TEST_SECRET_KEY']
    PERSONAL_EMAIL = env['PERSONAL_EMAIL']
    DEV_EMAIL = env['DEV_EMAIL']
    EMAIL_PASS = env['EMAIL_PASS']
    TAAPI = env['TAAPI']
    
    print("✅ All environment variables loaded successfully.\n")
    
else:
    print(f"❌ ERROR: The path {path} does not exist.")
    print("Please create a .env file with the required variables.\n")
    raise FileNotFoundError(f"Environment file not found: {path}")