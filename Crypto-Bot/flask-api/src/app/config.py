from dotenv import load_dotenv
import os

# Charge le fichier .env
load_dotenv()

# Clés Binance et TAAPI
SECRET_KEY = os.getenv("SECRET_KEY")
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
TAAPI = os.getenv("TAAPI")

# Emails
DEV_EMAIL = os.getenv("DEV_EMAIL")
EMAIL_PASS = os.getenv("EMAIL_PASS")
PERSONAL_EMAIL = os.getenv("PERSONAL_EMAIL")
