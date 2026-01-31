"""
Script de correction automatique - Crypto Bot
Copie les fichiers au bon endroit et vérifie la configuration
"""
import shutil
from pathlib import Path

print("="*60)
print("🔧 CORRECTION AUTOMATIQUE - CRYPTO BOT")
print("="*60)

# Chemins
source_dir = Path(r"C:\BC\Crypto-Bot\flask-api\app\output_data")
target_dir = Path(r"C:\BC\Crypto-Bot\output_data")

# Créer le dossier cible s'il n'existe pas
target_dir.mkdir(parents=True, exist_ok=True)

# Liste des fichiers à copier
files_to_copy = [
    "transaction_history.csv",
    "topNews.csv",
    "allNews.csv",
    "cryptoanalysis_data.csv",
    "ETH_hourly_data.csv",
    "feedingHistoryData.csv"
]

print(f"\n📁 Source: {source_dir}")
print(f"📁 Target: {target_dir}\n")

# Copier chaque fichier
copied = 0
for filename in files_to_copy:
    source_file = source_dir / filename
    target_file = target_dir / filename
    
    if source_file.exists():
        try:
            shutil.copy2(source_file, target_file)
            print(f"✓ Copié: {filename}")
            copied += 1
        except Exception as e:
            print(f"✗ Erreur pour {filename}: {e}")
    else:
        print(f"⚠ Fichier non trouvé: {filename}")

print(f"\n✅ {copied}/{len(files_to_copy)} fichiers copiés")

# Vérifier le résultat
print("\n📊 Vérification:")
for filename in files_to_copy:
    target_file = target_dir / filename
    if target_file.exists():
        size = target_file.stat().st_size
        print(f"  ✓ {filename} ({size:,} bytes)")
    else:
        print(f"  ✗ {filename} (manquant)")

print("\n" + "="*60)
print("✅ CORRECTION TERMINÉE")
print("="*60)
print("\nRedémarrez Flask avec: python app.py")