# setup_server.py
# Ein interaktives Skript zur erstmaligen Einrichtung des Servers.
# Generiert jetzt auch den JWT SECRET_KEY.

import logging
from pathlib import Path
import secrets
import getpass
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Importiere das User-Modell und die Hash-Funktion aus main.py
try:
    from main import Base, User, get_password_hash, get_api_key_hash
except ImportError as e:
    print(f"Fehler: Konnte Module aus main.py nicht importieren. Stellen Sie sicher, dass dieses Skript im selben Verzeichnis ('app') liegt. Fehler: {e}")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Konfigurationsvariablen ---
CONFIG_FILE_PATH = Path("./server_config.py")
DEFAULT_DATABASE_URL = "sqlite:///./ki_editor_main.db"

def create_initial_config_and_admin():
    """
    Führt den interaktiven Setup-Prozess durch.
    """
    print("--- Einrichtung des KI-Editor Servers ---")

    if CONFIG_FILE_PATH.exists():
        print(f"Konfigurationsdatei '{CONFIG_FILE_PATH}' existiert bereits.")
        overwrite = input("Möchten Sie sie überschreiben und einen neuen Admin-Benutzer erstellen? (ja/nein): ").lower()
        if overwrite != 'ja':
            print("Einrichtung abgebrochen.")
            return

    # --- Konfigurationswerte abfragen ---
    db_url_input = input(f"Geben Sie die Datenbank-URL an oder drücken Sie Enter für den Standard [{DEFAULT_DATABASE_URL}]: ")
    db_url = db_url_input if db_url_input.strip() else DEFAULT_DATABASE_URL

    admin_username = ""
    while not admin_username:
        admin_username = input("Geben Sie einen Benutzernamen für den ersten Administrator ein: ").strip()

    admin_password = ""
    while not admin_password:
        admin_password = getpass.getpass(f"Geben Sie ein Passwort für den Admin-Benutzer '{admin_username}' ein: ")
    
    # --- Geheimen Schlüssel für JWT generieren ---
    jwt_secret_key = secrets.token_hex(32) # Generiert einen 64-stelligen hexadezimalen String
    logger.info("Sicherer JWT SECRET_KEY wurde generiert.")


    # --- server_config.py erstellen ---
    try:
        # Pfade müssen ggf. an Ihre finale Server-Struktur angepasst werden
        batch_path = "/var/www/vhosts/last-strawberry.com/ki_editor_api/data_batches/"
        model_path = "/var/www/vhosts/last-strawberry.com/ki_editor_api/models/"
        
        config_content = (
            f'# Diese Datei wurde automatisch durch setup_server.py erstellt.\n\n'
            f'# Geheimer Schlüssel für die JWT-Signierung. Niemals manuell ändern oder weitergeben!\n'
            f'SECRET_KEY = "{jwt_secret_key}"\n'
            f'ACCESS_TOKEN_EXPIRE_MINUTES = 60\n\n'
            f'# Pfad zur SQLite-Datenbank für Benutzer und andere Daten\n'
            f'DATABASE_URL = "{db_url}"\n\n'
            f'# Pfade für Uploads (müssen existieren und beschreibbar sein)\n'
            f'BATCH_UPLOAD_PATH = "{batch_path}"\n'
            f'MODEL_UPLOAD_PATH = "{model_path}"\n'
        )
        with CONFIG_FILE_PATH.open("w", encoding="utf-8") as f:
            f.write(config_content)
        print(f"Konfigurationsdatei '{CONFIG_FILE_PATH}' erfolgreich erstellt/überschrieben.")
    except Exception as e:
        print(f"Fehler beim Schreiben der Konfigurationsdatei: {e}")
        return

    # --- Datenbankverbindung herstellen und Tabelle erstellen ---
    try:
        engine = create_engine(db_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=engine)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db: Session = SessionLocal()
    except Exception as e:
        print(f"Fehler bei der Datenbankverbindung oder Tabellenerstellung: {e}")
        return

    # --- Admin-Benutzer in der Datenbank erstellen ---
    try:
        db_user = db.query(User).filter(User.username == admin_username).first()
        if db_user:
            print(f"Benutzer '{admin_username}' existiert bereits. Aktualisiere Passwort...")
            db_user.password_hash = get_password_hash(admin_password)
            db.commit()
            print("Passwort für bestehenden Admin-Benutzer wurde aktualisiert.")
        else:
            api_key_hash = get_api_key_hash(secrets.token_urlsafe(32)) # Erzeuge einen initialen API-Key
            
            admin_user = User(
                username=admin_username,
                password_hash=get_password_hash(admin_password),
                api_key_hash=api_key_hash,
                roles="ROLE_USER_ADMIN,ROLE_TOWER_TRAINER,ROLE_DATA_CONTRIBUTOR,ROLE_MODEL_CONSUMER",
                is_active=True,
                notes=f"Initialer Admin-Benutzer, erstellt am {datetime.datetime.now(datetime.timezone.utc).isoformat()}"
            )
            db.add(admin_user)
            db.commit()
            print(f"--- Admin-Benutzer '{admin_username}' erfolgreich in der Datenbank angelegt. ---")
            print("Sie können sich nun mit diesem Benutzer und Passwort in der Web-UI anmelden.")
            print("Ein initialer API-Key wurde ebenfalls generiert. Sie können bei Bedarf über die UI einen neuen generieren.")
    except Exception as e:
        print(f"Fehler beim Erstellen/Aktualisieren des Admin-Benutzers in der Datenbank: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    create_initial_config_and_admin()
