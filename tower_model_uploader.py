# tower_model_uploader.py
import requests # pip install requests
import zipfile
from pathlib import Path
import datetime
import logging
import json # Zum Lesen der app_settings.json für die Server-URL und API-Key
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Konfiguration für den Tower ---
TOWER_CONFIG_FILE = Path("./app_settings_tower.json")
DEFAULT_ADAPTER_DIR_TO_ZIP = Path("./codellama-7b-lora-adapters")

def load_tower_config() -> dict:
    config = {}
    if TOWER_CONFIG_FILE.exists():
        try:
            with TOWER_CONFIG_FILE.open("r", encoding="utf-8") as f:
                config = json.load(f)
            logger.info(f"Tower-Konfiguration aus '{TOWER_CONFIG_FILE}' geladen.")
        except Exception as e:
            logger.error(f"Fehler beim Laden der Tower-Konfiguration: {e}", exc_info=True)
    else:
        logger.warning(f"Keine Tower-Konfigurationsdatei '{TOWER_CONFIG_FILE}' gefunden. Bitte erstellen.")
    
    config.setdefault("server_api_url", "https://api.last-strawberry.com") # Standardwert, falls nicht in Datei
    config.setdefault("tower_api_key", "Ihr_API_Key_fuer_den_Tower")
    config.setdefault("default_adapter_dir", str(DEFAULT_ADAPTER_DIR_TO_ZIP))
    return config

def create_zip_archive(source_dir: Path, output_zip_path: Path) -> bool:
    if not source_dir.is_dir():
        logger.error(f"Quellverzeichnis '{source_dir}' nicht gefunden oder kein Verzeichnis.")
        return False
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for entry in source_dir.rglob('*'):
                zipf.write(entry, entry.relative_to(source_dir))
        logger.info(f"ZIP-Archiv '{output_zip_path}' erfolgreich erstellt aus '{source_dir}'.")
        return True
    except Exception as e:
        logger.error(f"Fehler beim Erstellen des ZIP-Archivs '{output_zip_path}': {e}", exc_info=True)
        return False

def upload_model_to_server(
    server_url_base: str,
    api_key: str,
    model_zip_path: Path,
    model_version: str,
    training_date_iso: Optional[str] = None,
    notes: Optional[str] = None
) -> bool:
    upload_url = f"{server_url_base.rstrip('/')}/model/upload"
    headers = {"X-API-Key": api_key}
    
    form_data = {
        "model_version": (None, model_version),
    }
    if training_date_iso:
        form_data["training_date"] = (None, training_date_iso)
    if notes:
        form_data["notes"] = (None, notes)

    try:
        with model_zip_path.open("rb") as f_zip:
            files = {'model_file': (model_zip_path.name, f_zip, 'application/zip')}
            
            logger.info(f"Lade Modell '{model_version}' ({model_zip_path.name}) hoch nach {upload_url}...")
            response = requests.post(upload_url, headers=headers, files=files, data=form_data, timeout=300)
            response.raise_for_status()
            
            response_json = response.json()
            logger.info(f"Modell-Upload erfolgreich. Server-Antwort: {response_json}")
            return True
            
    except requests.exceptions.HTTPError as http_err:
        error_detail = "Keine Details vom Server."
        if http_err.response is not None:
            try: error_detail = http_err.response.json().get("detail", str(http_err))
            except json.JSONDecodeError: error_detail = http_err.response.text[:200]
        logger.error(f"HTTP-Fehler beim Modell-Upload: {http_err.response.status_code if http_err.response is not None else 'N/A'} - {error_detail}", exc_info=True)
        return False
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Fehler bei der Serveranfrage für Modell-Upload: {req_err}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unerwarteter Fehler beim Modell-Upload: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    tower_config = load_tower_config()
    server_url = tower_config.get("server_api_url")
    tower_key = tower_config.get("tower_api_key")
    adapter_dir_str = tower_config.get("default_adapter_dir")

    # Korrigierte Bedingung:
    # Prüft nur, ob die Werte überhaupt vorhanden sind und ob der API-Key vom Standard abweicht.
    # Die server_url wird nicht mehr auf den Default-String geprüft.
    if not all([server_url, tower_key, adapter_dir_str]) or \
       tower_key == "Ihr_API_Key_fuer_den_Tower":
        logger.error("Bitte erstellen und konfigurieren Sie die Datei 'app_settings_tower.json' korrekt mit "
                     "'server_api_url', einem gültigen 'tower_api_key' und 'default_adapter_dir'. "
                     "Stellen Sie sicher, dass 'tower_api_key' nicht der Standardplatzhalter ist.")
    else:
        adapter_source_dir = Path(adapter_dir_str)
        
        now = datetime.datetime.now(datetime.timezone.utc)
        model_version_name = f"adapter_{now.strftime('%Y%m%d-%H%M%S')}"
        training_date_str = now.strftime('%Y-%m-%d')
        
        temp_zip_file = Path(f"./{model_version_name}.zip")

        if not adapter_source_dir.exists():
            logger.error(f"Adapter-Quellverzeichnis '{adapter_source_dir}' nicht gefunden!")
        elif create_zip_archive(adapter_source_dir, temp_zip_file):
            upload_notes = input(f"Optionale Notizen für Modell '{model_version_name}' (Enter für keine): ")
            
            success = upload_model_to_server(
                server_url_base=server_url,
                api_key=tower_key,
                model_zip_path=temp_zip_file,
                model_version=model_version_name,
                training_date_iso=training_date_str,
                notes=upload_notes if upload_notes.strip() else None
            )
            if success:
                logger.info("Modell erfolgreich zum Server hochgeladen.")
            else:
                logger.error("Modell-Upload zum Server fehlgeschlagen.")
            
            temp_zip_file.unlink(missing_ok=True)
        else:
            logger.error("Konnte kein ZIP-Archiv erstellen.")
