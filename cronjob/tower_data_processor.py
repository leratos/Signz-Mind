# cronjob/tower_data_processor.py
# -*- coding: utf-8 -*-

"""
Handles the processing of new data from the central server.

This script defines functions to:
- Load the tower's local configuration.
- List new data batches available on the server.
- Download a specific data batch.
- Mark a batch as processed on the server after successful import.
- The main entry point `process_new_data_from_server` orchestrates this flow.
"""

import requests
from pathlib import Path
import logging
import json
import sys
import shutil
from typing import Optional, List, Dict, Any

# Ensure the project root is in the system path for package imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ..Klassen.data.data_manager import DataManager
except ImportError as e:
    logging.critical(
        f"Konnte DataManager nicht importieren. Sicherstellen, dass K-Training im PYTHONPATH ist oder das Skript aus K-Training gestartet wird. Fehler: {e}"
    )
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration is expected to be in the same directory as this script.
TOWER_CONFIG_FILE = Path(__file__).resolve().parent / "app_settings_tower.json"
DOWNLOAD_TEMP_DIR = Path(__file__).resolve().parent / "tower_data_inbox_temp"
DOWNLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_TOWER_DB_PATH = PROJECT_ROOT / "tower_code_snippets.db"


def load_tower_config() -> Dict[str, Any]:
    """Loads the tower's configuration from a JSON file."""
    config = {}
    if TOWER_CONFIG_FILE.exists():
        try:
            with TOWER_CONFIG_FILE.open("r", encoding="utf-8") as f:
                config = json.load(f)
            logger.info(f"Tower-Konfiguration aus '{TOWER_CONFIG_FILE}' geladen.")
        except Exception as e:
            logger.error(
                f"Fehler beim Laden der Tower-Konfiguration '{TOWER_CONFIG_FILE}': {e}",
                exc_info=True,
            )
    else:
        logger.warning(
            f"Keine Tower-Konfigurationsdatei '{TOWER_CONFIG_FILE}' gefunden. Verwende Defaults."
        )

    config.setdefault("server_api_url", "https://api.last-strawberry.com")
    config.setdefault("tower_api_key", "BITTE_API_KEY_IN_CONFIG_SETZEN")
    config.setdefault("tower_db_path", str(DEFAULT_TOWER_DB_PATH.resolve()))
    return config


def list_new_batches_from_server(
    server_url_base: str, api_key: str
) -> Optional[List[str]]:
    """Fetches the list of new, unprocessed data batches from the server."""
    list_url = f"{server_url_base.rstrip('/')}/data/list_new_batches"
    headers = {"X-API-Key": api_key, "accept": "application/json"}
    try:
        logger.info(f"Frage neue Batch-Dateien von {list_url} ab...")
        response = requests.get(list_url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        batch_files = data.get("new_batch_files", [])
        logger.info(f"{len(batch_files)} neue Batch-Dateien gefunden: {batch_files}")
        return batch_files
    except requests.exceptions.RequestException as e:
        logger.error(
            f"Fehler beim Abrufen der Batch-Liste von {list_url}: {e}", exc_info=True
        )
        return None
    except json.JSONDecodeError as e:
        logger.error(
            f"Fehler beim Parsen der Batch-Listen-Antwort von {list_url}: {e}. Antworttext: {response.text[:200]}"
        )
        return None


def download_batch_file_from_server(
    server_url_base: str, api_key: str, batch_filename: str, target_path: Path
) -> bool:
    """Downloads a single batch file from the server."""
    download_url = f"{server_url_base.rstrip('/')}/data/download_batch/{batch_filename}"
    headers = {"X-API-Key": api_key, "accept": "application/jsonl"}
    try:
        logger.info(
            f"Lade Batch-Datei '{batch_filename}' von {download_url} herunter..."
        )
        with requests.get(download_url, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            with target_path.open("wb") as f:
                shutil.copyfileobj(r.raw, f)
        logger.info(
            f"Batch-Datei '{batch_filename}' erfolgreich nach '{target_path}' heruntergeladen."
        )
        return True
    except requests.exceptions.RequestException as e:
        logger.error(
            f"Fehler beim Herunterladen der Batch-Datei '{batch_filename}' von {download_url}: {e}",
            exc_info=True,
        )
        return False


def mark_batch_processed_on_server(
    server_url_base: str, api_key: str, batch_filename: str
) -> bool:
    """Notifies the server that a batch has been successfully processed."""
    mark_url = (
        f"{server_url_base.rstrip('/')}/data/mark_batch_processed/{batch_filename}"
    )
    headers = {"X-API-Key": api_key, "accept": "application/json"}
    try:
        logger.info(
            f"Markiere Batch-Datei '{batch_filename}' auf Server als verarbeitet ({mark_url})..."
        )
        response = requests.post(mark_url, headers=headers, timeout=30)
        response.raise_for_status()
        logger.info(
            f"Batch-Datei '{batch_filename}' erfolgreich als verarbeitet markiert. Server: {response.json().get('message')}"
        )
        return True
    except requests.exceptions.RequestException as e:
        logger.error(
            f"Fehler beim Markieren der Batch-Datei '{batch_filename}' als verarbeitet auf {mark_url}: {e}",
            exc_info=True,
        )
        return False


def process_new_data_from_server() -> bool:
    """
    Main function to orchestrate downloading and importing of new data batches.

    Returns:
        True if at least one new batch was successfully processed, False otherwise.
    """
    if DataManager is None:
        logger.critical("DataManager Klasse nicht verfügbar. Import fehlgeschlagen.")
        return False

    tower_config = load_tower_config()
    server_url = tower_config.get("server_api_url")
    tower_key = tower_config.get("tower_api_key")
    tower_db_path_str = tower_config.get("tower_db_path")

    if (
        not server_url
        or not tower_key
        or tower_key == "BITTE_API_KEY_IN_CONFIG_SETZEN"
        or not tower_db_path_str
    ):
        logger.error(
            "Server-URL, Tower API-Key oder DB-Pfad nicht korrekt in 'app_settings_tower.json' konfiguriert."
        )
        return False

    dm_tower = DataManager(db_path=Path(tower_db_path_str))
    if not dm_tower.initialize_database():
        logger.error(
            f"Konnte die Tower-Datenbank unter {tower_db_path_str} nicht initialisieren."
        )
        return False

    new_batch_files = list_new_batches_from_server(server_url, tower_key)

    if new_batch_files is None:
        logger.error(
            "Abbruch des Imports aufgrund eines Fehlers beim Abrufen der Batch-Liste."
        )
        dm_tower.close_connection()
        return False

    if not new_batch_files:
        logger.info("Keine neuen Daten-Batches vom Server zum Importieren vorhanden.")
        dm_tower.close_connection()
        return False

    new_data_was_successfully_processed = False
    for batch_filename in new_batch_files:
        logger.info(f"--- Beginne Verarbeitung von Batch: {batch_filename} ---")
        temp_download_path = DOWNLOAD_TEMP_DIR / batch_filename

        if download_batch_file_from_server(
            server_url, tower_key, batch_filename, temp_download_path
        ):
            logger.info(
                f"Importiere Daten aus heruntergeladenem Batch: {temp_download_path}"
            )
            stats = dm_tower.import_data_from_jsonl_file(temp_download_path)
            logger.info(f"Import-Statistik für {batch_filename}: {stats}")

            num_errors_in_batch = (
                stats.get("parse_errors", 0)
                + stats.get("snippets_errors", 0)
                + stats.get("feedback_errors", 0)
            )

            # Only mark as processed if the entire batch was imported without errors
            if num_errors_in_batch == 0:
                if mark_batch_processed_on_server(
                    server_url, tower_key, batch_filename
                ):
                    logger.info(
                        f"Batch {batch_filename} erfolgreich verarbeitet und auf Server markiert."
                    )
                    new_data_was_successfully_processed = True
                else:
                    logger.error(
                        f"Fehler beim Markieren von Batch {batch_filename} als verarbeitet. Wird beim nächsten Mal erneut versucht."
                    )
            else:
                logger.warning(
                    f"Batch {batch_filename} enthielt Import-Fehler ({num_errors_in_batch} Fehler). "
                    "Wird nicht als verarbeitet auf dem Server markiert, um manuelle Prüfung zu ermöglichen."
                )
            temp_download_path.unlink(missing_ok=True)
        else:
            logger.error(
                f"Download von Batch {batch_filename} fehlgeschlagen. Wird übersprungen."
            )
        logger.info(f"--- Beende Verarbeitung von Batch: {batch_filename} ---")

    dm_tower.close_connection()
    logger.info(
        f"Automatischer Datenimport abgeschlossen. Status 'new_data_was_successfully_processed': {new_data_was_successfully_processed}"
    )
    return new_data_was_successfully_processed


if __name__ == "__main__":
    if process_new_data_from_server is not None:
        process_new_data_from_server()
    else:
        logger.critical(
            "Hauptfunktion process_new_data_from_server ist nicht verfügbar."
        )
