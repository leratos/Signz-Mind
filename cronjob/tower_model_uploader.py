# cronjob/tower_model_uploader.py
# -*- coding: utf-8 -*-

"""
Handles packaging and uploading of a trained model adapter to the server.

This script provides functionalities to:
- Load the tower's configuration.
- Create a ZIP archive from a directory.
- Upload the ZIP file to the server's model endpoint.
It can also be run standalone for manual uploads.
"""

import requests
import zipfile
from pathlib import Path
import datetime
import logging
import json
from typing import Optional, Dict, Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration file is expected to be in the same directory.
TOWER_CONFIG_FILE = Path(__file__).resolve().parent / "app_settings_tower.json"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTER_DIR_TO_ZIP = PROJECT_ROOT / "codellama-7b-lora-adapters"


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
                f"Fehler Laden Tower-Konfig '{TOWER_CONFIG_FILE}': {e}", exc_info=True
            )
    else:
        logger.warning(f"Keine Tower-Konfig-Datei '{TOWER_CONFIG_FILE}'.")

    config.setdefault("server_api_url", "https://api.last-strawberry.com")
    config.setdefault("tower_api_key", "BITTE_API_KEY_IN_CONFIG_SETZEN")
    config.setdefault("default_adapter_dir", str(DEFAULT_ADAPTER_DIR_TO_ZIP.resolve()))
    return config


def create_zip_archive(source_dir: Path, output_zip_path: Path) -> bool:
    """
    Creates a ZIP archive from the contents of a source directory.

    Args:
        source_dir: The directory to be zipped.
        output_zip_path: The path for the output ZIP file.

    Returns:
        True if the archive was created successfully, False otherwise.
    """
    if not source_dir.is_dir():
        logger.error(
            f"Quellverzeichnis '{source_dir}' nicht gefunden oder kein Verzeichnis."
        )
        return False
    try:
        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for entry in source_dir.rglob("*"):
                zipf.write(entry, entry.relative_to(source_dir))
        logger.info(
            f"ZIP-Archiv '{output_zip_path}' erfolgreich erstellt aus '{source_dir}'."
        )
        return True
    except Exception as e:
        logger.error(
            f"Fehler beim Erstellen des ZIP-Archivs '{output_zip_path}': {e}",
            exc_info=True,
        )
        return False


def upload_model_to_server(
    server_url_base: str,
    api_key: str,
    model_zip_path: Path,
    model_version: str,
    training_date_iso: Optional[str] = None,
    notes: Optional[str] = None,
) -> bool:
    """
    Uploads a model ZIP file to the central server.

    Args:
        server_url_base: The base URL of the server API.
        api_key: The API key for authentication.
        model_zip_path: The path to the model's ZIP archive.
        model_version: A unique version name for the model.
        training_date_iso: The training date in ISO format (YYYY-MM-DD).
        notes: Optional notes about the model version.

    Returns:
        True if the upload was successful, False otherwise.
    """
    upload_url = f"{server_url_base.rstrip('/')}/model/upload"
    headers = {"X-API-Key": api_key}
    # Metadata is sent as multipart/form-data along with the file
    form_data = {"model_version": (None, model_version)}
    if training_date_iso:
        form_data["training_date"] = (None, training_date_iso)
    if notes:
        form_data["notes"] = (None, notes)
    try:
        with model_zip_path.open("rb") as f_zip:
            files = {"model_file": (model_zip_path.name, f_zip, "application/zip")}
            logger.info(
                f"Lade Modell '{model_version}' ({model_zip_path.name}) hoch nach {upload_url}..."
            )
            response = requests.post(
                upload_url, headers=headers, files=files, data=form_data, timeout=300
            )  # 5 Min Timeout
            response.raise_for_status()
            response_json = response.json()
            logger.info(f"Modell-Upload erfolgreich. Server-Antwort: {response_json}")
            return True
    except requests.exceptions.HTTPError as http_err:
        error_detail = "Keine Details."
        if http_err.response is not None:
            try:
                error_detail = http_err.response.json().get("detail", str(http_err))
            except json.JSONDecodeError:
                error_detail = http_err.response.text[:500]  # Mehr Kontext
        logger.error(
            f"HTTP-Fehler {http_err.response.status_code if http_err.response is not None else 'N/A'} beim Modell-Upload zu {upload_url}: {error_detail}",
            exc_info=False,
        )  # exc_info=False, da Fehler oft vom Server kommt
        return False
    except requests.exceptions.RequestException as req_err:
        logger.error(
            f"Fehler bei der Serveranfrage für Modell-Upload ({upload_url}): {req_err}",
            exc_info=True,
        )
        return False
    except Exception as e:
        logger.error(
            f"Unerwarteter Fehler beim Modell-Upload ({upload_url}): {e}", exc_info=True
        )
        return False


if __name__ == "__main__":
    # This block allows the script to be run manually for a one-off upload.
    tower_config = load_tower_config()
    server_url = tower_config.get("server_api_url")
    tower_key = tower_config.get("tower_api_key")
    adapter_dir_to_zip_str = tower_config.get("default_adapter_dir")

    if (
        not all([server_url, tower_key, adapter_dir_to_zip_str])
        or tower_key == "BITTE_API_KEY_IN_CONFIG_SETZEN"
    ):
        logger.error(
            "Bitte 'app_settings_tower.json' korrekt konfigurieren mit 'server_api_url', 'tower_api_key' und 'default_adapter_dir'."
        )
    else:
        adapter_source_dir = Path(adapter_dir_to_zip_str)
        if (
            not adapter_source_dir.is_absolute()
        ):  # Sicherstellen, dass der Pfad absolut ist oder relativ zum Projekt-Root
            adapter_source_dir = (PROJECT_ROOT / adapter_dir_to_zip_str).resolve()

        now = datetime.datetime.now(datetime.timezone.utc)
        model_version_name = f"manual_upload_adapter_{now.strftime('%Y%m%d-%H%M%S')}"
        training_date_str = now.strftime("%Y-%m-%d")

        # Temporäres ZIP im cronjob-Ordner erstellen
        temp_zip_file = Path(__file__).resolve().parent / f"{model_version_name}.zip"

        if not adapter_source_dir.exists():
            logger.error(
                f"Adapter-Quellverzeichnis '{adapter_source_dir}' nicht gefunden!"
            )
        elif create_zip_archive(adapter_source_dir, temp_zip_file):
            upload_notes = input(
                f"Optionale Notizen für Modell '{model_version_name}' (Enter für keine): "
            )
            success = upload_model_to_server(
                server_url,
                tower_key,
                temp_zip_file,
                model_version_name,
                training_date_str,
                upload_notes if upload_notes.strip() else None,
            )
            if success:
                logger.info("Modell erfolgreich zum Server hochgeladen.")
            else:
                logger.error("Modell-Upload zum Server fehlgeschlagen.")
            temp_zip_file.unlink(missing_ok=True)
        else:
            logger.error("Konnte kein ZIP-Archiv erstellen.")
