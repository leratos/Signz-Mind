# Klassen/server_connector.py
# -*- coding: utf-8 -*-

import logging
import requests  # Für HTTP-Requests, muss ggf. installiert werden (pip install requests)
import json
import datetime  # Für Timestamps
from pathlib import Path  # Hinzugefügt für Pfadoperationen
import shutil  # Hinzugefügt für Dateioperationen
import zipfile  # Hinzugefügt für ZIP-Dateien
from typing import TYPE_CHECKING, Optional, Tuple, Any, Dict

if TYPE_CHECKING:
    from .ui.ui_main_window import (
        MainWindow,
    )  # Für Zugriff auf runtime_settings und UI-Feedback
    from .data.data_manager import DataManager

logger = logging.getLogger(__name__)


class ServerConnector:
    def __init__(self, main_window: "MainWindow", data_manager: "DataManager"):
        self.main_window = main_window
        self.data_manager = data_manager
        logger.info("ServerConnector initialisiert.")

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """Hilfsmethode zum Abrufen von Konfigurationswerten aus MainWindow."""
        if hasattr(self.main_window, "runtime_settings"):
            return self.main_window.runtime_settings.get(key, default)
        logger.warning(
            f"Konnte Konfigurationsschlüssel '{key}' nicht finden, MainWindow.runtime_settings fehlt."
        )
        return default

    def upload_data_batch_to_server(self) -> Tuple[bool, str]:
        """
        Exportiert neue Daten aus dem DataManager und lädt sie zum Server hoch.
        Aktualisiert den 'last_successful_sync_timestamp_utc' bei Erfolg.

        Returns:
            Tuple[bool, str]: (Erfolg, Statusmeldung)
        """
        server_url_base = self._get_config_value("server_api_url")
        api_key = self._get_config_value("client_api_key")
        client_sync_id = self._get_config_value("client_sync_id", "unknown_client")
        last_sync_ts_iso = self._get_config_value("last_successful_sync_timestamp_utc")

        if (
            not server_url_base or server_url_base == "https://example.com/api"
        ):  # Prüft auch auf Default-Platzhalter
            msg = "Server-URL nicht (korrekt) in den Einstellungen konfiguriert."
            logger.error(msg)
            return False, msg
        if (
            not api_key or api_key == "PLEASE_CONFIGURE_CLIENT_API_KEY"
        ):  # Prüft auch auf Default-Platzhalter
            msg = "Client API-Key nicht (korrekt) in den Einstellungen konfiguriert."
            logger.error(msg)
            return False, msg

        if not self.data_manager:
            msg = "DataManager ist nicht verfügbar."
            logger.error(msg)
            return False, msg

        # Aktuellen UTC-Zeitpunkt für diesen Sync-Versuch festhalten
        current_sync_attempt_ts_utc = datetime.datetime.now(datetime.timezone.utc)

        logger.info(
            f"Starte Datenexport für Server-Upload (Client: {client_sync_id}, seit: {last_sync_ts_iso or 'Anfang'})."
        )
        jsonl_data_str = self.data_manager.export_new_data_as_jsonl(
            since_timestamp_utc_iso=last_sync_ts_iso
        )

        if jsonl_data_str is None:  # Fehler beim Export
            msg = "Fehler beim Exportieren der Daten aus der lokalen Datenbank."
            logger.error(msg)
            return False, msg

        if not jsonl_data_str.strip():  # Kein Fehler, aber keine neuen Daten
            msg = "Keine neuen Daten zum Synchronisieren gefunden."
            logger.info(msg)
            # Es ist kein Fehler, also True, aber mit entsprechender Nachricht.
            # Wir aktualisieren den Timestamp trotzdem, um unnötige zukünftige Abfragen für denselben Zeitraum zu vermeiden.
            if hasattr(self.main_window, "runtime_settings") and hasattr(
                self.main_window, "_save_runtime_settings"
            ):
                new_last_sync_ts_iso = (
                    current_sync_attempt_ts_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                    + "Z"
                )  # Millisekunden-Präzision
                self.main_window.runtime_settings[
                    "last_successful_sync_timestamp_utc"
                ] = new_last_sync_ts_iso
                self.main_window._save_runtime_settings()
                logger.info(
                    f"Keine neuen Daten, aber 'last_successful_sync_timestamp_utc' aktualisiert auf: {new_last_sync_ts_iso}"
                )
            return True, msg

        upload_url = f"{server_url_base.rstrip('/')}/data/upload_batch"
        headers = {"X-API-Key": api_key}
        params = {"client_id": client_sync_id}

        files = {
            "batch_file": (
                "data_batch.jsonl",
                jsonl_data_str.encode("utf-8"),
                "application/jsonl",
            )
        }

        self.main_window.statusBar().showMessage(
            f"Sende Daten-Batch ({len(jsonl_data_str.encode('utf-8'))} Bytes) an Server...",
            0,
        )
        try:
            logger.info(f"Sende Daten an: {upload_url} für Client-ID: {client_sync_id}")
            response = requests.post(
                upload_url, headers=headers, params=params, files=files, timeout=120
            )  # Timeout auf 120s erhöht
            response.raise_for_status()

            response_json = response.json()
            logger.info(f"Server-Antwort: {response_json}")

            new_last_sync_ts_iso = (
                current_sync_attempt_ts_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            )  # Millisekunden

            if hasattr(self.main_window, "runtime_settings") and hasattr(
                self.main_window, "_save_runtime_settings"
            ):
                self.main_window.runtime_settings[
                    "last_successful_sync_timestamp_utc"
                ] = new_last_sync_ts_iso
                self.main_window._save_runtime_settings()
                logger.info(
                    f"Neuer 'last_successful_sync_timestamp_utc' gespeichert: {new_last_sync_ts_iso}"
                )
            else:
                logger.warning(
                    "Konnte 'last_successful_sync_timestamp_utc' nicht in runtime_settings speichern."
                )

            msg = response_json.get("message", "Daten erfolgreich hochgeladen.")
            return True, msg

        except requests.exceptions.HTTPError as http_err:
            error_detail = "Keine Details vom Server."
            if http_err.response is not None:
                try:
                    error_detail = http_err.response.json().get("detail", str(http_err))
                except json.JSONDecodeError:
                    error_detail = http_err.response.text[
                        :200
                    ]  # Gekürzt, um UI nicht zu überfluten
            msg = f"HTTP-Fehler beim Upload: {http_err.response.status_code if http_err.response is not None else 'N/A'} - {error_detail}"
            logger.error(msg, exc_info=True)
            return False, msg
        except requests.exceptions.RequestException as req_err:
            msg = f"Fehler bei der Serveranfrage: {req_err}"
            logger.error(msg, exc_info=True)
            return False, msg
        except Exception as e:
            msg = f"Unerwarteter Fehler beim Daten-Upload: {e}"
            logger.error(msg, exc_info=True)
            return False, msg

    def get_latest_model_info_from_server(self) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Fragt den Server nach Informationen zur neuesten Modellversion.

        Returns:
            Tuple[Optional[Dict[str, Any]], str]: (Metadaten-Dict oder None bei Fehler, Statusmeldung)
        """
        server_url_base = self._get_config_value("server_api_url")
        api_key = self._get_config_value("client_api_key")

        if (
            not server_url_base
            or not api_key
            or api_key == "PLEASE_CONFIGURE_CLIENT_API_KEY"
        ):
            msg = "Server-URL oder API-Key nicht konfiguriert für Modell-Info-Abruf."
            logger.error(msg)
            return None, msg

        info_url = f"{server_url_base.rstrip('/')}/model/latest_version"
        headers = {"X-API-Key": api_key}

        logger.info(f"Frage neueste Modellinformationen von {info_url} ab...")
        try:
            response = requests.get(
                info_url, headers=headers, timeout=30
            )  # 30s Timeout
            response.raise_for_status()
            model_info = response.json()
            logger.info(
                f"Erfolgreich Modellinformationen erhalten: {model_info.get('latest_model_version')}"
            )
            return (
                model_info,
                f"Neueste Version: {model_info.get('latest_model_version', 'Unbekannt')}",
            )
        except requests.exceptions.HTTPError as http_err:
            error_detail = "Keine Details."
            if http_err.response is not None:
                try:
                    error_detail = http_err.response.json().get("detail", str(http_err))
                except json.JSONDecodeError:
                    error_detail = http_err.response.text[:200]
            msg = f"HTTP-Fehler bei Modell-Info: {http_err.response.status_code if http_err.response is not None else 'N/A'} - {error_detail}"
            logger.error(msg, exc_info=True)
            return None, msg
        except requests.exceptions.RequestException as req_err:
            msg = f"Fehler bei Serveranfrage für Modell-Info: {req_err}"
            logger.error(msg, exc_info=True)
            return None, msg
        except Exception as e:
            msg = f"Unerwarteter Fehler bei Modell-Info: {e}"
            logger.error(msg, exc_info=True)
            return None, msg

    def download_model_from_server(
        self, model_version_name: str, target_dir: Path
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Lädt ein spezifisches Modell-ZIP vom Server herunter und entpackt es.

        Args:
            model_version_name (str): Der Name der herunterzuladenden Modellversion.
            target_dir (Path): Das Verzeichnis, in das der Adapter entpackt werden soll.
                               (z.B. LORA_ADAPTER_PATH aus config.py)

        Returns:
            Tuple[bool, str, Optional[Path]]: (Erfolg, Statusmeldung, Pfad zum entpackten Adapter-Verzeichnis)
        """
        server_url_base = self._get_config_value("server_api_url")
        api_key = self._get_config_value("client_api_key")

        if (
            not server_url_base
            or not api_key
            or api_key == "PLEASE_CONFIGURE_CLIENT_API_KEY"
        ):
            msg = "Server-URL oder API-Key nicht konfiguriert für Modell-Download."
            logger.error(msg)
            return False, msg, None

        if not model_version_name:
            msg = "Kein Modellversionsname für Download angegeben."
            logger.error(msg)
            return False, msg, None

        download_url = (
            f"{server_url_base.rstrip('/')}/model/download/{model_version_name}"
        )
        headers = {"X-API-Key": api_key}

        # Temporärer Dateiname für das ZIP-Archiv
        temp_zip_path = (
            target_dir.parent / f"{model_version_name}_temp.zip"
        )  # Im übergeordneten Verzeichnis speichern

        logger.info(
            f"Starte Download von Modell '{model_version_name}' von {download_url}..."
        )
        self.main_window.statusBar().showMessage(
            f"Lade Modell '{model_version_name}' herunter...", 0
        )
        try:
            with requests.get(
                download_url, headers=headers, stream=True, timeout=600
            ) as r:  # 10 Min Timeout
                r.raise_for_status()
                with temp_zip_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.info(
                f"Modell '{model_version_name}' erfolgreich als '{temp_zip_path.name}' heruntergeladen."
            )

            # Entpacken des Adapters
            # Der Adapter wird normalerweise in ein Verzeichnis mit dem Namen des Adapters entpackt.
            # Wir gehen davon aus, dass LORA_ADAPTER_PATH das Zielverzeichnis ist, in dem die
            # Adapter-Dateien (adapter_config.json, adapter_model.bin etc.) direkt liegen sollen.
            # Wenn das ZIP den Adapter in einem Unterverzeichnis enthält, muss das hier angepasst werden.
            # Fürs Erste: Lösche altes Adapter-Verzeichnis (falls vorhanden) und entpacke neu.

            if target_dir.exists() and target_dir.is_dir():
                logger.info(f"Entferne existierendes Adapter-Verzeichnis: {target_dir}")
                shutil.rmtree(
                    target_dir
                )  # Vorsicht: Löscht das Verzeichnis und seinen Inhalt!
            target_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Entpacke '{temp_zip_path.name}' nach '{target_dir}'...")
            with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                zip_ref.extractall(target_dir)
            logger.info(
                f"Modell '{model_version_name}' erfolgreich nach '{target_dir}' entpackt."
            )

            # Speichere die heruntergeladene Version lokal (z.B. in runtime_settings)
            if hasattr(self.main_window, "runtime_settings") and hasattr(
                self.main_window, "_save_runtime_settings"
            ):
                self.main_window.runtime_settings["current_model_adapter_version"] = (
                    model_version_name
                )
                # Ggf. auch den Pfad zum Adapter speichern, falls er variabel ist
                self.main_window.runtime_settings["lora_adapter_path"] = str(
                    target_dir.resolve()
                )
                self.main_window._save_runtime_settings()
                logger.info(
                    f"Lokale Modellversion auf '{model_version_name}' aktualisiert und Pfad '{target_dir}' gespeichert."
                )

            return (
                True,
                f"Modell '{model_version_name}' erfolgreich heruntergeladen und entpackt.",
                target_dir,
            )

        except requests.exceptions.HTTPError as http_err:
            error_detail = "Keine Details."
            if http_err.response is not None:
                try:
                    error_detail = http_err.response.json().get("detail", str(http_err))
                except json.JSONDecodeError:
                    error_detail = http_err.response.text[:200]
            msg = f"HTTP-Fehler bei Modell-Download: {http_err.response.status_code if http_err.response is not None else 'N/A'} - {error_detail}"
            logger.error(msg, exc_info=True)
            return False, msg, None
        except requests.exceptions.RequestException as req_err:
            msg = f"Fehler bei Serveranfrage für Modell-Download: {req_err}"
            logger.error(msg, exc_info=True)
            return False, msg, None
        except zipfile.BadZipFile:
            msg = f"Fehler: Heruntergeladene Datei '{temp_zip_path.name}' ist kein gültiges ZIP-Archiv."
            logger.error(msg, exc_info=True)
            return False, msg, None
        except Exception as e:
            msg = f"Unerwarteter Fehler bei Modell-Download/Entpacken: {e}"
            logger.error(msg, exc_info=True)
            return False, msg, None
        finally:
            if temp_zip_path.exists():  # Temporäre ZIP-Datei löschen
                temp_zip_path.unlink(missing_ok=True)
