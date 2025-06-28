# tower_import_script.py
from pathlib import Path
import logging
from .Klassen.data.data_manager import DataManager 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    data_inbox_path = Path("./tower_data_inbox/").expanduser() # Passen Sie den Pfad an
    processed_path = data_inbox_path / "processed"
    error_path = data_inbox_path / "error"
    processed_path.mkdir(exist_ok=True)
    error_path.mkdir(exist_ok=True)

    # Pfad zur Datenbank auf dem Tower
    tower_db_path = Path("./tower_code_snippets.db").resolve()
    dm = DataManager(db_path=tower_db_path)
    if not dm.initialize_database():
        logger.error(f"Konnte die Tower-Datenbank unter {tower_db_path} nicht initialisieren.")
        return

    logger.info(f"Suche nach .jsonl Dateien in: {data_inbox_path}")
    jsonl_files = list(data_inbox_path.glob("*.jsonl"))

    if not jsonl_files:
        logger.info("Keine .jsonl Dateien im Inbox-Verzeichnis gefunden.")
        return

    for file_path in jsonl_files:
        logger.info(f"Verarbeite Datei: {file_path.name}...")
        try:
            stats = dm.import_data_from_jsonl_file(file_path)
            logger.info(f"Import-Statistik für {file_path.name}: {stats}")
            if stats.get("parse_errors", 0) > 0 or \
               stats.get("snippets_errors", 0) > 0 or \
               stats.get("feedback_errors", 0) > 0:
                logger.warning(f"Fehler beim Import von {file_path.name}. Datei wird nach '{error_path}' verschoben.")
                file_path.rename(error_path / file_path.name)
            else:
                file_path.rename(processed_path / file_path.name)
        except Exception as e:
            logger.error(f"Schwerwiegender Fehler bei der Verarbeitung von {file_path.name}: {e}", exc_info=True)
            try:
                file_path.rename(error_path / file_path.name)
            except Exception as move_e:
                logger.error(f"Konnte {file_path.name} nicht in Fehlerverzeichnis verschieben: {move_e}")

    dm.close_connection()
    logger.info("Alle Dateien verarbeitet.")

if __name__ == "__main__":
    main()