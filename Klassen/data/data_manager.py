# Klassen/data_manager.py
# -*- coding: utf-8 -*-

"""
Verwaltet Datenbankinteraktionen und die Vorbereitung von Daten für das Training.
Stellt Methoden zum Initialisieren der Datenbank, Laden von Code-Snippets
und Abrufen von Daten mit optionalen Qualitätsfiltern bereit.
NEU: Tabelle und Methode zum Speichern von KI-Feedback.
NEU: Methode zum Exportieren von Daten als JSON-Lines.
"""

import sqlite3
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any
import logging
import json  # Für JSON-Lines Export
import datetime  # Für Timestamps
import uuid  # Für eindeutige IDs

# Importiere Konfiguration relativ zum aktuellen Paket
from ..core import config as global_config

logger = logging.getLogger(__name__)


class DataManager:
    """
    Verwaltet Datenbankinteraktionen zum Speichern und Abrufen von Code-Snippets
    sowie zum Speichern von Feedback zu KI-Aktionen und deren Export.
    """

    DEFAULT_QUALITY_LABEL: str = "neutral"

    def __init__(self, db_path: Union[str, Path, None] = None):
        if db_path is None:
            if hasattr(global_config, "DB_PATH"):
                self.db_path: Path = Path(global_config.DB_PATH)
            else:
                logger.warning(
                    "DB_PATH nicht in global_config gefunden, verwende Standard 'code_snippets.db'"
                )
                self.db_path = Path("code_snippets.db")
        else:
            self.db_path = Path(db_path)

        self.conn: Optional[sqlite3.Connection] = None
        logger.info(f"DataManager initialisiert mit db_path: {self.db_path}")

    def _connect(self) -> Optional[sqlite3.Connection]:
        if self.conn is None:
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
                logger.info(f"Datenbankverbindung zu {self.db_path} hergestellt.")
            except sqlite3.Error as e:
                logger.error(
                    f"Fehler beim Verbinden mit der Datenbank {self.db_path}: {e}"
                )
                self.conn = None
            except Exception as e:
                logger.error(
                    f"Unerwarteter Fehler beim Verbindungsaufbau zu {self.db_path}: {e}"
                )
                self.conn = None
        return self.conn

    def close_connection(self):
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
                logger.info(f"Datenbankverbindung zu {self.db_path} geschlossen.")
            except sqlite3.Error as e:
                logger.error(f"Fehler beim Schließen der Datenbankverbindung: {e}")

    def initialize_database(self) -> bool:
        conn = self._connect()
        if not conn:
            return False

        try:
            cursor = conn.cursor()

            # --- Tabelle 'snippets' ---
            # Prüfen und ggf. Spalte 'uuid' hinzufügen
            cursor.execute("PRAGMA table_info(snippets)")
            snippet_columns = {info[1]: info for info in cursor.fetchall()}
            if "uuid" not in snippet_columns:
                logger.info(
                    "Spalte 'uuid' fehlt in 'snippets', versuche sie hinzuzufügen."
                )
                try:
                    # Hinzufügen ohne UNIQUE Constraint, da alte Einträge keine UUID haben.
                    # Neue Einträge sollten eine bekommen.
                    cursor.execute("ALTER TABLE snippets ADD COLUMN uuid TEXT")
                    conn.commit()  # Commit nach ALTER TABLE
                    logger.info("Spalte 'uuid' zur Tabelle 'snippets' hinzugefügt.")
                except sqlite3.OperationalError as alter_err:
                    logger.warning(
                        f"Konnte 'uuid' nicht zu 'snippets' hinzufügen (existiert ggf.): {alter_err}"
                    )

            if "quality_label" not in snippet_columns:
                logger.info(
                    "Spalte 'quality_label' fehlt in 'snippets', versuche sie hinzuzufügen."
                )
                try:
                    cursor.execute(
                        f"ALTER TABLE snippets ADD COLUMN quality_label TEXT DEFAULT '{self.DEFAULT_QUALITY_LABEL}'"
                    )
                    conn.commit()
                    logger.info(
                        "Spalte 'quality_label' zur Tabelle 'snippets' hinzugefügt."
                    )
                except sqlite3.OperationalError as alter_err:
                    logger.warning(
                        f"Konnte 'quality_label' nicht zu 'snippets' hinzufügen: {alter_err}"
                    )

            # Spalte für letzten Sync-Timestamp oder Modifikations-Timestamp
            if "last_modified_utc" not in snippet_columns:
                logger.info(
                    "Spalte 'last_modified_utc' fehlt in 'snippets', versuche sie hinzuzufügen."
                )
                try:
                    # Default CURRENT_TIMESTAMP setzt den Wert beim Einfügen, was gut ist.
                    # Wir brauchen aber auch einen Trigger für Updates.
                    cursor.execute(
                        "ALTER TABLE snippets ADD COLUMN last_modified_utc TEXT"
                    )  # Als TEXT für ISO Format
                    conn.commit()
                    logger.info("Spalte 'last_modified_utc' zu 'snippets' hinzugefügt.")
                    # Fülle initial für bestehende Einträge (optional, aber gut für ersten Sync)
                    cursor.execute(
                        "UPDATE snippets SET last_modified_utc = strftime('%Y-%m-%dT%H:%M:%fZ', timestamp) WHERE last_modified_utc IS NULL"
                    )
                    conn.commit()
                except sqlite3.OperationalError as alter_err:
                    logger.warning(
                        f"Konnte 'last_modified_utc' nicht zu 'snippets' hinzufügen: {alter_err}"
                    )

            cursor.execute(
                f"""CREATE TABLE IF NOT EXISTS snippets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT, -- Wird beim Erstellen gesetzt, kann UNIQUE sein für neue DBs
                    path TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, -- Erstellungszeitpunkt
                    quality_label TEXT DEFAULT '{self.DEFAULT_QUALITY_LABEL}',
                    last_modified_utc TEXT -- UTC ISO8601 String, wann zuletzt geändert/für Sync relevant
                )"""
            )
            # Trigger, um last_modified_utc bei UPDATE zu aktualisieren
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS update_snippets_last_modified
                AFTER UPDATE ON snippets
                FOR EACH ROW
                BEGIN
                    UPDATE snippets SET last_modified_utc = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE id = OLD.id;
                END;
            """
            )
            # Trigger, um last_modified_utc und uuid bei INSERT zu setzen
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS insert_snippets_defaults
                AFTER INSERT ON snippets
                FOR EACH ROW
                BEGIN
                    UPDATE snippets SET
                        last_modified_utc = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                        uuid = CASE WHEN NEW.uuid IS NULL THEN lower(hex(randomblob(16))) ELSE NEW.uuid END
                    WHERE id = NEW.id;
                END;
            """
            )
            logger.debug(
                "Tabelle 'snippets' und Trigger erfolgreich initialisiert/verifiziert."
            )

            # --- Tabelle 'ki_feedback' ---
            cursor.execute("PRAGMA table_info(ki_feedback)")
            feedback_columns = {info[1]: info for info in cursor.fetchall()}
            if "uuid" not in feedback_columns:
                logger.info(
                    "Spalte 'uuid' fehlt in 'ki_feedback', versuche sie hinzuzufügen."
                )
                try:
                    cursor.execute(
                        "ALTER TABLE ki_feedback ADD COLUMN uuid TEXT"
                    )  # UNIQUE später, wenn alle Daten UUIDs haben
                    conn.commit()
                    logger.info("Spalte 'uuid' zur Tabelle 'ki_feedback' hinzugefügt.")
                except sqlite3.OperationalError as alter_err:
                    logger.warning(
                        f"Konnte 'uuid' nicht zu 'ki_feedback' hinzufügen: {alter_err}"
                    )

            if (
                "last_modified_utc" not in feedback_columns
            ):  # Feedback ist meist append-only, aber für Konsistenz
                logger.info(
                    "Spalte 'last_modified_utc' fehlt in 'ki_feedback', versuche sie hinzuzufügen."
                )
                try:
                    cursor.execute(
                        "ALTER TABLE ki_feedback ADD COLUMN last_modified_utc TEXT"
                    )
                    conn.commit()
                    logger.info(
                        "Spalte 'last_modified_utc' zu 'ki_feedback' hinzugefügt."
                    )
                    cursor.execute(
                        "UPDATE ki_feedback SET last_modified_utc = strftime('%Y-%m-%dT%H:%M:%fZ', timestamp) WHERE last_modified_utc IS NULL"
                    )
                    conn.commit()
                except sqlite3.OperationalError as alter_err:
                    logger.warning(
                        f"Konnte 'last_modified_utc' nicht zu 'ki_feedback' hinzufügen: {alter_err}"
                    )

            cursor.execute(
                """CREATE TABLE IF NOT EXISTS ki_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT, -- Wird beim Erstellen gesetzt
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    source_code_context TEXT NOT NULL,
                    ai_action_type TEXT NOT NULL,
                    ai_generated_output TEXT NOT NULL,
                    user_feedback_type TEXT NOT NULL,
                    original_file_path TEXT,
                    notes TEXT,
                    last_modified_utc TEXT -- UTC ISO8601 String
                )"""
            )
            # Trigger für ki_feedback
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS insert_ki_feedback_defaults
                AFTER INSERT ON ki_feedback
                FOR EACH ROW
                BEGIN
                    UPDATE ki_feedback SET
                        last_modified_utc = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                        uuid = CASE WHEN NEW.uuid IS NULL THEN lower(hex(randomblob(16))) ELSE NEW.uuid END
                    WHERE id = NEW.id;
                END;
            """
            )
            logger.debug(
                "Tabelle 'ki_feedback' und Trigger erfolgreich initialisiert/verifiziert."
            )

            conn.commit()
            logger.info("Datenbanktabellen erfolgreich initialisiert/verifiziert.")
            return True
        except sqlite3.Error as e:
            logger.error(
                f"Fehler bei der Initialisierung der Datenbanktabellen: {e}",
                exc_info=True,
            )
            return False

    def add_ki_feedback(
        self,
        source_code_context: str,
        ai_action_type: str,
        ai_generated_output: str,
        user_feedback_type: str,
        original_file_path: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> bool:
        conn = self._connect()
        if not conn:
            logger.error("Kann KI-Feedback nicht hinzufügen, keine DB-Verbindung.")
            return False
        try:
            cursor = conn.cursor()
            # UUID und last_modified_utc werden durch Trigger gesetzt
            cursor.execute(
                """INSERT INTO ki_feedback
                   (source_code_context, ai_action_type, ai_generated_output,
                    user_feedback_type, original_file_path, notes)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    source_code_context,
                    ai_action_type,
                    ai_generated_output,
                    user_feedback_type,
                    original_file_path,
                    notes,
                ),
            )
            conn.commit()
            logger.info(
                f"KI-Feedback hinzugefügt: Typ='{ai_action_type}', "
                f"Feedback='{user_feedback_type}', File='{original_file_path or 'N/A'}'"
            )
            return True
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Hinzufügen von KI-Feedback: {e}", exc_info=True)
            return False

    def load_files_into_db(
        self, file_paths: List[str], default_quality: str = DEFAULT_QUALITY_LABEL
    ) -> Tuple[int, int]:
        conn = self._connect()
        if not conn:
            logger.error("Kann Dateien nicht laden, keine DB-Verbindung.")
            return 0, len(file_paths)

        added_count = 0
        skipped_count = 0
        cursor = conn.cursor()
        for p_str in file_paths:
            try:
                p = Path(p_str)
                if not p.is_file():
                    logger.warning(f"Pfad ist keine Datei: {p_str}, wird übersprungen.")
                    skipped_count += 1
                    continue

                text = p.read_text(encoding="utf-8", errors="replace")
                # UUID und last_modified_utc werden durch Trigger gesetzt/aktualisiert
                # Beim Ersetzen wird der Timestamp beibehalten, aber last_modified_utc durch Trigger aktualisiert.
                cursor.execute(
                    """INSERT OR REPLACE INTO snippets (path, content, quality_label, uuid, timestamp, last_modified_utc)
                       VALUES (?, ?, ?,
                               COALESCE((SELECT uuid FROM snippets WHERE path = ?), lower(hex(randomblob(16)))),
                               COALESCE((SELECT timestamp FROM snippets WHERE path = ?), CURRENT_TIMESTAMP),
                               strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                              )""",
                    (str(p), text, default_quality, str(p), str(p)),
                )
                added_count += 1
                logger.info(
                    f"Datei '{p_str}' in DB geladen/aktualisiert mit Qualität '{default_quality}'."
                )
            except FileNotFoundError:
                logger.warning(f"Datei nicht gefunden: {p_str}, wird übersprungen.")
                skipped_count += 1
            except Exception as e:
                logger.warning(
                    f"Konnte Datei nicht lesen/einfügen {p_str}: {e}", exc_info=True
                )
                skipped_count += 1
        try:
            conn.commit()
            logger.info(
                f"DB-Commit nach Laden. Hinzugefügt/Ersetzt: {added_count}, Übersprungen: {skipped_count}"
            )
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Committen der Änderungen in die Datenbank: {e}")
        return added_count, skipped_count

    def get_all_snippets_text(
        self, quality_filter: Optional[List[str]] = None
    ) -> Optional[str]:
        conn = self._connect()
        if not conn:
            logger.error("Kann Snippets nicht abrufen, keine DB-Verbindung.")
            return None
        try:
            cursor = conn.cursor()
            query = "SELECT content FROM snippets"
            params: List[str] = []

            if (
                quality_filter
                and isinstance(quality_filter, list)
                and len(quality_filter) > 0
            ):
                placeholders = ",".join("?" * len(quality_filter))
                query += f" WHERE quality_label IN ({placeholders})"
                params.extend(quality_filter)
            query += " ORDER BY id"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            if not rows:
                logger.info(
                    f"Keine Snippets in der Datenbank gefunden (Filter: {quality_filter})."
                )
                return ""

            all_text = "\n\n".join(row[0] for row in rows if isinstance(row[0], str))
            logger.info(
                f"{len(rows)} Snippets abgerufen (Filter: {quality_filter}), "
                f"Gesamtlänge des Textes: {len(all_text)}"
            )
            return all_text
        except sqlite3.Error as e:
            logger.error(
                f"Fehler beim Abrufen der Snippets aus der DB: {e}", exc_info=True
            )
            return None

    def update_snippet_quality(self, path_str: str, quality_label: str) -> bool:
        conn = self._connect()
        if not conn:
            logger.error(
                f"Kann Qualität für {path_str} nicht aktualisieren, keine DB-Verbindung."
            )
            return False
        try:
            cursor = conn.cursor()
            # last_modified_utc wird durch Trigger aktualisiert
            cursor.execute(
                "UPDATE snippets SET quality_label = ? WHERE path = ?",
                (quality_label, path_str),
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(
                    f"Qualitätslabel für '{path_str}' zu '{quality_label}' aktualisiert."
                )
                return True
            else:
                logger.info(
                    f"Kein Snippet mit Pfad '{path_str}' gefunden, um Qualitätslabel zu aktualisieren."
                )
                return False
        except sqlite3.Error as e:
            logger.error(
                f"Fehler beim Aktualisieren des Qualitätslabels für '{path_str}': {e}",
                exc_info=True,
            )
            return False

    def get_snippet_quality(self, path_str: str) -> Optional[str]:
        conn = self._connect()
        if not conn:
            logger.error(
                f"Kann Qualität für {path_str} nicht abrufen, keine DB-Verbindung."
            )
            return None
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT quality_label FROM snippets WHERE path = ?", (path_str,)
            )
            row = cursor.fetchone()
            if row:
                return str(row[0])
            else:
                logger.debug(
                    f"Kein Qualitätslabel für Pfad '{path_str}' in DB gefunden."
                )
                return None
        except sqlite3.Error as e:
            logger.error(
                f"Fehler beim Abrufen des Qualitätslabels für '{path_str}': {e}",
                exc_info=True,
            )
            return None

    def get_feedback_data_for_training(
        self,
        feedback_types: Optional[List[str]] = None,
        action_types: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        conn = self._connect()
        if not conn:
            logger.error("Kann Feedback-Daten nicht abrufen, keine DB-Verbindung.")
            return []
        try:
            cursor = conn.cursor()
            query = "SELECT source_code_context, ai_generated_output FROM ki_feedback"
            conditions = []
            params: List[str] = []

            if (
                feedback_types
                and isinstance(feedback_types, list)
                and len(feedback_types) > 0
            ):
                placeholders = ",".join("?" * len(feedback_types))
                conditions.append(f"user_feedback_type IN ({placeholders})")
                params.extend(feedback_types)

            if (
                action_types
                and isinstance(action_types, list)
                and len(action_types) > 0
            ):
                placeholders = ",".join("?" * len(action_types))
                conditions.append(f"ai_action_type IN ({placeholders})")
                params.extend(action_types)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY timestamp"
            cursor.execute(query, params)
            rows = cursor.fetchall()
            training_data = []
            for row in rows:
                if row[0] and row[1]:
                    training_data.append(
                        {"prompt": str(row[0]), "completion": str(row[1])}
                    )
            logger.info(
                f"{len(training_data)} Feedback-Datensätze für Training abgerufen "
                f"(Filter: feedback_types={feedback_types}, action_types={action_types})."
            )
            return training_data
        except sqlite3.Error as e:
            logger.error(
                f"Fehler beim Abrufen von Feedback-Daten für Training: {e}",
                exc_info=True,
            )
            return []

    def _import_single_snippet(
        self, cursor: sqlite3.Cursor, snippet_data: Dict[str, Any]
    ) -> str:
        """
        Importiert/Aktualisiert ein einzelnes Snippet in die Datenbank.
        Gibt einen Statusstring zurück ("added", "updated", "skipped", "error").
        """
        s_uuid = snippet_data.get("uuid")
        s_path = snippet_data.get("path")
        s_content = snippet_data.get("content")
        s_quality = snippet_data.get("quality_label")
        s_lmu_utc_from_file = snippet_data.get("last_modified_utc")
        # timestamp_created_db aus dem Export wird hier nicht direkt für die Logik verwendet,
        # aber könnte für Audit-Zwecke in der Zukunft interessant sein.

        if not all([s_uuid, s_path, s_content, s_quality, s_lmu_utc_from_file]):
            logger.warning(
                f"Unvollständige Snippet-Daten für Import übersprungen: uuid={s_uuid}, path={s_path}"
            )
            return "skipped_incomplete"

        try:
            cursor.execute(
                "SELECT path, last_modified_utc FROM snippets WHERE uuid = ?", (s_uuid,)
            )
            existing_snippet = cursor.fetchone()

            if existing_snippet:
                # UUID existiert, prüfe auf Update-Notwendigkeit
                db_path, db_lmu_utc = existing_snippet
                if s_lmu_utc_from_file > db_lmu_utc:
                    # Prüfe auf Pfadkollision, falls sich der Pfad geändert hat
                    if s_path != db_path:
                        cursor.execute(
                            "SELECT 1 FROM snippets WHERE path = ? AND uuid != ?",
                            (s_path, s_uuid),
                        )
                        if cursor.fetchone():
                            logger.error(
                                f"Snippet-Update für UUID {s_uuid}: Neuer Pfad '{s_path}' kollidiert mit einem bestehenden Eintrag. Update übersprungen."
                            )
                            return "error_path_conflict_on_update"

                    cursor.execute(
                        """UPDATE snippets SET path = ?, content = ?, quality_label = ?, last_modified_utc = ?
                           WHERE uuid = ?""",
                        (s_path, s_content, s_quality, s_lmu_utc_from_file, s_uuid),
                    )
                    logger.debug(
                        f"Snippet UUID {s_uuid} (Pfad: {s_path}) aktualisiert."
                    )
                    return "updated"
                else:
                    logger.debug(
                        f"Snippet UUID {s_uuid} (Pfad: {s_path}) ist aktuell, kein Update nötig."
                    )
                    return "skipped_up_to_date"
            else:
                # UUID existiert nicht, versuche einzufügen
                # Prüfe zuerst, ob der Pfad bereits von einer anderen UUID verwendet wird
                cursor.execute("SELECT uuid FROM snippets WHERE path = ?", (s_path,))
                path_owner_uuid = cursor.fetchone()
                if path_owner_uuid:
                    logger.error(
                        f"Snippet mit neuem UUID {s_uuid}: Pfad '{s_path}' wird bereits von UUID {path_owner_uuid[0]} verwendet. Einfügen übersprungen."
                    )
                    return "error_path_conflict_on_insert"

                # Der Trigger `insert_snippets_defaults` würde `last_modified_utc` und `uuid` setzen,
                # aber wir wollen die Werte aus der Datei verwenden.
                # Der `timestamp` (Erstellungszeitpunkt in der *Client*-DB) wird ebenfalls aus der Datei übernommen,
                # wenn vorhanden, sonst CURRENT_TIMESTAMP der Tower-DB.
                s_timestamp_created = snippet_data.get(
                    "timestamp_created_db",
                    datetime.datetime.now(datetime.timezone.utc).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                )

                cursor.execute(
                    """INSERT INTO snippets (uuid, path, content, quality_label, timestamp, last_modified_utc)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        s_uuid,
                        s_path,
                        s_content,
                        s_quality,
                        s_timestamp_created,
                        s_lmu_utc_from_file,
                    ),
                )
                logger.debug(
                    f"Neues Snippet UUID {s_uuid} (Pfad: {s_path}) hinzugefügt."
                )
                return "added"

        except (
            sqlite3.IntegrityError
        ) as ie:  # Sollte durch die expliziten Pfadprüfungen selten sein
            logger.error(
                f"Datenbank Integritätsfehler beim Import von Snippet UUID {s_uuid}, Pfad {s_path}: {ie}",
                exc_info=True,
            )
            return "error_integrity"
        except Exception as e:
            logger.error(
                f"Allgemeiner Fehler beim Import von Snippet UUID {s_uuid}, Pfad {s_path}: {e}",
                exc_info=True,
            )
            return "error_generic"

    def _import_single_feedback(
        self, cursor: sqlite3.Cursor, feedback_data: Dict[str, Any]
    ) -> str:
        """
        Importiert einen einzelnen Feedback-Eintrag in die Datenbank.
        Feedback-Einträge gelten als unveränderlich; existierende UUIDs werden übersprungen.
        Gibt einen Statusstring zurück ("added", "skipped_exists", "error").
        """
        fb_uuid = feedback_data.get("uuid")
        # Andere Felder für Validierung (rudimentär)
        if (
            not fb_uuid
            or not feedback_data.get("source_code_context")
            or not feedback_data.get("ai_action_type")
        ):
            logger.warning(
                f"Unvollständige Feedback-Daten für Import übersprungen: uuid={fb_uuid}"
            )
            return "skipped_incomplete"

        try:
            cursor.execute("SELECT 1 FROM ki_feedback WHERE uuid = ?", (fb_uuid,))
            if cursor.fetchone():
                logger.debug(
                    f"Feedback UUID {fb_uuid} existiert bereits, wird übersprungen."
                )
                return "skipped_exists"
            else:
                # Werte aus der Datei übernehmen
                fb_timestamp_created = feedback_data.get(
                    "timestamp_created_db",
                    datetime.datetime.now(datetime.timezone.utc).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                )
                fb_lmu_utc = feedback_data.get(
                    "last_modified_utc",
                    datetime.datetime.now(datetime.timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%S.%fZ"
                    )[:-4]
                    + "Z",
                )

                cursor.execute(
                    """INSERT INTO ki_feedback
                       (uuid, timestamp, source_code_context, ai_action_type, ai_generated_output,
                        user_feedback_type, original_file_path, notes, last_modified_utc)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        fb_uuid,
                        fb_timestamp_created,
                        feedback_data.get("source_code_context"),
                        feedback_data.get("ai_action_type"),
                        feedback_data.get("ai_generated_output"),
                        feedback_data.get("user_feedback_type"),
                        feedback_data.get("original_file_path"),
                        feedback_data.get("notes"),
                        fb_lmu_utc,
                    ),
                )
                logger.debug(f"Neues Feedback UUID {fb_uuid} hinzugefügt.")
                return "added"
        except Exception as e:
            logger.error(
                f"Fehler beim Import von Feedback UUID {fb_uuid}: {e}", exc_info=True
            )
            return "error_generic"

    def import_data_from_jsonl_file(self, file_path: Path) -> Dict[str, int]:
        """
        Importiert Daten (Snippets und Feedback) aus einer JSON-Lines-Datei
        in die Datenbank des Towers.

        Args:
            file_path (Path): Der Pfad zur JSON-Lines-Datei.

        Returns:
            Ein Dictionary mit Zählern für verarbeitete, hinzugefügte, aktualisierte,
            übersprungene und fehlerhafte Einträge.
        """
        conn = self._connect()
        if not conn:
            logger.error("Kann Daten nicht aus JSONL importieren, keine DB-Verbindung.")
            return {
                "total_lines": 0,
                "processed": 0,
                "snippets_added": 0,
                "snippets_updated": 0,
                "snippets_skipped": 0,
                "snippets_errors": 0,
                "feedback_added": 0,
                "feedback_skipped": 0,
                "feedback_errors": 0,
            }

        stats = {
            "total_lines": 0,
            "processed": 0,
            "parse_errors": 0,
            "snippets_added": 0,
            "snippets_updated": 0,
            "snippets_skipped": 0,
            "snippets_errors": 0,
            "feedback_added": 0,
            "feedback_skipped": 0,
            "feedback_errors": 0,
        }

        try:
            with file_path.open("r", encoding="utf-8") as f:
                cursor = conn.cursor()
                for line_number, line in enumerate(f, 1):
                    stats["total_lines"] = line_number
                    if not line.strip():
                        continue
                    try:
                        data_item = json.loads(line)
                        stats["processed"] += 1
                        item_type = data_item.get("type")

                        if item_type == "snippet":
                            status = self._import_single_snippet(cursor, data_item)
                            if status == "added":
                                stats["snippets_added"] += 1
                            elif status == "updated":
                                stats["snippets_updated"] += 1
                            elif status.startswith("skipped"):
                                stats["snippets_skipped"] += 1
                            elif status.startswith("error"):
                                stats["snippets_errors"] += 1
                        elif item_type == "feedback":
                            status = self._import_single_feedback(cursor, data_item)
                            if status == "added":
                                stats["feedback_added"] += 1
                            elif status.startswith("skipped"):
                                stats["feedback_skipped"] += 1
                            elif status.startswith("error"):
                                stats["feedback_errors"] += 1
                        else:
                            logger.warning(
                                f"Unbekannter Datentyp in Zeile {line_number}: '{item_type}'"
                            )
                            stats[
                                "parse_errors"
                            ] += 1  # Zählen als Parse-Fehler für diesen Zweck

                    except json.JSONDecodeError:
                        logger.error(
                            f"Fehler beim Parsen von JSON in Zeile {line_number} der Datei {file_path.name}",
                            exc_info=True,
                        )
                        stats["parse_errors"] += 1
                    except (
                        Exception
                    ) as e:  # Fängt Fehler aus _import_single_... falls nicht spezifisch behandelt
                        logger.error(
                            f"Allgemeiner Fehler bei der Verarbeitung von Zeile {line_number} aus {file_path.name}: {e}",
                            exc_info=True,
                        )
                        if item_type == "snippet":
                            stats["snippets_errors"] += 1
                        elif item_type == "feedback":
                            stats["feedback_errors"] += 1
                        else:
                            stats[
                                "parse_errors"
                            ] += 1  # Wenn Typ unklar, aber Fehler auftritt

                conn.commit()
                logger.info(
                    f"Import aus '{file_path.name}' abgeschlossen. Statistik: {stats}"
                )

        except FileNotFoundError:
            logger.error(f"Importdatei nicht gefunden: {file_path}")
            # stats bleiben auf 0
        except Exception as e:
            logger.error(
                f"Schwerwiegender Fehler beim Lesen der Importdatei {file_path}: {e}",
                exc_info=True,
            )
            # stats könnten teilweise gefüllt sein, aber der Prozess wurde abgebrochen

        return stats

    def export_new_data_as_jsonl(
        self, since_timestamp_utc_iso: Optional[str] = None
    ) -> Optional[str]:
        """
        Exportiert neue oder geänderte Snippets und Feedback-Einträge als JSON-Lines String.

        Args:
            since_timestamp_utc_iso (Optional[str]): Ein ISO 8601 UTC Zeitstempel.
                Nur Daten, die neuer oder gleich diesem Zeitstempel sind, werden exportiert.
                Wenn None, werden alle Daten exportiert.

        Returns:
            Ein String im JSON-Lines Format oder None bei einem Fehler.
            Gibt einen leeren String zurück, wenn keine neuen Daten gefunden wurden.
        """
        conn = self._connect()
        if not conn:
            logger.error("Kann Daten nicht exportieren, keine DB-Verbindung.")
            return None

        exported_lines: List[str] = []

        try:
            cursor = conn.cursor()

            # Snippets exportieren
            snippet_query = "SELECT uuid, path, content, quality_label, timestamp, last_modified_utc FROM snippets"
            snippet_params: List[str] = []
            if since_timestamp_utc_iso:
                snippet_query += " WHERE last_modified_utc >= ?"
                snippet_params.append(since_timestamp_utc_iso)

            cursor.execute(snippet_query, snippet_params)
            snippet_rows = cursor.fetchall()
            for row in snippet_rows:
                snippet_data = {
                    "type": "snippet",
                    "uuid": (
                        row[0] if row[0] else str(uuid.uuid4())
                    ),  # Fallback UUID, falls DB alt
                    "path": row[1],
                    "content": row[2],
                    "quality_label": row[3],
                    "timestamp_created_db": row[4],  # Originaler DB-Timestamp
                    "last_modified_utc": row[5],  # Wichtig für inkrementellen Sync
                }
                exported_lines.append(json.dumps(snippet_data))
            logger.info(
                f"{len(snippet_rows)} Snippets für Export ausgewählt (since: {since_timestamp_utc_iso})."
            )

            # Feedback exportieren
            feedback_query = (
                "SELECT uuid, timestamp, source_code_context, ai_action_type, "
                "ai_generated_output, user_feedback_type, original_file_path, notes, last_modified_utc "
                "FROM ki_feedback"
            )
            feedback_params: List[str] = []
            if since_timestamp_utc_iso:
                feedback_query += " WHERE last_modified_utc >= ?"
                feedback_params.append(since_timestamp_utc_iso)

            cursor.execute(feedback_query, feedback_params)
            feedback_rows = cursor.fetchall()
            for row in feedback_rows:
                feedback_data = {
                    "type": "feedback",
                    "uuid": row[0] if row[0] else str(uuid.uuid4()),
                    "timestamp_created_db": row[1],
                    "source_code_context": row[2],
                    "ai_action_type": row[3],
                    "ai_generated_output": row[4],
                    "user_feedback_type": row[5],
                    "original_file_path": row[6],
                    "notes": row[7],
                    "last_modified_utc": row[8],
                }
                exported_lines.append(json.dumps(feedback_data))
            logger.info(
                f"{len(feedback_rows)} Feedback-Einträge für Export ausgewählt (since: {since_timestamp_utc_iso})."
            )

            if not exported_lines:
                logger.info("Keine neuen Daten für den Export gefunden.")
                return ""

            return "\n".join(exported_lines)

        except sqlite3.Error as e:
            logger.error(
                f"Fehler beim Exportieren von Daten aus der DB: {e}", exc_info=True
            )
            return None
        except (
            json.JSONDecodeError
        ) as e:  # Sollte beim Encoden nicht passieren, aber sicher ist sicher
            logger.error(
                f"Fehler beim Erstellen des JSON für den Export: {e}", exc_info=True
            )
            return None
