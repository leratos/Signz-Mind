# Klassen/linter_service.py
# -*- coding: utf-8 -*-

"""
Ein Dienst zur Durchführung von Code-Linting mit Flake8.
Verarbeitet sowohl stdout als auch stderr für Linting-Ausgaben.
"""

import logging
import os
import re
import subprocess
import tempfile
from typing import List, Optional, TypedDict

# Importiere Konfiguration relativ zum aktuellen Paket
from ..core import config as global_config

# Initialisiere den Logger für dieses Modul
logger = logging.getLogger(__name__)


class LintError(TypedDict):
    """Struktur für einen einzelnen Linter-Fehler."""

    line: int
    column: int
    code: str
    message: str
    physical_line: Optional[str]


class LinterService:
    """
    Führt Code-Linting mit Flake8 durch und parst die Ergebnisse.
    """

    DEFAULT_FLAKE8_PATH = "flake8"  # Standardpfad, falls nicht in Konfig

    def __init__(self, config_dict: Optional[dict] = None):
        """
        Initialisiert den LinterService.

        Args:
            config_dict (Optional[dict]): Ein Dictionary mit Laufzeitkonfigurationen.
                                          Erwartet optional den Schlüssel 'flake8_path'.
        """
        self.config_dict = config_dict if config_dict is not None else {}
        self.flake8_path = self.config_dict.get(
            "flake8_path",
            getattr(global_config, "FLAKE8_PATH", self.DEFAULT_FLAKE8_PATH),
        )

        self.ignore_codes: List[str] = self.config_dict.get("linter_ignore_codes", [])

        logger.info(f"LinterService initialisiert. Flake8 Pfad: '{self.flake8_path}'")
        self._check_flake8_availability()

    def _check_flake8_availability(self):
        """Überprüft, ob Flake8 aufrufbar ist und loggt die Version."""
        try:
            process = subprocess.Popen(
                [self.flake8_path, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            stdout, stderr = process.communicate(timeout=5)
            if process.returncode == 0:
                logger.info(f"Flake8 gefunden und funktionsfähig: {stdout.strip()}")
            else:
                logger.warning(
                    f"Flake8 --version gab Rückgabecode {process.returncode} zurück. "
                    f"Stderr: {stderr.strip()}"
                )
        except FileNotFoundError:
            logger.error(
                f"Flake8 Executable nicht unter '{self.flake8_path}' gefunden. "
                "Linting wird fehlschlagen. Bitte installieren oder Pfad korrigieren."
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                f"Timeout bei der Überprüfung der Flake8-Version ('{self.flake8_path} --version')."
            )
        except Exception as e:
            logger.error(
                f"Fehler bei der Überprüfung der Flake8-Version: {e}", exc_info=True
            )

    def _parse_flake8_output_line(self, line: str) -> Optional[LintError]:
        """
        Parst eine einzelne Zeile der Flake8-Ausgabe (von stdout oder stderr).
        Versucht zuerst das Standard Flake8-Format, dann ein einfacheres für direkte Python-Fehler.

        Args:
            line (str): Eine einzelne Zeile der Flake8-Ausgabe.

        Returns:
            Optional[LintError]: Ein Dictionary mit Fehlerinformationen oder None,
                                 wenn die Zeile nicht geparst werden konnte.
        """
        stripped_line = line.strip()
        if not stripped_line:  # Ignoriere leere Zeilen
            return None

        # Primärer Versuch: Standard Flake8 Format (path:line:col: CODE message)
        # Diese Regex ist robuster für Pfade mit Doppelpunkten (Windows)
        # und flexibler für Fehlercodes (z.B. E999, WPS301).
        primary_match = re.match(
            r"^(.*?):(\d+):(\d+):\s*([A-Za-z0-9]+)\s+(.*)$", stripped_line
        )

        if primary_match:
            try:
                # filename = primary_match.group(1) # Nicht direkt für die Fehlermeldung benötigt
                line_num_str = primary_match.group(2)
                col_num_str = primary_match.group(3)
                error_code = primary_match.group(4)
                error_message = primary_match.group(5).strip()
                return LintError(
                    line=int(line_num_str),
                    column=int(col_num_str),
                    code=error_code,
                    message=error_message,
                    physical_line=None,  # Wird später hinzugefügt
                )
            except ValueError:
                logger.debug(
                    f"Konnte Zeile/Spalte nicht in Integer umwandeln: '{stripped_line}'"
                )
                return None
            except Exception as e:  # Fange unerwartete Fehler beim Parsen ab
                logger.warning(
                    f"Unerwarteter Fehler beim Parsen (primary_match) der Zeile '{stripped_line}': {e}",
                    exc_info=False,  # Traceback hier oft nicht nötig
                )
                return None

        # Fallback-Versuch: Einfacheres Splitting für Fälle, wo Flake8 ggf. direkte Python-Fehler ausgibt
        # oder das Format leicht abweicht und der Dateipfad am Anfang steht.
        parts = stripped_line.split(":", 3)
        if len(parts) == 4:
            try:
                # filename_fb = parts[0]
                line_num_str_fb = parts[1]
                col_num_str_fb = parts[2]
                details_fb = parts[3].strip()

                # Versuche, aus dem "details"-Teil einen Fehlercode und eine Nachricht zu extrahieren
                code_message_match = re.match(r"([A-Za-z0-9]+)\s+(.*)", details_fb)
                error_code_fb: str
                error_message_fb: str

                if code_message_match:
                    error_code_fb = code_message_match.group(1)
                    error_message_fb = code_message_match.group(2).strip()
                # Wenn kein Flake8-Code, prüfe auf direkte Python-Fehlertypen
                elif details_fb.startswith(
                    ("SyntaxError:", "IndentationError:", "TabError:")
                ):
                    error_code_fb = (
                        "PYTHON_ERROR"  # Generischer Code für direkte Python-Fehler
                    )
                    error_message_fb = details_fb
                else:
                    logger.debug(
                        f"Fallback-Parse: Detail-Teil '{details_fb}' passt nicht zu bekanntem "
                        f"Flake8-Code oder Python-Fehlertyp für Zeile '{stripped_line}'."
                    )
                    return None

                return LintError(
                    line=int(line_num_str_fb),
                    column=int(col_num_str_fb),
                    code=error_code_fb,
                    message=error_message_fb,
                    physical_line=None,
                )
            except ValueError:
                logger.debug(
                    f"Fallback-Parse: Konnte Zeile/Spalte nicht in Integer umwandeln: '{stripped_line}'"
                )
                return None
            except Exception as e:
                logger.warning(
                    f"Unerwarteter Fehler beim Parsen (fallback) der Zeile '{stripped_line}': {e}",
                    exc_info=False,
                )
                return None

        logger.debug(
            f"Zeile '{stripped_line}' passt zu keinem erwarteten Flake8-Format."
        )
        return None

    def lint_code(self, code_string: str) -> List[LintError]:
        """
        Führt Flake8 für den gegebenen Code-String aus und gibt eine Liste von Fehlern zurück.
        Verarbeitet sowohl stdout als auch stderr für Linting-Ausgaben.

        Args:
            code_string (str): Der zu lintende Python-Code.

        Returns:
            List[LintError]: Eine Liste von gefundenen Linter-Fehlern.
        """
        logger.info(f"Starte Linting für Code (Länge: {len(code_string)} Zeichen).")
        if not code_string.strip():
            logger.debug(
                "Code-String ist leer oder nur Whitespace. Gebe keine Fehler zurück."
            )
            return []

        temp_file_path: Optional[str] = None
        try:
            # Stelle sicher, dass der Code mit einem Newline endet
            code_to_lint = (
                code_string if code_string.endswith("\n") else code_string + "\n"
            )

            # Erstelle eine temporäre Datei für Flake8
            # delete=False, damit der Pfad für subprocess gültig bleibt; manuelle Löschung im finally-Block
            with tempfile.NamedTemporaryFile(
                mode="w+", delete=False, suffix=".py", encoding="utf-8"
            ) as tf:
                tf.write(code_to_lint)
                temp_file_path = tf.name
            logger.debug(f"Code in temporäre Datei geschrieben: {temp_file_path}")

            command = [self.flake8_path, temp_file_path]
            logger.debug(f"Führe Flake8-Kommando aus: {' '.join(command)}")

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            stdout_data, stderr_data = process.communicate(
                timeout=10
            )  # Timeout für den Prozess

            logger.debug(f"Flake8 Rückgabecode: {process.returncode}")
            if stdout_data.strip():
                logger.debug(f"Flake8 stdout:\n---\n{stdout_data.strip()}\n---")
            if stderr_data.strip():
                logger.debug(f"Flake8 stderr:\n---\n{stderr_data.strip()}\n---")

            linting_errors: List[LintError] = []
            code_lines = code_to_lint.splitlines()

            # Funktion zum Verarbeiten einer Liste von Ausgabezeilen
            def process_output_stream(stream_data: str, stream_name: str):
                if stream_data:
                    for line in stream_data.strip().splitlines():
                        logger.debug(f"Parse Flake8 {stream_name} Zeile: '{line}'")
                        parsed_error = self._parse_flake8_output_line(line)
                        if parsed_error:
                            # Füge die physische Zeile hinzu, falls möglich
                            if 0 < parsed_error["line"] <= len(code_lines):
                                parsed_error["physical_line"] = code_lines[
                                    parsed_error["line"] - 1
                                ]

                            # Vermeide Duplikate (kann passieren, wenn derselbe Fehler auf stdout & stderr)
                            is_duplicate = any(
                                err["line"] == parsed_error["line"]
                                and err["column"] == parsed_error["column"]
                                and err["message"]
                                == parsed_error["message"]  # Vergleiche auch Nachricht
                                and err["code"] == parsed_error["code"]
                                for err in linting_errors
                            )
                            if not is_duplicate:
                                linting_errors.append(parsed_error)
                            else:
                                logger.debug(
                                    f"Duplizierter Fehler von {stream_name} übersprungen: '{line}'"
                                )
                        else:
                            logger.warning(
                                f"Konnte Flake8 {stream_name} Zeile nicht parsen: '{line}'"
                            )

            # Verarbeite stdout und stderr
            process_output_stream(stdout_data, "stdout")
            # Verarbeite stderr, wenn Flake8 Fehler meldet (returncode != 0)
            # oder auch wenn returncode 0 ist, aber stderr Inhalt hat (für manche Plugins)
            if stderr_data and (process.returncode != 0 or stdout_data.strip() == ""):
                logger.info(
                    f"Verarbeite Flake8 stderr (Rückgabecode {process.returncode})."
                )
                process_output_stream(stderr_data, "stderr")

            if self.ignore_codes:
                filtered_errors = [
                    error for error in linting_errors
                    if error.get('code') not in self.ignore_codes
                ]
                logger.info(f"{len(linting_errors) - len(filtered_errors)} Fehler wurden ignoriert. Codes: {self.ignore_codes}")
            else:
                filtered_errors = linting_errors
            # Wenn der Returncode > 1 ist (oder ein anderer Fehler) und nichts geparst wurde,
            # aber stderr/stdout Inhalt hat, logge dies als generischen Fehler.
            if (
                process.returncode not in [0, 1]
                and not filtered_errors
                and (stderr_data.strip() or stdout_data.strip())
            ):
                combined_output = (
                    stdout_data.strip() + " " + stderr_data.strip()
                ).strip()
                error_message = f"Flake8 Ausführung fehlgeschlagen mit Code {process.returncode}: {combined_output}"
                logger.error(error_message)
                # Nur hinzufügen, wenn nicht schon ein spezifischerer Fehler gemeldet wurde
                if not any(
                    err["code"] == "FLAKE8_EXEC_ERROR" for err in filtered_errors
                ):
                    filtered_errors.append(
                        {
                            "line": 0,
                            "column": 0,
                            "code": "FLAKE8_EXEC_ERROR",
                            "message": error_message,
                            "physical_line": "",
                        }
                    )

            logger.info(
                f"Linting abgeschlossen. {len(filtered_errors)} Fehler/Warnungen gefunden."
            )
            return filtered_errors

        except FileNotFoundError:
            msg = (
                f"Flake8 nicht gefunden unter '{self.flake8_path}'. Bitte installieren "
                "und/oder Pfad in den Einstellungen konfigurieren."
            )
            logger.error(msg)
            return [
                {
                    "line": 0,
                    "column": 0,
                    "code": "LINTER_NOT_FOUND",
                    "message": msg,
                    "physical_line": "",
                }
            ]
        except subprocess.TimeoutExpired:
            msg = "Flake8-Ausführung überschritt das Zeitlimit."
            logger.error(msg)
            return [
                {
                    "line": 0,
                    "column": 0,
                    "code": "LINTER_TIMEOUT",
                    "message": msg,
                    "physical_line": "",
                }
            ]
        except Exception as e:
            msg = f"Unerwarteter Fehler beim Ausführen von Flake8: {e}"
            logger.error(msg, exc_info=True)
            # Gebe leere Liste zurück oder einen generischen Fehler
            return [
                {
                    "line": 0,
                    "column": 0,
                    "code": "LINTER_UNEXPECTED_ERROR",
                    "message": msg,
                    "physical_line": "",
                }
            ]
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.debug(f"Temporäre Datei {temp_file_path} gelöscht.")
                except Exception as e_del:
                    logger.warning(
                        f"Konnte temporäre Datei nicht löschen {temp_file_path}: {e_del}"
                    )
