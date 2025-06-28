# Klassen/file_manager.py
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING, List, Dict, Any

from PySide6.QtWidgets import QFileDialog, QMessageBox, QInputDialog, QApplication

if TYPE_CHECKING:
    from ..tab_manager import TabManager
    from ..data.data_manager import DataManager
    from ..ui.ui_main_window import MainWindow

logger = logging.getLogger(__name__)


class FileManager:
    """
    Manages file operations such as opening, saving, save as,
    and interaction with the file system and DataManager.
    """

    def __init__(
        self,
        main_window: "MainWindow",
        tab_manager: "TabManager",
        data_manager: "DataManager",
    ):
        self.main_window = main_window
        self.tab_manager = tab_manager
        self.data_manager = data_manager
        logger.info("FileManager initialisiert.")

    def get_default_start_dir(self, for_save_as: bool = False) -> str:
        """
        Determines the default starting directory for file dialogs.
        """
        current_tab_data = self.tab_manager.get_current_tab_data()

        if for_save_as and current_tab_data and current_tab_data.get("path"):
            return str(current_tab_data["path"].parent)

        if (
            self.main_window.project_root_path
            and self.main_window.project_root_path.is_dir()
        ):
            return str(self.main_window.project_root_path)

        return str(Path.home())

    def handle_new_file(self):
        """Requests the TabManager to create a new, empty tab."""
        if not self.tab_manager:
            logger.warning(
                "FileManager: TabManager nicht verfügbar für handle_new_file."
            )
            return
        self.tab_manager.add_new_tab()
        logger.info("FileManager: Neues Tab angefordert.")

    def handle_open_files(self) -> List[Path]:
        """
        Opens the "Open File" dialog and initiates opening of selected files.
        Returns a list of paths successfully selected for opening.
        """
        if not self.tab_manager:
            logger.warning(
                "FileManager: TabManager nicht verfügbar für handle_open_files."
            )
            return []

        start_dir = self.get_default_start_dir()

        file_paths_tuple, _ = QFileDialog.getOpenFileNames(
            self.main_window,
            "Python-Datei(en) öffnen",
            start_dir,
            "Python Dateien (*.py);;Alle Dateien (*.*)",
        )

        opened_paths: List[Path] = []
        if (
            file_paths_tuple
        ):  # QFileDialog returns a tuple (list_of_paths, filter_string)
            for file_path_str in file_paths_tuple:
                if file_path_str:  # Ensure the path string is not empty
                    file_path = Path(file_path_str)
                    if self._open_single_file_in_tab(file_path):
                        opened_paths.append(file_path)

        if (
            not opened_paths and file_paths_tuple
        ):  # Check if any files were actually selected
            self.main_window.statusBar().showMessage(
                "Keine neuen Dateien geöffnet (ggf. bereits offen oder Fehler).", 3000
            )
        elif opened_paths:
            self.main_window.statusBar().showMessage(
                f"{len(opened_paths)} Datei(en) geöffnet/aktiviert.", 3000
            )
        return opened_paths

    def _open_single_file_in_tab(self, file_path: Path) -> bool:
        """
        Internal method to open or activate a single file in a tab.
        """
        if not self.tab_manager:
            return False

        if not file_path.is_file():
            logger.warning(f"FileManager: Versuch, Nicht-Datei zu öffnen: {file_path}")
            QMessageBox.warning(
                self.main_window,
                "Fehler",
                f"Der Pfad '{file_path}' ist keine gültige Datei.",
            )
            return False

        existing_tab_index = self.tab_manager.find_tab_by_path(file_path)
        if existing_tab_index != -1:
            self.tab_manager.tab_widget.setCurrentIndex(existing_tab_index)
            logger.info(
                f"FileManager: Datei '{file_path.name}' ist bereits geöffnet, wechsle zu Tab."
            )
            return True

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            self.tab_manager.add_new_tab(file_path=file_path, content=content)
            logger.info(f"FileManager: Datei '{file_path}' in neuem Tab geöffnet.")
            return True
        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "Fehler beim Öffnen",
                f"Datei '{file_path}' konnte nicht gelesen werden:\n{e}",
            )
            logger.error(
                f"FileManager: Fehler beim Öffnen der Datei {file_path}: {e}",
                exc_info=True,
            )
            return False

    def handle_save_file(self) -> bool:
        """Saves the file in the current tab. May call 'Save As' if needed."""
        if not self.tab_manager:
            return False

        current_index = self.tab_manager.get_current_editor_index()
        if current_index == -1:
            self.main_window.statusBar().showMessage(
                "Kein aktives Tab zum Speichern.", 3000
            )
            return False

        tab_data = self.tab_manager.get_tab_data(current_index)
        if not tab_data or tab_data.get("path") is None:
            return self.handle_save_file_as()

        return self._save_tab_to_file(current_index, tab_data)

    def handle_save_file_as(self) -> bool:
        """Opens the "Save As" dialog for the current tab."""
        if not self.tab_manager:
            return False

        current_index = self.tab_manager.get_current_editor_index()
        if current_index == -1:
            self.main_window.statusBar().showMessage(
                "Kein aktives Tab für 'Speichern unter'.", 3000
            )
            return False

        tab_data = self.tab_manager.get_tab_data(current_index)
        if not tab_data:
            return False

        start_dir = self.get_default_start_dir(for_save_as=True)
        current_path = tab_data.get("path")
        default_filename = current_path.name if current_path else "unbenannt.py"
        default_path_str = str(Path(start_dir) / default_filename)

        file_path_str, _ = QFileDialog.getSaveFileName(
            self.main_window,
            "Datei speichern unter",
            default_path_str,
            "Python Dateien (*.py);;Alle Dateien (*.*)",
        )

        if file_path_str:
            new_file_path = Path(file_path_str)
            self.tab_manager.set_path(current_index, new_file_path)
            return self._save_tab_to_file(
                current_index, self.tab_manager.get_tab_data(current_index)
            )
        return False

    def _save_tab_to_file(
        self, tab_index: int, tab_data: Optional[Dict[str, Any]]
    ) -> bool:
        """Internal method to save the content of a tab to a file."""
        if not tab_data:
            return False

        editor = tab_data.get("editor")
        file_path = tab_data.get("path")

        if not editor or not file_path:
            logger.error(
                "FileManager: _save_tab_to_file ohne Editor oder Pfad aufgerufen."
            )
            return False

        content = editor.toPlainText()
        try:
            file_path.write_text(content, encoding="utf-8")
            editor.document().setModified(False)

            current_quality = tab_data.get(
                "quality_label", self.data_manager.DEFAULT_QUALITY_LABEL
            )
            self.main_window.statusBar().showMessage(
                f"Datei '{file_path.name}' gespeichert.", 3000
            )
            logger.info(f"FileManager: Datei '{file_path.name}' gespeichert.")

            if self.data_manager:
                reply = QMessageBox.question(
                    self.main_window,
                    "Für KI-Training verwenden?",
                    (
                        f"Möchten Sie den aktuellen Inhalt von '{file_path.name}' "
                        f"mit Qualität '{current_quality}' in die Datenbank für das KI-Training übernehmen/aktualisieren?"
                    ),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    added_count, _ = self.data_manager.load_files_into_db(
                        [str(file_path)], default_quality=current_quality
                    )
                    if added_count > 0:
                        self.main_window.statusBar().showMessage(
                            f"'{file_path.name}' (Qualität: {current_quality}) in DB übernommen.",
                            3000,
                        )
                    else:
                        self.main_window.statusBar().showMessage(
                            f"'{file_path.name}' nicht in DB übernommen (ggf. keine Änderung).",
                            3000,
                        )
                else:
                    self.data_manager.update_snippet_quality(
                        str(file_path), current_quality
                    )
                    self.main_window.statusBar().showMessage(
                        f"'{file_path.name}' gespeichert, DB-Qualität für Pfad aktualisiert.",
                        3000,
                    )

            if content.strip() and hasattr(
                self.main_window, "on_lint_request_received"
            ):
                self.main_window.on_lint_request_received(editor)
            return True

        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "Fehler beim Speichern",
                f"Datei konnte nicht geschrieben werden:\n{e}",
            )
            logger.error(
                f"FileManager: Fehler beim Speichern der Datei {file_path}: {e}",
                exc_info=True,
            )
            return False

    def maybe_save_current_tab(self) -> bool:
        """Checks if the current tab has unsaved changes and prompts if necessary."""
        if not self.tab_manager:
            return True

        current_tab_data = self.tab_manager.get_current_tab_data()
        if not current_tab_data:
            return True

        editor = current_tab_data.get("editor")
        if not editor or not editor.document().isModified():
            return True

        tab_name = (
            current_tab_data.get("path").name
            if current_tab_data.get("path")
            else self.tab_manager.tab_widget.tabText(
                self.tab_manager.get_current_editor_index()
            )
        )

        ret = QMessageBox.warning(
            self.main_window,
            "Ungespeicherte Änderungen",
            f"Das Dokument '{tab_name}' wurde geändert.\nMöchtest du deine Änderungen speichern?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if ret == QMessageBox.StandardButton.Save:
            return self.handle_save_file()
        elif ret == QMessageBox.StandardButton.Cancel:
            logger.info(
                f"FileManager: Speichervorgang für '{tab_name}' vom Benutzer abgebrochen."
            )
            return False
        logger.info(f"FileManager: Änderungen für '{tab_name}' verworfen.")
        return True

    def close_all_tabs_with_save_check(self) -> bool:
        """
        Attempts to close all tabs, prompting for unsaved changes in each.
        Returns True if all tabs could be closed (or no action was needed),
        False if the user cancelled the operation.
        """
        if not self.tab_manager:
            return True

        for i in range(self.tab_manager.count() - 1, -1, -1):
            self.tab_manager.tab_widget.setCurrentIndex(i)
            if not self.maybe_save_current_tab():
                return False
        return True

    def handle_load_py_files_to_db(self):
        """Loads selected .py files into the database."""
        if not self.data_manager:
            self.main_window.statusBar().showMessage(
                "DataManager nicht initialisiert.", 3000
            )
            return

        start_dir = self.get_default_start_dir()
        file_paths_tuple, _ = QFileDialog.getOpenFileNames(
            self.main_window,
            "Wähle Python-Dateien für DB-Import aus",
            start_dir,
            "Python Dateien (*.py);;Alle Dateien (*.*)",
        )
        file_paths: List[str] = list(file_paths_tuple)

        if not file_paths:
            self.main_window.statusBar().showMessage(
                "Keine Dateien für DB-Import ausgewählt.", 3000
            )
            return

        selected_label, ok = QInputDialog.getItem(
            self.main_window,
            "Qualitätslabel auswählen",
            "Wähle ein Qualitätslabel für die zu importierenden Dateien:",
            self.main_window.QUALITY_LABELS,
            0,
            False,
        )

        if not ok or not selected_label:
            self.main_window.statusBar().showMessage(
                "DB-Import abgebrochen: Kein Qualitätslabel ausgewählt.", 3000
            )
            return

        self.main_window.statusBar().showMessage(
            f"Lade {len(file_paths)} Datei(en) mit Qualität '{selected_label}' in DB...",
            2000,
        )
        QApplication.processEvents()

        added_count, skipped_count = self.data_manager.load_files_into_db(
            file_paths, default_quality=selected_label
        )
        status_msg = (
            f"DB-Import abgeschlossen. Hinzugefügt/Ersetzt: {added_count}, "
            f"Übersprungen: {skipped_count}. Qualität: '{selected_label}'"
        )
        self.main_window.statusBar().showMessage(status_msg, 5000)
        logger.info(status_msg)
