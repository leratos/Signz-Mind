# Klassen/search_controller.py
# -*- coding: utf-8 -*-

import logging
import re
import os
import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QObject, Slot, QEvent
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QSpacerItem,
    QSizePolicy,
    QInputDialog,
    QListWidgetItem,
    QApplication,
    QMessageBox,
)
from PySide6.QtGui import (
    QTextDocument,
    QKeyEvent,
)

if TYPE_CHECKING:
    from .ui.ui_main_window import MainWindow
    from .tab_manager import TabManager
    from .core.file_manager import FileManager

logger = logging.getLogger(__name__)


class SearchController(QObject):
    """
    Manages the find/replace bar and project-wide search functionality.
    """

    def __init__(
        self,
        main_window: "MainWindow",
        tab_manager: "TabManager",
        file_manager: "FileManager",
    ):
        super().__init__(main_window)
        self.main_window = main_window
        self.tab_manager = tab_manager
        self.file_manager = file_manager

        self._setup_find_replace_bar_ui()
        logger.info("SearchController initialisiert.")

    def _setup_find_replace_bar_ui(self):
        """Creates the find/replace bar widget and its components, assigning them to MainWindow."""
        mw = self.main_window

        mw.find_replace_widget = QWidget()
        find_replace_layout = QHBoxLayout(mw.find_replace_widget)
        find_replace_layout.setContentsMargins(0, 5, 0, 5)

        mw.find_input = QLineEdit()
        mw.find_input.setPlaceholderText("Suchen...")
        mw.find_input.textChanged.connect(self._update_find_buttons_state)
        mw.find_input.installEventFilter(self)
        find_replace_layout.addWidget(mw.find_input)

        mw.find_next_button = QPushButton("Nächstes")
        mw.find_next_button.clicked.connect(self.on_find_next)
        find_replace_layout.addWidget(mw.find_next_button)

        mw.find_prev_button = QPushButton("Vorheriges")
        mw.find_prev_button.clicked.connect(self.on_find_prev)
        find_replace_layout.addWidget(mw.find_prev_button)

        mw.replace_input = QLineEdit()
        mw.replace_input.setPlaceholderText("Ersetzen durch...")
        mw.replace_input.installEventFilter(self)
        find_replace_layout.addWidget(mw.replace_input)

        mw.replace_button = QPushButton("Ersetzen")
        mw.replace_button.clicked.connect(self.on_replace_current)
        find_replace_layout.addWidget(mw.replace_button)

        mw.replace_all_button = QPushButton("Alle ersetzen")
        mw.replace_all_button.clicked.connect(self.on_replace_all)
        find_replace_layout.addWidget(mw.replace_all_button)

        mw.case_sensitive_checkbox = QCheckBox("Groß/Klein")
        find_replace_layout.addWidget(mw.case_sensitive_checkbox)

        mw.whole_word_checkbox = QCheckBox("Ganzes Wort")
        find_replace_layout.addWidget(mw.whole_word_checkbox)

        find_replace_layout.addSpacerItem(
            QSpacerItem(10, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        close_find_bar_button = QPushButton("✕")
        close_find_bar_button.setToolTip("Suchleiste schließen")
        close_find_bar_button.setFixedSize(25, 25)
        close_find_bar_button.clicked.connect(
            self.hide_bar
        )  # hide_bar ist eine Methode dieses Controllers
        find_replace_layout.addWidget(close_find_bar_button)

        mw.find_replace_widget.setVisible(False)
        self._update_find_buttons_state()

    def eventFilter(self, watched_object: QObject, event: QEvent) -> bool:
        mw = self.main_window
        # Nur Events für die Such-Eingabefelder dieses Controllers verarbeiten
        if watched_object is mw.find_input and event.type() == QEvent.Type.KeyPress:
            key_event = QKeyEvent(event)  # type: ignore
            if key_event.key() in [Qt.Key.Key_Enter, Qt.Key.Key_Return]:
                self.on_find_next()
                return True
        elif (
            watched_object is mw.replace_input and event.type() == QEvent.Type.KeyPress
        ):
            key_event = QKeyEvent(event)  # type: ignore
            if key_event.key() in [Qt.Key.Key_Enter, Qt.Key.Key_Return]:
                self.on_replace_current()
                return True
        return super().eventFilter(watched_object, event)

    @Slot()
    def show_bar(self):
        """Makes the find/replace bar visible and focuses the find input."""
        mw = self.main_window
        if hasattr(mw, "find_replace_widget") and mw.find_replace_widget:
            mw.find_replace_widget.setVisible(True)
            if hasattr(mw, "find_input") and mw.find_input:
                mw.find_input.setFocus()
                current_editor = self.tab_manager.get_current_editor()
                if current_editor and current_editor.textCursor().hasSelection():
                    selected_text = current_editor.textCursor().selectedText()
                    if "\n" not in selected_text:
                        mw.find_input.setText(selected_text)
                        mw.find_input.selectAll()
            self._update_find_buttons_state()

    @Slot()
    def hide_bar(self):
        """Hides the find/replace bar and returns focus to the current editor."""
        mw = self.main_window
        if hasattr(mw, "find_replace_widget") and mw.find_replace_widget:
            mw.find_replace_widget.setVisible(False)
            current_editor = self.tab_manager.get_current_editor()
            if current_editor:
                current_editor.setFocus()

    def _get_find_flags(self) -> QTextDocument.FindFlags:
        """Returns the QTextDocument.FindFlags based on checkbox states in MainWindow."""
        mw = self.main_window
        flags = QTextDocument.FindFlags()
        if (
            hasattr(mw, "case_sensitive_checkbox")
            and mw.case_sensitive_checkbox
            and mw.case_sensitive_checkbox.isChecked()
        ):
            flags |= QTextDocument.FindFlag.FindCaseSensitively
        if (
            hasattr(mw, "whole_word_checkbox")
            and mw.whole_word_checkbox
            and mw.whole_word_checkbox.isChecked()
        ):
            flags |= QTextDocument.FindFlag.FindWholeWords
        return flags

    @Slot()
    def _update_find_buttons_state(self):
        """Updates the enabled state of find/replace buttons in MainWindow."""
        mw = self.main_window
        # Sicherstellen, dass alle UI-Elemente in MainWindow existieren
        if not all(
            hasattr(mw, attr_name) and getattr(mw, attr_name) is not None
            for attr_name in [
                "find_input",
                "find_next_button",
                "find_prev_button",
                "replace_all_button",
                "replace_button",
                "case_sensitive_checkbox",
                "whole_word_checkbox",
            ]
        ):
            logger.debug(
                "SearchController: Nicht alle UI-Elemente für Suchleiste in MainWindow vorhanden."
            )
            return

        has_text_to_find = bool(mw.find_input.text())
        mw.find_next_button.setEnabled(has_text_to_find)
        mw.find_prev_button.setEnabled(has_text_to_find)
        mw.replace_all_button.setEnabled(has_text_to_find)

        current_editor = self.tab_manager.get_current_editor()
        can_replace_current = False
        if current_editor and current_editor.textCursor().hasSelection():
            selected_text = current_editor.textCursor().selectedText()
            find_input_text = mw.find_input.text()
            if not find_input_text:
                can_replace_current = True
            else:
                if mw.case_sensitive_checkbox.isChecked():
                    can_replace_current = selected_text == find_input_text
                else:
                    can_replace_current = (
                        selected_text.lower() == find_input_text.lower()
                    )

        mw.replace_button.setEnabled(can_replace_current)

    @Slot()
    def on_find_next(self):
        mw = self.main_window
        current_editor = self.tab_manager.get_current_editor()
        if not current_editor or not hasattr(mw, "find_input"):
            return

        text_to_find = mw.find_input.text()
        if not text_to_find:
            mw.statusBar().showMessage("Kein Suchtext eingegeben.", 2000)
            return
        find_flags = self._get_find_flags()
        if not current_editor.find_text(
            text_to_find, find_flags, cursor=current_editor.textCursor()
        ):
            mw.statusBar().showMessage(
                f"'{text_to_find}' nicht gefunden (Suche am Ende/Anfang fortgesetzt).",
                2000,
            )
        else:
            mw.statusBar().clearMessage()
        self._update_find_buttons_state()

    @Slot()
    def on_find_prev(self):
        mw = self.main_window
        current_editor = self.tab_manager.get_current_editor()
        if not current_editor or not hasattr(mw, "find_input"):
            return

        text_to_find = mw.find_input.text()
        if not text_to_find:
            mw.statusBar().showMessage("Kein Suchtext eingegeben.", 2000)
            return
        find_flags = self._get_find_flags() | QTextDocument.FindFlag.FindBackward

        if not current_editor.find_text(
            text_to_find, find_flags, cursor=current_editor.textCursor()
        ):
            mw.statusBar().showMessage(
                f"'{text_to_find}' nicht gefunden (Suche am Anfang/Ende fortgesetzt).",
                2000,
            )
        else:
            mw.statusBar().clearMessage()
        self._update_find_buttons_state()

    @Slot()
    def on_replace_current(self):
        mw = self.main_window
        current_editor = self.tab_manager.get_current_editor()
        if (
            not current_editor
            or not hasattr(mw, "replace_input")
            or not hasattr(mw, "find_input")
            or not hasattr(mw, "case_sensitive_checkbox")
        ):
            return

        replace_text_str = mw.replace_input.text()
        text_to_find = mw.find_input.text()

        if current_editor.textCursor().hasSelection():
            if text_to_find:
                current_selection_text = current_editor.textCursor().selectedText()
                matches_find_text = False
                if mw.case_sensitive_checkbox.isChecked():
                    matches_find_text = current_selection_text == text_to_find
                else:
                    matches_find_text = (
                        current_selection_text.lower() == text_to_find.lower()
                    )

                if not matches_find_text:
                    self.on_find_next()
                    if current_editor.textCursor().hasSelection():
                        new_selection_text = current_editor.textCursor().selectedText()
                        if mw.case_sensitive_checkbox.isChecked():
                            matches_find_text = new_selection_text == text_to_find
                        else:
                            matches_find_text = (
                                new_selection_text.lower() == text_to_find.lower()
                            )

                        if matches_find_text:
                            if current_editor.replace_current_selection(
                                replace_text_str
                            ):
                                mw.statusBar().showMessage(
                                    "Ersetzt und weitergesucht.", 2000
                                )
                            self.on_find_next()
                    return

            if current_editor.replace_current_selection(replace_text_str):
                mw.statusBar().showMessage("Ersetzt.", 2000)
            if text_to_find:
                self.on_find_next()
        else:
            self.on_find_next()
        self._update_find_buttons_state()

    @Slot()
    def on_replace_all(self):
        mw = self.main_window
        current_editor = self.tab_manager.get_current_editor()
        if (
            not current_editor
            or not hasattr(mw, "replace_input")
            or not hasattr(mw, "find_input")
        ):
            return

        text_to_find = mw.find_input.text()
        replace_text_str = mw.replace_input.text()
        if not text_to_find:
            mw.statusBar().showMessage(
                "Kein Suchtext für 'Alle ersetzen' angegeben.", 2000
            )
            return
        find_flags = self._get_find_flags()
        count = current_editor.replace_all_occurrences(
            text_to_find, replace_text_str, find_flags
        )
        mw.statusBar().showMessage(f"{count} Vorkommen ersetzt.", 3000)
        self._update_find_buttons_state()

    @Slot()
    def trigger_project_search(self):
        """Initiates a project-wide search."""
        mw = self.main_window
        if not mw.project_root_path or not mw.project_root_path.is_dir():
            QMessageBox.information(
                mw,
                "Projektweite Suche",
                "Bitte zuerst einen Projektordner im Projekt-Explorer öffnen.",
            )
            return

        search_term, ok = QInputDialog.getText(mw, "Im Projekt suchen", "Suchbegriff:")
        if not (ok and search_term):
            mw.statusBar().showMessage("Projektweite Suche abgebrochen.", 2000)
            return

        mw.statusBar().showMessage(f"Suche im Projekt nach '{search_term}'...", 0)
        QApplication.processEvents()

        search_flags = self._get_find_flags()

        self._perform_project_search(search_term, search_flags)

        mw.update_button_states()

    def _perform_project_search(
        self, search_term: str, search_flags: QTextDocument.FindFlags
    ):
        """Performs the actual project-wide search."""
        mw = self.main_window
        if not hasattr(mw, "results_display_list"):
            logger.error("MainWindow hat kein results_display_list Attribut.")
            return

        mw.results_display_list.clear()
        mw.results_display_list.addItem(
            QListWidgetItem(f"--- Ergebnisse für '{search_term}' im Projekt ---")
        )

        found_count = 0

        case_sensitive = bool(search_flags & QTextDocument.FindFlag.FindCaseSensitively)
        whole_word = bool(search_flags & QTextDocument.FindFlag.FindWholeWords)

        if whole_word:
            pattern_term = r"\b" + re.escape(search_term) + r"\b"
        else:
            pattern_term = re.escape(search_term)

        regex_flags = 0 if case_sensitive else re.IGNORECASE

        try:
            search_regex = re.compile(pattern_term, regex_flags)
        except re.error as e:
            mw.results_display_list.addItem(
                QListWidgetItem(f"Fehler im Suchmuster: {e}")
            )
            mw.statusBar().showMessage("Fehler im Suchmuster.", 3000)
            return

        for root, dirs, files in os.walk(str(mw.project_root_path)):
            dirs[:] = [
                d
                for d in dirs
                if not any(
                    fnmatch.fnmatchcase(d, pattern)
                    for pattern in mw.EXCLUDE_DIR_PATTERNS_PROJECT_SEARCH
                )
            ]

            for filename in files:
                if (
                    Path(filename).suffix.lower()
                    not in mw.INCLUDE_EXTENSIONS_PROJECT_SEARCH
                ):
                    continue

                file_path = Path(root) / filename
                try:
                    relative_path = file_path.relative_to(mw.project_root_path)
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        for line_num, line_content in enumerate(f, 1):
                            for match in search_regex.finditer(line_content):
                                found_count += 1
                                item_text = f"{relative_path} : {line_num} : {line_content.strip()}"
                                list_item = QListWidgetItem(item_text)
                                list_item.setData(
                                    Qt.ItemDataRole.UserRole,
                                    {
                                        "file_path": str(file_path),
                                        "line_number": line_num,
                                        "column": match.start(),
                                    },
                                )
                                mw.results_display_list.addItem(list_item)
                                if found_count > 500:
                                    mw.results_display_list.addItem(
                                        QListWidgetItem(
                                            "... (Mehr als 500 Treffer, Suche gestoppt)"
                                        )
                                    )
                                    mw.statusBar().showMessage(
                                        f"Mehr als 500 Treffer für '{search_term}'.",
                                        3000,
                                    )
                                    return
                except Exception as e:
                    logger.warning(
                        f"Fehler beim Lesen/Durchsuchen der Datei {file_path}: {e}"
                    )
                    mw.results_display_list.addItem(
                        QListWidgetItem(f"Fehler bei Datei {file_path}: {e}")
                    )

        if found_count == 0:
            mw.results_display_list.addItem(QListWidgetItem("Keine Treffer gefunden."))
        mw.statusBar().showMessage(
            f"Projektweite Suche abgeschlossen: {found_count} Treffer für '{search_term}'.",
            5000,
        )
