# Klassen/ai_service_coordinator.py
# -*- coding: utf-8 -*-

"""
Coordinates all AI-related services and their interaction with the UI.

This class acts as a central hub that receives requests from the UI (e.g., via
button clicks or key events), invokes the appropriate backend service (like
linter, code corrector, or inference engine), and processes the results to
update the UI (e.g., show suggestions, display errors, or apply code changes).
"""

import ast
import logging
import re
from typing import TYPE_CHECKING, Optional, List, Dict, Any

from PySide6.QtCore import QObject, Slot, QTimer, Qt, QEvent
from PySide6.QtGui import QKeyEvent, QTextCursor
from PySide6.QtWidgets import (
    QListWidgetItem,
    QApplication,
    QMessageBox,
)

if TYPE_CHECKING:
    from ..ui.ui_main_window import MainWindow
    from ..tab_manager import TabManager
    from ..data.data_manager import DataManager
    from .inference_service import InferenceService
    from .code_corrector import CodeCorrector
    from .linter_service import LinterService, LintError
    from ..ui.code_editor_widget import CodeEditor
    from .code_context_analyzer import CodeContextAnalyzer
    from .formatting_service import FormattingService

logger = logging.getLogger(__name__)


class AIServiceCoordinator(QObject):
    """
    Orchestrates AI services and manages their UI interactions.
    """

    # Defines the number of characters before the cursor to use for autocomplete context.
    AUTOCOMPLETE_CONTEXT_WINDOW = 512

    def __init__(
        self,
        main_window: "MainWindow",
        tab_manager: "TabManager",
        data_manager: "DataManager",
        inference_service: Optional["InferenceService"],
        code_corrector: Optional["CodeCorrector"],
        linter_service: Optional["LinterService"],
        code_context_analyzer: "CodeContextAnalyzer",
        formatting_service: Optional["FormattingService"],
    ):

        """
        Initializes the AIServiceCoordinator.

        Args:
            main_window: Reference to the main application window.
            tab_manager: Service for managing editor tabs.
            data_manager: Service for database interactions.
            inference_service: Service for AI model inference (e.g., suggestions).
            code_corrector: Service for AI-based code correction.
            linter_service: Service for code linting (e.g., flake8).
            code_context_analyzer: Service for static code analysis.
            formatting_service: Service for code formatting (e.g., black).
        """
        super().__init__(main_window)
        self.main_window = main_window
        self.tab_manager = tab_manager
        self.data_manager = data_manager
        self.inference_service = inference_service
        self.code_corrector = code_corrector
        self.linter_service = linter_service
        self.code_context_analyzer = code_context_analyzer
        self.formatting_service = formatting_service

        # Connect to the autocomplete popup if it exists.
        if (
            hasattr(self.main_window, "autocomplete_popup")
            and self.main_window.autocomplete_popup
        ):

            self.main_window.autocomplete_popup.itemActivated.connect(
                self.insert_selected_completion
            )
            logger.info(
                "AIServiceCoordinator: autocomplete_popup.itemActivated verbunden."
            )
        else:
            logger.error(
                "AIServiceCoordinator: MainWindow.autocomplete_popup nicht gefunden bei Initialisierung!"
            )

        logger.info("AIServiceCoordinator initialisiert.")

    def install_event_filter_on_editor(self, editor: "CodeEditor"):
        """
        Installs an event filter on the given editor for AI-related key events.

        Args:
            editor: The CodeEditor widget to monitor.
        """
        editor.installEventFilter(self)
        logger.debug(
            f"EventFilter auf Editor {editor} installiert durch AIServiceCoordinator."
        )

    def eventFilter(self, watched_object: QObject, event: QEvent) -> bool:
        """
        Handles key events for AI features on the watched editor and autocomplete popup.

        This central event filter captures key presses to trigger actions like
        autocomplete (Ctrl+Space) or to handle navigation within the popup.

        Args:
            watched_object: The object that emitted the event.
            event: The event that occurred.

        Returns:
            True if the event was handled and should be stopped, False otherwise.
        """
        current_editor = self.tab_manager.get_current_editor()
        mw = self.main_window

        # Event filter for the code editor (Ctrl+Space for autocomplete)
        if (
            current_editor
            and watched_object is current_editor
            and event.type() == QEvent.Type.KeyPress
        ):
            key_event = QKeyEvent(event)  # type: ignore

            if (
                key_event.key() == Qt.Key.Key_Space
                and key_event.modifiers() == Qt.KeyboardModifier.ControlModifier
            ):
                logger.debug(
                    "AIServiceCoordinator: Strg+Leertaste (Vorschläge) im Editor erkannt."
                )
                self.handle_autocomplete_request(current_editor)
                return True  # Mark event as handled

        # Event filter for the autocomplete popup (Enter, Escape, Arrow keys)
        if (
            hasattr(mw, "autocomplete_popup")
            and mw.autocomplete_popup
            and watched_object is mw.autocomplete_popup
            and mw.autocomplete_popup.isVisible()
        ):
            if event.type() == QEvent.Type.KeyPress:
                key_event = QKeyEvent(event)
                if (
                    key_event.key() == Qt.Key.Key_Return
                    or key_event.key() == Qt.Key.Key_Enter
                ):
                    current_item = mw.autocomplete_popup.currentItem()
                    if current_item:
                        self.insert_selected_completion(current_item)
                    elif mw.autocomplete_popup.count() > 0:
                        # If no item is selected, use the first one
                        self.insert_selected_completion(
                            mw.autocomplete_popup.item(0)
                        )
                    return True
                elif key_event.key() == Qt.Key.Key_Escape:
                    mw.autocomplete_popup.hide()
                    if current_editor:
                        current_editor.setFocus()
                    return True
                # Let the QListWidget handle navigation keys
                elif key_event.key() in [
                    Qt.Key.Key_Up,
                    Qt.Key.Key_Down,
                    Qt.Key.Key_PageUp,
                    Qt.Key.Key_PageDown,
                ]:
                    return False

            elif event.type() == QEvent.Type.FocusOut:
                pass

        return super().eventFilter(watched_object, event)

    @Slot(QListWidgetItem)
    def insert_selected_completion(self, list_item: QListWidgetItem):
        """
        Intelligently inserts the selected completion text into the editor.

        This method replaces a partially typed word or inserts a new word,
        and then saves the user's choice as feedback for the AI model.

        Args:
            list_item: The QListWidgetItem selected by the user.
        """
        completion_text = list_item.text().strip()
        current_editor = self.tab_manager.get_current_editor()
        current_tab_data = self.tab_manager.get_current_tab_data()
        mw = self.main_window

        if not all([current_editor, current_tab_data, completion_text]) or (
            completion_text.startswith("[Fehler")
            or completion_text.startswith("[Error")
        ):
            if hasattr(mw, "autocomplete_popup") and mw.autocomplete_popup:
                mw.autocomplete_popup.hide()
            return

        cursor = current_editor.textCursor()

        # Find the word prefix right before the cursor
        pos_in_line = cursor.positionInBlock()
        line_text = cursor.block().text()
        text_before_cursor = line_text[:pos_in_line]

        match = re.search(r"(\w*)$", text_before_cursor)
        prefix = ""
        if match:
            prefix = match.group(1)

        # --- Intelligent Insertion Logic ---
        if prefix and completion_text.startswith(prefix):
            # Case A: Completion extends the typed prefix (e.g., prefix="imp", completion="import")
            # Replace the typed prefix with the full completion.
            cursor.movePosition(
                QTextCursor.MoveOperation.PreviousCharacter,
                QTextCursor.MoveMode.KeepAnchor,
                len(prefix),
            )
            cursor.insertText(completion_text)
            logger.debug(
                f"Autocomplete: Ersetze Präfix '{prefix}' mit '{completion_text}'."
            )

        else:
            # Case B: Prefix is empty or completion is a different word.
            # Insert the completion as a new word.
            text_to_insert = completion_text
            last_char_before = text_before_cursor[-1:] if text_before_cursor else ""
            if (
                last_char_before
                and not last_char_before.isspace()
                and last_char_before not in "([{,."
            ):
                text_to_insert = " " + text_to_insert

            cursor.insertText(text_to_insert)
            logger.debug(f"Autocomplete: Füge neues Wort '{text_to_insert}' ein.")

        # Save feedback for the used suggestion
        if self.data_manager and current_tab_data.get("last_prefix_for_suggestion"):
            self.data_manager.add_ki_feedback(
                source_code_context=current_tab_data["last_prefix_for_suggestion"],
                ai_action_type="suggestion",
                ai_generated_output=completion_text,
                user_feedback_type="suggestion_used",
                original_file_path=(
                    str(current_tab_data["path"]) if current_tab_data["path"] else None
                ),
            )
            logger.info(
                f"KI-Feedback 'suggestion_used' für '{completion_text}' gespeichert."
            )
        current_tab_data["last_prefix_for_suggestion"] = None

        # Hide popup and return focus to editor
        if hasattr(mw, "autocomplete_popup") and mw.autocomplete_popup:
            mw.autocomplete_popup.hide()
        current_editor.setFocus()
        if hasattr(mw, "update_button_states"):
            mw.update_button_states()

    def handle_autocomplete_request(self, editor: "CodeEditor"):
        """
        Triggers an autocomplete request to the AI service and displays the results.

        Args:
            editor: The editor widget where the request originated.
        """
        mw = self.main_window
        if not self.inference_service:
            if mw.statusBar():
                mw.statusBar().showMessage(
                    "Inference Service nicht bereit für Vorschläge.", 2000
                )
            return

        current_code = editor.toPlainText()
        current_tab_data = self.tab_manager.get_current_tab_data()
        if not current_tab_data:
            return

        # Get context from the entire file for better suggestions
        current_file_path = current_tab_data.get("path")
        ai_context: Optional[Dict[str, Any]] = None
        if current_file_path and self.code_context_analyzer:
            logger.debug(f"Rufe Kontext für Autocomplete ab: {current_file_path.name}")
            ai_context = self.code_context_analyzer.get_context_for_ai(
                current_code, current_file_path
            )
        else:
            logger.debug(
                "Kein Dateipfad für Kontext oder CodeContextAnalyzer nicht verfügbar."
            )

        # The prefix sent to the AI includes a window of text before the cursor
        cursor_position = editor.textCursor().position()

        start_pos = max(0, cursor_position - self.AUTOCOMPLETE_CONTEXT_WINDOW)
        prefix_for_suggestion = current_code[start_pos:cursor_position]
        logger.debug(
            f"Fokussierter Präfix für Suggestion (letzte {self.AUTOCOMPLETE_CONTEXT_WINDOW} Zeichen):\n---\n{prefix_for_suggestion}\n---"
        )

        prefix_for_suggestion = current_code[:cursor_position]
        current_tab_data["last_prefix_for_suggestion"] = prefix_for_suggestion

        suggestions = self.inference_service.get_suggestions(
            prefix_for_suggestion, context=ai_context
        )

        # Display the suggestions in the autocomplete popup
        if hasattr(mw, "autocomplete_popup") and mw.autocomplete_popup:
            mw.autocomplete_popup.hide()
        QApplication.processEvents()  # Ensure UI updates before showing new popup
        if hasattr(editor, "clear_lint_errors"):
            editor.clear_lint_errors()
        valid_suggestions = [
            s
            for s in suggestions
            if s and not (s.startswith("[Fehler") or s.startswith("[Error"))
        ]
        if (
            valid_suggestions
            and hasattr(mw, "autocomplete_popup")
            and mw.autocomplete_popup
        ):
            mw.autocomplete_popup.clear()
            for sug in valid_suggestions:
                mw.autocomplete_popup.addItem(QListWidgetItem(sug))
            cursor_rect = editor.cursorRect()
            global_pos = editor.mapToGlobal(cursor_rect.bottomLeft())
            mw.autocomplete_popup.move(global_pos)

            # Resize popup to fit content
            font_metrics = mw.autocomplete_popup.fontMetrics()
            row_height = font_metrics.height() + 4
            popup_height = min(row_height * len(valid_suggestions) + 6, 200)
            max_text_width = (
                max(
                    (font_metrics.horizontalAdvance(s) for s in valid_suggestions),
                    default=100,
                )
                if valid_suggestions
                else 100
            )
            popup_width = min(max_text_width + 40, 350)
            mw.autocomplete_popup.resize(popup_width, popup_height)
            mw.autocomplete_popup.show()
            if mw.autocomplete_popup.count() > 0:
                QTimer.singleShot(
                    0,
                    lambda: (
                        mw.autocomplete_popup.setCurrentRow(0)
                        if hasattr(mw, "autocomplete_popup") and mw.autocomplete_popup
                        else None
                    ),
                )
            QTimer.singleShot(
                0,
                lambda: (
                    mw.autocomplete_popup.setFocus()
                    if hasattr(mw, "autocomplete_popup") and mw.autocomplete_popup
                    else None
                ),
            )
            if hasattr(mw.autocomplete_popup, "installEventFilter"):
                mw.autocomplete_popup.installEventFilter(self)  # EventFilter für Popup
        elif hasattr(mw, "results_display_list") and mw.results_display_list:
            mw.results_display_list.clear()
            mw.results_display_list.addItem(
                QListWidgetItem("Keine Code-Vervollständigungsvorschläge gefunden.")
            )
        if hasattr(mw, "update_button_states"):
            mw.update_button_states()

    @Slot(object, bool)
    def handle_lint_request(
        self, editor_instance: "CodeEditor", triggered_by_timer: bool = False
    ):
        """
        Handles a linting request for the given editor instance.

        Args:
            editor_instance: The editor to lint.
            triggered_by_timer: True if the linting was triggered automatically.
        """
        mw = self.main_window
        current_editor = editor_instance

        if not current_editor:
            if mw.statusBar():
                mw.statusBar().showMessage("Kein Editor für Linting vorhanden.", 3000)
            return

        if not self.linter_service:
            if (
                mw._get_current_editor() == current_editor
                and hasattr(mw, "results_display_list")
                and mw.results_display_list
            ):
                mw.results_display_list.clear()
                mw.results_display_list.addItem(
                    QListWidgetItem("Linter Service nicht verfügbar.")
                )
            return

        current_code = current_editor.toPlainText()
        if not current_code.strip():
            if (
                mw._get_current_editor() == current_editor
                and hasattr(mw, "results_display_list")
                and mw.results_display_list
            ):
                mw.results_display_list.clear()
                mw.results_display_list.addItem(
                    QListWidgetItem("Kein Code zum Linten vorhanden.")
                )
            if hasattr(current_editor, "clear_lint_errors"):
                current_editor.clear_lint_errors()
            return

        if not triggered_by_timer and mw._get_current_editor() == current_editor:
            if mw.statusBar():
                mw.statusBar().showMessage("Führe Code-Linting durch (Flake8)...", 2000)
        QApplication.processEvents()

        linting_results: List["LintError"] = self.linter_service.lint_code(current_code)

        # Update UI only if the linted editor is still the active one
        if (
            mw._get_current_editor() == current_editor
            and hasattr(mw, "results_display_list")
            and mw.results_display_list
        ):
            mw.results_display_list.clear()
            mw.results_display_list.addItem(
                QListWidgetItem("--- Flake8 Linter Ergebnisse ---")
            )
            if not linting_results:
                mw.results_display_list.addItem(
                    QListWidgetItem("Keine Fehler oder Warnungen von Flake8 gefunden.")
                )
                if not triggered_by_timer and mw.statusBar():
                    mw.statusBar().showMessage(
                        "Linting abgeschlossen: Keine Fehler gefunden.", 3000
                    )
            else:
                for error in linting_results:
                    item_text = f"Z.{error.get('line', 'N/A')}, Sp.{error.get('column', 'N/A')}: [{error.get('code', 'N/A')}] {error.get('message', 'N/A')}"
                    mw.results_display_list.addItem(QListWidgetItem(item_text))
                if not triggered_by_timer and mw.statusBar():
                    mw.statusBar().showMessage(
                        f"Linting abgeschlossen: {len(linting_results)} Problem(e) gefunden.",
                        3000,
                    )
            mw.results_display_list.scrollToBottom()

        if hasattr(current_editor, "set_lint_errors"):
            current_editor.set_lint_errors(linting_results)

        if mw._get_current_editor() == current_editor and hasattr(
            mw, "update_button_states"
        ):
            mw.update_button_states()

    @Slot()
    def handle_code_correction_request(self):
        """Initiates an AI-based code correction for the current tab."""
        mw = self.main_window
        current_tab_data = self.tab_manager.get_current_tab_data()
        editor = self.tab_manager.get_current_editor()

        if not (current_tab_data and editor):
            if mw.statusBar():
                mw.statusBar().showMessage("Kein aktives Tab für Code-Korrektur.", 3000)
            return
        if not self.code_corrector:
            if hasattr(mw, "results_display_list") and mw.results_display_list:
                mw.results_display_list.clear()
                mw.results_display_list.addItem(
                    QListWidgetItem("Code Corrector Service nicht verfügbar.")
                )
            return

        current_code = editor.toPlainText()
        if not current_code.strip():
            if hasattr(mw, "results_display_list") and mw.results_display_list:
                mw.results_display_list.clear()
                mw.results_display_list.addItem(
                    QListWidgetItem("Kein Code zur Korrektur vorhanden.")
                )
            return

        current_file_path = current_tab_data.get("path")
        ai_context: Optional[Dict[str, Any]] = None
        if current_file_path and self.code_context_analyzer:
            logger.debug(
                f"Rufe Kontext für Code-Korrektur ab: {current_file_path.name}"
            )
            ai_context = self.code_context_analyzer.get_context_for_ai(
                current_code, current_file_path
            )
        else:
            logger.debug(
                "Kein Dateipfad für Kontext oder CodeContextAnalyzer nicht verfügbar für Korrektur."
            )

        if mw.statusBar():
            mw.statusBar().showMessage(
                "Starte KI-Code-Analyse & Korrektur (akt. Tab)...", 0
            )
        if hasattr(mw, "results_display_list") and mw.results_display_list:
            mw.results_display_list.clear()
            mw.results_display_list.addItem(
                QListWidgetItem("--- Starte iterative Code-Analyse & KI-Korrektur ---")
            )
        QApplication.processEvents()

        current_tab_data["source_code_for_last_correction"] = current_code
        current_tab_data["feedback_given_for_last_correction"] = False

        analysis_result_dict = self.code_corrector.analyze_and_correct(
            current_code, context=ai_context
        )
        status = analysis_result_dict.get("status", "Unbekannt")
        final_code = analysis_result_dict.get("code", current_code)

        if hasattr(mw, "results_display_list") and mw.results_display_list:
            mw.results_display_list.addItem(
                QListWidgetItem(f"Analyse Status: {status.replace('_', ' ').title()}")
            )
            for log_line in analysis_result_dict.get("log", []):
                for (
                    sub_line
                ) in (
                    log_line.splitlines()
                ):
                    mw.results_display_list.addItem(QListWidgetItem(f"  {sub_line}"))

            error_details_dict = analysis_result_dict.get("error_details")
            if error_details_dict and isinstance(error_details_dict, dict):
                mw.results_display_list.addItem(
                    QListWidgetItem("--- Verbleibende Fehlerdetails ---")
                )

                for key, val in error_details_dict.items():
                    mw.results_display_list.addItem(QListWidgetItem(f"  {key}: {val}"))
            mw.results_display_list.scrollToBottom()

        if (
            status in ["corrected_and_formatted", "formatted", "corrected"]
            and final_code.strip() != current_code.strip()
        ):
            current_tab_data["last_suggested_code"] = final_code
            if mw.statusBar():
                mw.statusBar().showMessage(
                    "Korrekturvorschlag verfügbar. Feedback geben oder 'Korrekturen anwenden'.",
                    5000,
                )
        else:
            current_tab_data["last_suggested_code"] = None
            current_tab_data["source_code_for_last_correction"] = None
            if mw.statusBar():
                mw.statusBar().showMessage(
                    f"Code-Korrektur Status: {status.replace('_', ' ').title()}", 3000
                )

        if hasattr(mw, "update_button_states"):
            mw.update_button_states()

    @Slot()
    def handle_apply_corrections(self):
        """Applies the last suggested AI correction to the current editor."""
        mw = self.main_window
        current_tab_data = self.tab_manager.get_current_tab_data()
        if not current_tab_data or current_tab_data.get("last_suggested_code") is None:
            if mw.statusBar():
                mw.statusBar().showMessage(
                    "Kein anwendbarer Korrekturvorschlag im aktuellen Tab vorhanden.",
                    3000,
                )
            return

        editor = current_tab_data["editor"]
        last_suggested_code = current_tab_data["last_suggested_code"]
        source_code_for_correction = current_tab_data["source_code_for_last_correction"]
        current_path = current_tab_data["path"]

        reply = QMessageBox.question(
            mw,
            "Korrekturen anwenden",
            "Möchtest du den aktuellen Editorinhalt mit dem korrigierten Code überschreiben?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            if self.data_manager and source_code_for_correction is not None:
                self.data_manager.add_ki_feedback(
                    source_code_context=source_code_for_correction,
                    ai_action_type="correction",
                    ai_generated_output=last_suggested_code,
                    user_feedback_type="correction_applied",
                    original_file_path=str(current_path) if current_path else None,
                )
                logger.info("KI-Feedback 'correction_applied' gespeichert.")

            current_cursor_pos = editor.textCursor().position()
            editor.setPlainText(last_suggested_code)
            new_cursor = editor.textCursor()
            new_pos = min(
                current_cursor_pos, len(last_suggested_code)
            )
            new_cursor.setPosition(new_pos)
            editor.setTextCursor(new_cursor)

            if mw.statusBar():
                mw.statusBar().showMessage(
                    "Korrekturen angewendet. Sie können diese jetzt bewerten.", 3000
                )
            logger.info("Korrekturen auf Code angewendet.")

            editor.document().setModified(True)
        else:
            if mw.statusBar():
                mw.statusBar().showMessage(
                    "Anwenden der Korrekturen abgebrochen.", 3000
                )
            logger.info("Anwenden der Korrekturen abgebrochen.")
        if hasattr(mw, "update_button_states"):
            mw.update_button_states()

    def _handle_correction_feedback(self, feedback_type: str):
        """
        A generic handler for saving feedback (good/bad/applied) for the last correction.

        Args:
            feedback_type: The type of feedback to save (e.g., "correction_good").
        """
        mw = self.main_window
        current_tab_data = self.tab_manager.get_current_tab_data()

        # Check if all required data for feedback is present
        if not (
            self.data_manager
            and current_tab_data
            and current_tab_data.get("last_suggested_code")
            and current_tab_data.get("source_code_for_last_correction")
        ):
            logger.warning(
                f"Konnte '{feedback_type}'-Feedback nicht speichern: Bedingungen nicht erfüllt."
            )
            if mw.statusBar():
                mw.statusBar().showMessage(
                    "Fehler: Feedback konnte nicht gespeichert werden.", 3000
                )
            return

        self.data_manager.add_ki_feedback(
            source_code_context=current_tab_data["source_code_for_last_correction"],
            ai_action_type="correction",
            ai_generated_output=current_tab_data["last_suggested_code"],
            user_feedback_type=feedback_type,
            original_file_path=(
                str(current_tab_data["path"]) if current_tab_data["path"] else None
            ),
        )
        if mw.statusBar():
            mw.statusBar().showMessage(f"Feedback '{feedback_type}' gespeichert.", 3000)
        logger.info(f"KI-Feedback '{feedback_type}' gespeichert.")

        # Mark feedback as given to prevent multiple submissions for the same correction
        current_tab_data["feedback_given_for_last_correction"] = True
        if hasattr(mw, "update_button_states"):
            mw.update_button_states()

    @Slot()
    def handle_correction_feedback_good(self):
        self._handle_correction_feedback("correction_good")

    @Slot()
    def handle_correction_feedback_bad(self):
        self._handle_correction_feedback("correction_bad")

    @Slot(str, bool)  # code_to_improve, is_selection
    def handle_code_improvement_request(self, code_to_improve: str, is_selection: bool):
        """
        Handles a request to improve a snippet of code using the AI.

        Args:
            code_to_improve: The code snippet (selection or full text) to improve.
            is_selection: A boolean indicating if the code is from a selection.
        """
        mw = self.main_window
        current_tab_data = self.tab_manager.get_current_tab_data()
        editor = self.tab_manager.get_current_editor()

        if not current_tab_data or not editor:
            logger.warning("Kein aktives Tab/Editor für Code-Verbesserung.")
            if mw.statusBar():
                mw.statusBar().showMessage("Kein aktives Tab für Verbesserung.", 3000)
            return

        if not self.inference_service or not hasattr(
            self.inference_service, "improve_code_snippet"
        ):
            logger.warning(
                "InferenceService oder Methode 'improve_code_snippet' nicht verfügbar."
            )
            if mw.statusBar():
                mw.statusBar().showMessage(
                    "KI-Dienst für Verbesserung nicht bereit.", 3000
                )
            return

        full_current_code = editor.toPlainText()
        current_file_path = current_tab_data.get("path")

        ai_context: Optional[Dict[str, Any]] = None
        if current_file_path and self.code_context_analyzer:
            try:
                ast.parse(full_current_code)
                logger.debug(
                    f"Rufe Kontext für Code-Verbesserung ab (Datei ist parsebar): {current_file_path.name}"
                )
                ai_context = self.code_context_analyzer.get_context_for_ai(
                    full_current_code, current_file_path
                )
            except SyntaxError:
                logger.warning(
                    "Gesamtcode enthält Syntaxfehler. Kontext für Verbesserung wird leer sein."
                )
                ai_context = {
                    "file_path": current_file_path.name,
                    "imports_context": "Imports: [] # File has syntax errors",
                    "definitions_context": "Definitions: [] # File has syntax errors",
                }
        else:
            logger.debug(
                "Kein Dateipfad oder CodeContextAnalyzer für Verbesserung verfügbar."
            )

        logger.info(
            f"Anfrage zur Code-Verbesserung für (markiert: {is_selection}):\n{code_to_improve[:200]}..."
        )
        if ai_context:
            imports_log = ai_context.get("imports_context", "N/A")
            defs_log = ai_context.get("definitions_context", "N/A")
            logger.info(
                f"  Mit Kontext: Imports='{imports_log[:100]}...', Defs='{defs_log[:100]}...'"
            )

        if hasattr(mw, "results_display_list") and mw.results_display_list:
            mw.results_display_list.clear()
            mw.results_display_list.addItem(
                QListWidgetItem("--- Anfrage zur KI-Code-Verbesserung ---")
            )
            mw.results_display_list.addItem(
                QListWidgetItem(
                    f"Zu verbessernder Code ({'Markierung' if is_selection else 'Gesamter Tab'}):\n{code_to_improve}"
                )
            )
            if ai_context:
                mw.results_display_list.addItem(
                    QListWidgetItem(
                        f"Kontext (Importe): {ai_context.get('imports_context')}"
                    )
                )
                mw.results_display_list.addItem(
                    QListWidgetItem(
                        f"Kontext (Definitionen): {ai_context.get('definitions_context')}"
                    )
                )
            mw.results_display_list.addItem(
                QListWidgetItem("Warte auf Antwort der KI...")
            )

        if mw.statusBar():
            mw.statusBar().showMessage(
                "Sende Anfrage zur Code-Verbesserung an KI...", 0
            )
        QApplication.processEvents()

        improved_code_or_error = self.inference_service.improve_code_snippet(
            code_to_improve, ai_context
        )

        self._handle_improvement_response(
            improved_code_or_error, is_selection, code_to_improve
        )

    def _handle_improvement_response(
        self, improved_code: str, is_selection: bool, original_snippet: str
    ):
        """
        Handles the AI's response for a code improvement request.

        Args:
            improved_code: The improved code returned by the AI.
            is_selection: Whether the original request was for a selection.
            original_snippet: The original code that was sent for improvement.
        """
        mw = self.main_window
        if hasattr(mw, "results_display_list") and mw.results_display_list:
            for i in range(
                mw.results_display_list.count() - 1, -1, -1
            ):
                if (
                    "Warte auf Antwort der KI..."
                    in mw.results_display_list.item(i).text()
                ):
                    mw.results_display_list.takeItem(i)
                    break
            mw.results_display_list.addItem(
                QListWidgetItem("--- Antwort der KI für Verbesserung ---")
            )

            suggestion_item = QListWidgetItem(improved_code)
            mw.results_display_list.addItem(suggestion_item)
            mw.results_display_list.scrollToBottom()

        if mw.statusBar():
            mw.statusBar().showMessage(
                "Vorschlag zur Code-Verbesserung erhalten.", 5000
            )

        current_editor = self.tab_manager.get_current_editor()
        if current_editor and not improved_code.startswith("[Fehler:"):
            if improved_code.strip() != original_snippet.strip():
                replace_msg = (
                    "Möchtest du den markierten Text durch den KI-Vorschlag ersetzen?"
                    if is_selection
                    else "Möchtest du den gesamten Code im Editor durch den KI-Vorschlag ersetzen?"
                )

                reply = QMessageBox.question(
                    mw,
                    "Verbesserung anwenden?",
                    f"{replace_msg}\n\nKI-Vorschlag:\n---\n{improved_code}\n---",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    cursor = current_editor.textCursor()
                    if is_selection and cursor.hasSelection():
                        cursor.insertText(improved_code)
                        logger.info("Markierter Text durch KI-Verbesserung ersetzt.")
                    elif not is_selection:
                        current_editor.setPlainText(improved_code)
                        logger.info("Gesamter Code durch KI-Verbesserung ersetzt.")
                    current_editor.document().setModified(True)
            else:
                logger.info(
                    "KI-Verbesserung ist identisch mit dem Original-Snippet oder leer."
                )
                if mw.statusBar():
                    mw.statusBar().showMessage(
                        "Keine Änderung durch KI vorgeschlagen.", 3000
                    )

    @Slot()
    def handle_format_code_request(self):
        """
        Handles a request to format the code in the current editor using black.
        Formats the selection if one exists, otherwise formats the entire document.
        """
        if not self.formatting_service:
            logger.warning(
                "Formatierungsanfrage erhalten, aber FormattingService ist nicht verfügbar."
            )
            self.main_window.statusBar().showMessage(
                "Formatierungsdienst nicht verfügbar.", 3000
            )
            return

        editor = self.tab_manager.get_current_editor()
        if not editor:
            self.main_window.statusBar().showMessage(
                "Kein aktiver Editor zum Formatieren.", 3000
            )
            return

        cursor = editor.textCursor()
        is_selection = cursor.hasSelection()

        if is_selection:
            code_to_format = cursor.selectedText()
            log_message = "Formatiere ausgewählten Code..."
        else:
            code_to_format = editor.toPlainText()
            log_message = "Formatiere gesamten Code im aktuellen Tab..."

        if not code_to_format.strip():
            self.main_window.statusBar().showMessage(
                "Kein Code zum Formatieren vorhanden.", 2000
            )
            return

        logger.info(log_message)
        self.main_window.statusBar().showMessage(log_message, 2000)

        success, result_text = self.formatting_service.format_code(code_to_format)

        if success:
            if result_text != code_to_format:
                if is_selection:
                    cursor.insertText(result_text)
                    logger.info("Auswahl erfolgreich formatiert.")
                else:
                    original_pos = cursor.position()
                    editor.setPlainText(result_text)
                    editor.document().setModified(True)
                    new_cursor = editor.textCursor()
                    new_cursor.setPosition(min(original_pos, len(result_text)))
                    editor.setTextCursor(new_cursor)
                    logger.info("Gesamtes Dokument erfolgreich formatiert.")
                self.main_window.statusBar().showMessage(
                    "Code erfolgreich formatiert.", 3000
                )
            else:
                logger.info("Code ist bereits korrekt formatiert.")
                self.main_window.statusBar().showMessage(
                    "Code ist bereits formatiert.", 2000
                )
        else:
            logger.error(f"Fehler bei der Code-Formatierung: {result_text}")
            QMessageBox.warning(self.main_window, "Formatierungsfehler", result_text)
            self.main_window.statusBar().showMessage(
                "Fehler bei der Formatierung.", 5000
            )
