# Klassen/tab_manager.py
# -*- coding: utf-8 -*-

from PySide6.QtWidgets import QTabWidget
from PySide6.QtCore import QObject, Signal, QTimer
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Tuple

# TYPE_CHECKING Block für Zirkelbezüge bei Typ-Hinweisen
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ui.code_editor_widget import CodeEditor
    from .core.python_highlighter import PythonHighlighter
    from .data.data_manager import DataManager
    from .ui.ui_main_window import MainWindow  # Für Linter-Trigger-Callback

logger = logging.getLogger(__name__)


class TabManager(QObject):
    """
    Verwaltet die Editor-Tabs, deren Zustand und Interaktionen.
    """

    # Signale
    current_tab_changed_signal = Signal(
        int, object
    )  # index, tab_data (kann None sein, wenn kein Tab)
    tab_added_signal = Signal(int, object)  # index, tab_data
    tab_closed_signal = Signal(object)  # path_of_closed_tab (Path or None)
    last_tab_closed_signal = Signal()
    tab_modification_changed_signal = Signal(int, bool)  # index, is_modified
    # Signal, das von MainWindow verbunden wird, um Linting für einen Editor auszulösen
    lint_request_for_editor_signal = Signal(object)  # editor_instance
    tab_title_updated_signal = Signal(int, str)  # Hinzugefügt für Titel-Updates

    def __init__(
        self,
        tab_widget: QTabWidget,
        data_manager: "DataManager",
        code_editor_class: type,  # Klasse CodeEditor
        python_highlighter_class: Optional[type],  # Klasse PythonHighlighter
        linting_delay_ms: int,
        quality_labels: List[str],
        main_window_ref: "MainWindow",  # Referenz für Linter-Callback
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.tab_widget = tab_widget
        self.data_manager = data_manager
        self.CodeEditor = code_editor_class
        self.PythonHighlighter = python_highlighter_class
        self.LINTING_DELAY_MS = linting_delay_ms
        self.QUALITY_LABELS = quality_labels
        self.main_window = main_window_ref  # Für Linter-Trigger

        self.open_tabs_data: List[Dict[str, Any]] = []

        self.tab_widget.currentChanged.connect(self._on_internal_current_tab_changed)
        # tabCloseRequested wird von MainWindow gehandhabt, die dann remove_tab aufruft

        logger.info("TabManager initialisiert.")

    def _on_internal_current_tab_changed(self, index: int):
        """Interner Slot, reagiert auf QTabWidget.currentChanged."""
        tab_data = self.get_tab_data(index) if index != -1 else None
        logger.debug(
            f"TabManager: Internes currentChanged, Index: {index}, Tab-Pfad: {tab_data.get('path') if tab_data else 'Kein Tab'}"
        )
        self.current_tab_changed_signal.emit(index, tab_data)

    def add_new_tab(
        self,
        file_path: Optional[Path] = None,
        content: str = "",
        set_as_current: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Fügt ein neues Editor-Tab hinzu und gibt den Index und die Tab-Daten zurück.
        """
        editor: "CodeEditor" = self.CodeEditor()
        editor.setPlainText(content)
        # Der EventFilter für Autocomplete wird von MainWindow global auf den *aktuellen* Editor gesetzt,
        # nicht auf jeden einzelnen hier, da das Popup global ist.

        highlighter: Optional["PythonHighlighter"] = None
        if self.PythonHighlighter:
            highlighter = self.PythonHighlighter(editor.document())

        tab_data: Dict[str, Any] = {
            "editor": editor,
            "path": file_path,
            "highlighter": highlighter,
            # 'is_modified' wird direkt vom Dokument des Editors abgefragt
            "quality_label": self.data_manager.DEFAULT_QUALITY_LABEL,
            "last_suggested_code": None,
            "source_code_for_last_correction": None,
            "last_prefix_for_suggestion": None,
            "feedback_given_for_last_correction": False,
            "lint_delay_timer": QTimer(self),  # Parent ist TabManager
        }

        tab_data["lint_delay_timer"].setSingleShot(True)
        tab_data["lint_delay_timer"].setInterval(self.LINTING_DELAY_MS)
        # Verbinde das Timeout-Signal des Timers mit dem Linter-Request-Signal
        tab_data["lint_delay_timer"].timeout.connect(
            lambda ed=editor: self.lint_request_for_editor_signal.emit(ed)
        )

        editor.textChanged.connect(lambda td=tab_data: self._on_editor_text_changed(td))
        editor.document().modificationChanged.connect(
            lambda modified, td=tab_data: self._on_editor_modification_changed(
                td, modified
            )
        )

        self.open_tabs_data.append(tab_data)

        tab_name = (
            file_path.name if file_path else f"Unbenannt-{len(self.open_tabs_data)}"
        )
        tab_index = self.tab_widget.addTab(editor, tab_name)

        if set_as_current:
            self.tab_widget.setCurrentWidget(
                editor
            )  # Macht das Tab aktiv (löst currentChanged aus)

        if file_path and self.data_manager:
            quality = self.data_manager.get_snippet_quality(str(file_path))
            if quality:
                tab_data["quality_label"] = quality

        self.update_tab_title(tab_index)  # Titel initial setzen
        self.tab_added_signal.emit(tab_index, tab_data)
        logger.info(
            f"Neues Tab hinzugefügt: Index {tab_index}, Pfad: {file_path or 'Unbenannt'}"
        )
        return tab_index, tab_data

    def _on_editor_text_changed(self, tab_data: Dict[str, Any]):
        editor = tab_data["editor"]
        if hasattr(
            editor, "clear_lint_errors"
        ):  # Sicherstellen, dass Methode existiert
            editor.clear_lint_errors()

        lint_timer = tab_data.get("lint_delay_timer")
        if lint_timer:
            if editor.toPlainText().strip():
                lint_timer.start()
            else:
                lint_timer.stop()
                # MainWindow könnte hier das Log für das aktuelle Tab leeren, wenn es dieses Tab ist

    def _on_editor_modification_changed(self, tab_data: Dict[str, Any], modified: bool):
        try:
            tab_index = self.open_tabs_data.index(tab_data)
            self.update_tab_title(tab_index)
            self.tab_modification_changed_signal.emit(tab_index, modified)
        except ValueError:
            logger.warning(
                f"TabManager: Konnte Tab-Daten für modificationChanged nicht finden: {tab_data.get('path')}"
            )

    def get_tab_data(self, index: int) -> Optional[Dict[str, Any]]:
        if 0 <= index < len(self.open_tabs_data):
            return self.open_tabs_data[index]
        return None

    def get_current_tab_data(self) -> Optional[Dict[str, Any]]:
        current_index = self.tab_widget.currentIndex()
        return self.get_tab_data(current_index)

    def get_editor(self, index: int) -> Optional["CodeEditor"]:
        tab_data = self.get_tab_data(index)
        return tab_data.get("editor") if tab_data else None

    def get_current_editor(self) -> Optional["CodeEditor"]:
        tab_data = self.get_current_tab_data()
        return tab_data.get("editor") if tab_data else None

    def get_current_editor_index(self) -> int:
        return self.tab_widget.currentIndex()

    def get_path(self, index: int) -> Optional[Path]:
        tab_data = self.get_tab_data(index)
        return tab_data.get("path") if tab_data else None

    def set_path(self, index: int, file_path: Path):
        tab_data = self.get_tab_data(index)
        if tab_data:
            tab_data["path"] = file_path
            self.update_tab_title(index)

    def get_quality_label(self, index: int) -> Optional[str]:
        tab_data = self.get_tab_data(index)
        return tab_data.get("quality_label") if tab_data else None

    def set_quality_label(self, index: int, label: str):
        tab_data = self.get_tab_data(index)
        if tab_data:
            tab_data["quality_label"] = label
            editor = tab_data.get("editor")
            if editor and tab_data.get("path"):
                editor.document().setModified(True)

    def update_tab_title(self, index: int):
        tab_data = self.get_tab_data(index)
        if tab_data and 0 <= index < self.tab_widget.count():
            editor = tab_data["editor"]
            path = tab_data.get("path")
            base_name = path.name if path else f"Unbenannt-{index+1}"
            title = base_name + ("*" if editor.document().isModified() else "")
            self.tab_widget.setTabText(index, title)
            self.tab_title_updated_signal.emit(index, title)

    def remove_tab(self, index: int) -> Optional[Path]:
        """Entfernt das Tab am gegebenen Index. MainWindow ist für maybe_save zuständig."""
        if not (
            0 <= index < self.tab_widget.count() and index < len(self.open_tabs_data)
        ):
            logger.warning(f"TabManager: Ungültiger Index {index} für remove_tab.")
            return None

        tab_data_to_close = self.open_tabs_data.pop(index)
        closed_path = tab_data_to_close.get("path")
        editor_to_remove = tab_data_to_close.get("editor")

        lint_timer = tab_data_to_close.get("lint_delay_timer")
        if lint_timer:
            lint_timer.stop()
            try:
                lint_timer.timeout.disconnect()
            except RuntimeError:
                pass
            lint_timer.deleteLater()

        if editor_to_remove:
            try:
                editor_to_remove.textChanged.disconnect()
            except RuntimeError:
                pass
            try:
                editor_to_remove.document().modificationChanged.disconnect()
            except RuntimeError:
                pass

        self.tab_widget.removeTab(index)

        logger.info(f"Tab {index} (Pfad: {closed_path}) entfernt.")
        self.tab_closed_signal.emit(
            closed_path
        )  # closed_path ist Optional[Path], also object

        if self.tab_widget.count() == 0:
            self.last_tab_closed_signal.emit()

        return closed_path

    def count(self) -> int:
        return self.tab_widget.count()

    def find_tab_by_path(self, file_path: Path) -> int:
        """Sucht ein Tab anhand des Dateipfads und gibt dessen Index zurück oder -1."""
        for i, tab_data in enumerate(self.open_tabs_data):
            if tab_data.get("path") == file_path:
                return i
        return -1
