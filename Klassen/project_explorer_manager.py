# Klassen/project_explorer_manager.py
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QDir, QFileInfo, QModelIndex, Qt, Slot
from PySide6.QtWidgets import (
    QDockWidget,
    QTreeView,
    QFileSystemModel,
    QFileDialog,
    QMessageBox,
)

if TYPE_CHECKING:
    from .ui.ui_main_window import MainWindow
    from .core.file_manager import FileManager

logger = logging.getLogger(__name__)


class ProjectExplorerManager:
    """
    Manages the project explorer dock widget, tree view, and file system model.
    Handles opening folders and files from the explorer.
    """

    DEFAULT_PROJECT_PATH_KEY = "default_project_path"

    def __init__(self, main_window: "MainWindow", file_manager: "FileManager"):
        self.main_window = main_window
        self.file_manager = file_manager
        self.project_root_path: Optional[Path] = None

        # UI-Elemente werden Attribute der MainWindow, aber hier initialisiert und verwaltet
        self.main_window.project_explorer_dock = QDockWidget(
            "Projekt-Explorer", self.main_window
        )
        self.main_window.project_explorer_tree = QTreeView(
            self.main_window.project_explorer_dock
        )
        self.main_window.file_system_model = QFileSystemModel(
            self.main_window.project_explorer_tree
        )

        self._setup_ui_elements()
        self._load_initial_project_path()
        logger.info("ProjectExplorerManager initialisiert.")

    def _setup_ui_elements(self):
        """Sets up the QDockWidget, QTreeView, and QFileSystemModel."""
        mw = self.main_window

        mw.file_system_model.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs | QDir.Files)
        mw.file_system_model.setRootPath(
            QDir.homePath()
        )  # Temporärer Root, wird in _load_initial_project_path gesetzt

        mw.project_explorer_tree.setModel(mw.file_system_model)

        for i in range(1, mw.file_system_model.columnCount()):
            mw.project_explorer_tree.setColumnHidden(i, True)
        mw.project_explorer_tree.setHeaderHidden(True)
        mw.project_explorer_tree.setAnimated(False)
        mw.project_explorer_tree.setSortingEnabled(True)
        mw.project_explorer_tree.sortByColumn(0, Qt.SortOrder.AscendingOrder)

        mw.project_explorer_tree.activated.connect(self.on_project_item_activated)

        mw.project_explorer_dock.setWidget(mw.project_explorer_tree)
        mw.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, mw.project_explorer_dock)

        # Create the toggle action for the view menu and assign it to MainWindow
        mw.action_toggle_project_explorer = mw.project_explorer_dock.toggleViewAction()
        mw.action_toggle_project_explorer.setText("Projekt-Explorer")
        mw.action_toggle_project_explorer.setStatusTip(
            "Zeigt den Projekt-Explorer an oder blendet ihn aus."
        )
        mw.action_toggle_project_explorer.setShortcut("Ctrl+Shift+E")
        mw.action_toggle_project_explorer.setCheckable(True)
        # Der initiale Zustand (checked/unchecked) sollte durch MenuManager oder update_button_states gesetzt werden
        # basierend auf der Sichtbarkeit des Docks, oder hier direkt:
        mw.action_toggle_project_explorer.setChecked(
            not mw.project_explorer_dock.isHidden()
        )
        mw.project_explorer_dock.visibilityChanged.connect(
            mw.action_toggle_project_explorer.setChecked
        )

    def _load_initial_project_path(self):
        """Loads and sets the initial or last used project path."""
        default_path_str = self.main_window.runtime_settings.get(
            self.DEFAULT_PROJECT_PATH_KEY, QDir.homePath()
        )
        initial_path = Path(default_path_str)

        if not initial_path.exists() or not initial_path.is_dir():
            logger.warning(
                f"ProjectExplorerManager: Gespeicherter Projektpfad '{initial_path}' "
                f"ist ungültig. Fallback auf Home-Verzeichnis."
            )
            initial_path = Path(QDir.homePath())

        self.set_root_path(initial_path)

    def set_root_path(self, path: Path):
        """Sets the root path for the file system model and tree view."""
        mw = self.main_window
        if path.exists() and path.is_dir():
            self.project_root_path = path
            mw.project_root_path = path  # Sync mit MainWindow-Attribut

            str_path = str(path)
            if mw.file_system_model and mw.project_explorer_tree:
                mw.file_system_model.setRootPath(str_path)
                root_model_index = mw.file_system_model.index(str_path)
                mw.project_explorer_tree.setRootIndex(root_model_index)
                logger.info(
                    f"ProjectExplorerManager: Wurzelpfad auf '{str_path}' gesetzt."
                )

                mw.runtime_settings[self.DEFAULT_PROJECT_PATH_KEY] = str_path
                mw._save_runtime_settings()
                mw.update_button_states()
        else:
            QMessageBox.warning(
                mw,
                "Ungültiger Pfad",
                f"Der ausgewählte Pfad '{path}' ist kein gültiges Verzeichnis.",
            )
            logger.warning(
                f"ProjectExplorerManager: Ungültiger Projektpfad versucht zu setzen: {path}"
            )

    @Slot()
    def handle_open_folder(self):
        """Handles the 'Open Folder' action by showing a directory dialog."""
        current_path_str = (
            str(self.project_root_path)
            if self.project_root_path and self.project_root_path.is_dir()
            else QDir.homePath()
        )

        dir_path_str = QFileDialog.getExistingDirectory(
            self.main_window, "Projektordner auswählen", current_path_str
        )
        if dir_path_str:
            self.set_root_path(Path(dir_path_str))

    @Slot(QModelIndex)
    def on_project_item_activated(self, index: QModelIndex):
        """Handles activation (e.g., double-click) of an item in the project explorer."""
        if (
            not index.isValid()
            or not self.file_manager
            or not self.main_window.file_system_model
        ):
            return

        file_path_str = self.main_window.file_system_model.filePath(index)
        file_info = QFileInfo(file_path_str)

        if file_info.isFile():
            logger.debug(f"ProjectExplorerManager: Datei aktiviert: {file_path_str}")
            self.file_manager._open_single_file_in_tab(Path(file_path_str))
        elif file_info.isDir() and self.main_window.project_explorer_tree:
            if self.main_window.project_explorer_tree.isExpanded(index):
                self.main_window.project_explorer_tree.collapse(index)
            else:
                self.main_window.project_explorer_tree.expand(index)
