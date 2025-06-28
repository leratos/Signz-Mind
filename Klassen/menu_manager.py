# Klassen/menu_manager.py
# -*- coding: utf-8 -*-

import logging
from typing import TYPE_CHECKING

from PySide6.QtGui import QAction, QKeySequence

if TYPE_CHECKING:
    from .ui.ui_main_window import MainWindow

logger = logging.getLogger(__name__)


class MenuManager:
    """
    Verwaltet die Erstellung von Menüaktionen und der Menüleiste für die MainWindow.
    """

    def __init__(self, main_window: "MainWindow"):
        self.main_window = main_window
        logger.info("MenuManager initialisiert.")

        self._create_actions()
        self._create_menus()

    def _create_actions(self):
        """Erstellt alle QAction-Objekte für die Anwendung."""
        mw = self.main_window

        mw.action_new = QAction("&Neu", mw)
        mw.action_new.setShortcut(QKeySequence.StandardKey.New)
        mw.action_new.setStatusTip(
            "Erstellt eine neue, leere Datei in einem neuen Tab."
        )
        mw.action_new.triggered.connect(mw.on_file_new)

        mw.action_open = QAction("&Öffnen...", mw)
        mw.action_open.setShortcut(QKeySequence.StandardKey.Open)
        mw.action_open.setStatusTip(
            "Öffnet eine vorhandene Python-Datei in einem neuen Tab."
        )
        mw.action_open.triggered.connect(mw.on_file_open)

        mw.action_open_folder = QAction("Ordner ö&ffnen...", mw)
        mw.action_open_folder.setStatusTip("Öffnet einen Ordner im Projekt-Explorer.")
        mw.action_open_folder.triggered.connect(mw.on_open_folder_clicked)

        mw.action_save = QAction("&Speichern", mw)
        mw.action_save.setShortcut(QKeySequence.StandardKey.Save)
        mw.action_save.setStatusTip("Speichert die Datei im aktuellen Tab.")
        mw.action_save.triggered.connect(mw.on_file_save)

        mw.action_save_as = QAction("Speichern &unter...", mw)
        mw.action_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        mw.action_save_as.setStatusTip(
            "Speichert die Datei im aktuellen Tab unter einem neuen Namen."
        )
        mw.action_save_as.triggered.connect(mw.on_file_save_as)

        mw.action_close_tab = QAction("Tab schließen", mw)
        mw.action_close_tab.setShortcut(QKeySequence("Ctrl+W"))
        mw.action_close_tab.setStatusTip("Schließt das aktuelle Tab.")
        mw.action_close_tab.triggered.connect(mw.on_close_current_tab_action)

        mw.action_exit = QAction("B&eenden", mw)
        mw.action_exit.setShortcut(QKeySequence.StandardKey.Quit)
        mw.action_exit.setStatusTip("Beendet die Anwendung.")
        mw.action_exit.triggered.connect(mw.close)

        # Bearbeiten-Menü Aktionen
        mw.action_find_replace = QAction("Suchen/Ersetzen...", mw)
        mw.action_find_replace.setShortcut(QKeySequence.StandardKey.Find)
        mw.action_find_replace.setStatusTip("Öffnet die Suchen/Ersetzen-Leiste.")
        mw.action_find_replace.triggered.connect(
            mw.show_find_replace_bar
        )  # Slot in MainWindow

        mw.action_find_in_project = QAction("Im Projekt suchen...", mw)
        mw.action_find_in_project.setShortcut(QKeySequence("Ctrl+Shift+F"))
        mw.action_find_in_project.setStatusTip(
            "Durchsucht alle Dateien im aktuellen Projektordner."
        )
        mw.action_find_in_project.triggered.connect(
            mw.on_find_in_project_triggered
        )  # Slot in MainWindow

        mw.action_settings = QAction("&Einstellungen...", mw)
        mw.action_settings.setStatusTip("Öffnet den Einstellungsdialog.")
        mw.action_settings.triggered.connect(mw.on_open_settings)

        mw.action_sync_with_server = QAction("Daten &synchronisieren", mw)
        mw.action_sync_with_server.setStatusTip(
            "Exportiert neue Daten und sendet sie an den zentralen Server."
        )
        mw.action_sync_with_server.triggered.connect(mw.on_sync_with_server_clicked)
        if (
            not hasattr(mw, "server_connector") or not mw.server_connector
        ):  # Deaktivieren, falls nicht verfügbar
            mw.action_sync_with_server.setEnabled(False)

        mw.action_check_model_update = QAction("&Modell aktualisieren...", mw)
        mw.action_check_model_update.setStatusTip(
            "Prüft auf neue KI-Modellversionen auf dem Server und lädt diese ggf. herunter."
        )
        if hasattr(mw, "on_check_for_model_update_clicked"):
            mw.action_check_model_update.triggered.connect(
                mw.on_check_for_model_update_clicked
            )
        else:
            logger.error(
                "MenuManager: Slot 'on_check_for_model_update_clicked' nicht in MainWindow gefunden!"
            )
            mw.action_check_model_update.setEnabled(False)
        # Deaktivieren, falls ServerConnector nicht initialisiert wurde
        if not hasattr(mw, "server_connector") or not mw.server_connector:
            mw.action_check_model_update.setEnabled(False)

        if (
            hasattr(mw, "SettingsDialog") and mw.SettingsDialog is None
        ):  # Prüfe, ob SettingsDialog Klasse existiert
            mw.action_settings.setEnabled(False)

        if hasattr(mw, "on_sync_with_server_clicked"):
            mw.action_sync_with_server.triggered.connect(mw.on_sync_with_server_clicked)
        else:
            logger.error(
                "MenuManager: Slot 'on_sync_with_server_clicked' nicht in MainWindow gefunden!"
            )
            mw.action_sync_with_server.setEnabled(False)

        if (
            hasattr(mw, "project_explorer_dock")
            and mw.project_explorer_dock
            and hasattr(mw, "action_toggle_project_explorer")
            and mw.action_toggle_project_explorer
        ):
            # Die action_toggle_project_explorer wird vom ProjectExplorerManager erstellt und mw zugewiesen.
            # Hier wird sie nur referenziert.
            pass
        else:
            logger.warning(
                "MenuManager: action_toggle_project_explorer nicht in MainWindow gefunden oder Dock nicht initialisiert."
            )
            # Erstelle eine Dummy-Aktion, um Abstürze zu vermeiden, aber sie wird nicht funktionieren.
            mw.action_toggle_project_explorer = QAction("Projekt-Explorer (Fehler)", mw)
            mw.action_toggle_project_explorer.setEnabled(False)

        logger.debug("MenuManager: Aktionen erstellt und MainWindow zugewiesen.")

    def _create_menus(self):
        """Erstellt die Menüleiste und fügt die Aktionen hinzu."""
        mw = self.main_window
        menu_bar = mw.menuBar()

        file_menu = menu_bar.addMenu("&Datei")
        file_menu.addAction(mw.action_new)
        file_menu.addAction(mw.action_open)
        file_menu.addAction(mw.action_open_folder)
        file_menu.addSeparator()
        if mw.action_sync_with_server:
            file_menu.addAction(mw.action_sync_with_server)
            file_menu.addSeparator()
        if hasattr(mw, "action_check_model_update") and mw.action_check_model_update:
            file_menu.addAction(mw.action_check_model_update)
        file_menu.addAction(mw.action_save)
        file_menu.addAction(mw.action_save_as)
        file_menu.addSeparator()
        file_menu.addAction(mw.action_close_tab)
        file_menu.addSeparator()
        file_menu.addAction(mw.action_exit)

        edit_menu = menu_bar.addMenu("&Bearbeiten")
        edit_menu.addAction(mw.action_find_replace)
        edit_menu.addAction(mw.action_find_in_project)
        edit_menu.addSeparator()
        if hasattr(mw, "action_format_code") and mw.action_format_code:
            edit_menu.addAction(mw.action_format_code)
        edit_menu.addSeparator()
        edit_menu.addAction(mw.action_settings)

        view_menu = menu_bar.addMenu("&Ansicht")
        if (
            hasattr(mw, "action_toggle_project_explorer")
            and mw.action_toggle_project_explorer
        ):
            view_menu.addAction(mw.action_toggle_project_explorer)
        else:
            logger.warning(
                "MenuManager: action_toggle_project_explorer konnte dem Ansicht-Menü nicht hinzugefügt werden."
            )

        logger.debug("MenuManager: Menüs erstellt.")
