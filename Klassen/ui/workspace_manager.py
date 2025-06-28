# Klassen/workspace_manager.py
# -*- coding: utf-8 -*-

import logging
from typing import TYPE_CHECKING, Optional

# Notwendige Qt-Importe
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QProgressBar,
    QComboBox,
    QListWidget,
    QSpacerItem,
    QSizePolicy,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

# Matplotlib-Importe
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

    MATPLOTLIB_AVAILABLE_WM = True
    logger_wm = logging.getLogger(__name__)  # Lokaler Logger für WorkspaceManager
    logger_wm.info("Matplotlib für WorkspaceManager geladen.")
except ImportError:
    MATPLOTLIB_AVAILABLE_WM = False
    FigureCanvas = None  # type: ignore
    Figure = None  # type: ignore
    logger_wm = logging.getLogger(__name__)
    logger_wm.warning("Matplotlib für WorkspaceManager nicht gefunden.")


if TYPE_CHECKING:
    from .ui_main_window import MainWindow
    from ..tab_manager import TabManager
    from .code_editor_widget import CodeEditor

    # AIServiceCoordinator wird nicht direkt benötigt, da das Popup in MainWindow erstellt
    # und dann vom AIServiceCoordinator verwendet wird.

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """
    Verwaltet die generelle UI-Einrichtung, Aktualisierung von UI-Zuständen
    und das Hauptfenster-Layout.
    """

    def __init__(self, main_window: "MainWindow"):
        self.main_window = main_window
        # TabManager wird benötigt, um z.B. den Fenstertitel korrekt zu setzen
        self.tab_manager: Optional["TabManager"] = main_window.tab_manager

        logger.info("WorkspaceManager initialisiert.")

    def setup_ui_structure(self, central_widget: QWidget):
        """
        Richtet die grundlegende UI-Struktur im zentralen Widget ein.
        Das tab_widget selbst wird in MainWindow erstellt und hier platziert.
        """
        mw = self.main_window

        # Das main_central_layout wird auf dem central_widget von MainWindow erstellt.
        if not central_widget.layout():
            mw.main_central_layout = QVBoxLayout(central_widget)
            mw.main_central_layout.setContentsMargins(
                0, 0, 0, 0
            )  # Keine äußeren Ränder
        else:
            # Falls schon ein Layout existiert (sollte nicht der Fall sein, wenn diese Methode korrekt aufgerufen wird)
            mw.main_central_layout = central_widget.layout()  # type: ignore
            logger.warning(
                "WorkspaceManager: central_widget hatte bereits ein Layout bei setup_ui_structure."
            )

        mw.ui_content_widget = QWidget()
        content_layout = QVBoxLayout(mw.ui_content_widget)
        content_layout.setContentsMargins(
            5, 5, 5, 5
        )  # Kleine innere Ränder für den Inhaltsbereich

        if hasattr(mw, "tab_widget") and mw.tab_widget:
            content_layout.addWidget(
                mw.tab_widget, stretch=2
            )  # tab_widget nimmt mehr Platz ein
        else:
            logger.error(
                "WorkspaceManager: MainWindow.tab_widget nicht gefunden beim UI-Struktur-Setup!"
            )

        mw.main_central_layout.addWidget(mw.ui_content_widget)
        logger.debug("WorkspaceManager: Grundlegende UI-Struktur eingerichtet.")

    def setup_ui_components(self):
        """
        Initialisiert die spezifischen UI-Komponenten im ui_content_widget.
        """
        mw = self.main_window
        if (
            not hasattr(mw, "ui_content_widget")
            or not mw.ui_content_widget
            or not mw.ui_content_widget.layout()
        ):
            logger.error(
                "WorkspaceManager: ui_content_widget oder dessen Layout nicht initialisiert "
                "vor setup_ui_components!"
            )
            return

        content_layout = mw.ui_content_widget.layout()  # QVBoxLayout
        if not isinstance(content_layout, QVBoxLayout):  # Zusätzliche Typprüfung
            logger.error(
                f"WorkspaceManager: Layout von ui_content_widget ist kein QVBoxLayout, sondern {type(content_layout)}."
            )
            # Erstelle ein neues QVBoxLayout, falls das existierende Layout unpassend ist
            new_content_layout = QVBoxLayout(mw.ui_content_widget)
            content_layout = new_content_layout

        self._setup_autocomplete_popup()
        self._setup_quality_label_area(content_layout)
        self._setup_controls_panel(content_layout)

        if hasattr(mw, "find_replace_widget") and mw.find_replace_widget:
            content_layout.addWidget(mw.find_replace_widget)
            mw.find_replace_widget.setVisible(False)  # Standardmäßig ausblenden

        mw.progress_bar = QProgressBar()
        mw.progress_bar.setTextVisible(False)
        content_layout.addWidget(mw.progress_bar)

        self._setup_plot_info_area(content_layout)
        self._setup_results_log_area(content_layout)

        mw.statusBar().showMessage("Bereit.")
        logger.debug("WorkspaceManager: UI-Komponenten initialisiert.")

    def _setup_autocomplete_popup(self):
        mw = self.main_window
        mw.autocomplete_popup = QListWidget(mw)  # Parent ist MainWindow
        mw.autocomplete_popup.setWindowFlags(Qt.WindowType.Popup)
        logger.debug("WorkspaceManager: Autocomplete-Popup erstellt.")

    def _setup_quality_label_area(self, parent_layout: QVBoxLayout):
        mw = self.main_window
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Qualität (akt. Tab):"))
        mw.quality_label_combo = QComboBox()
        if hasattr(mw, "QUALITY_LABELS"):
            mw.quality_label_combo.addItems(mw.QUALITY_LABELS)
        else:
            logger.warning(
                "WorkspaceManager: MainWindow.QUALITY_LABELS nicht gefunden."
            )
            mw.quality_label_combo.addItems(["neutral", "gut", "schlecht"])  # Fallback
        mw.quality_label_combo.setToolTip(
            "Setzt das Qualitätskennzeichen für die Datei im aktuellen Tab (wird beim Speichern übernommen)."
        )
        mw.quality_label_combo.setEnabled(False)
        # Das Signal currentTextChanged wird in MainWindow verbunden, da es Logik in MainWindow auslöst.
        if hasattr(mw, "on_quality_label_changed_ui"):
            mw.quality_label_combo.currentTextChanged.connect(
                mw.on_quality_label_changed_ui
            )
        quality_layout.addWidget(mw.quality_label_combo)
        quality_layout.addSpacerItem(
            QSpacerItem(
                40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
        )
        parent_layout.addLayout(quality_layout)
        logger.debug("WorkspaceManager: Qualitätslabel-Bereich eingerichtet.")

    def _setup_controls_panel(self, parent_layout: QVBoxLayout):
        mw = self.main_window
        controls_layout = QHBoxLayout()

        mw.init_db_button = QPushButton("DB Initialisieren")
        if hasattr(mw, "on_init_db_clicked"):
            mw.init_db_button.clicked.connect(mw.on_init_db_clicked)
        controls_layout.addWidget(mw.init_db_button)

        mw.load_py_files_button = QPushButton(".py Dateien Laden")
        if hasattr(mw, "on_load_py_files_clicked"):
            mw.load_py_files_button.clicked.connect(mw.on_load_py_files_clicked)
        controls_layout.addWidget(mw.load_py_files_button)

        mw.format_code_button = QPushButton("Code formatieren")
        mw.format_code_button.setToolTip("Formatiert den Code mit 'black' (Ctrl+Alt+L)")
        if hasattr(mw, "action_format_code") and mw.action_format_code:
            mw.format_code_button.clicked.connect(mw.action_format_code.trigger)
        else:
            logger.error(
                "WorkspaceManager: Geteilte Aktion 'action_format_code' nicht in MainWindow gefunden."
            )
            mw.format_code_button.setEnabled(False)
        controls_layout.addWidget(mw.format_code_button)

        mw.lint_code_button = QPushButton("Code Linting (Flake8)")
        if hasattr(mw, "on_lint_code_clicked"):
            mw.lint_code_button.clicked.connect(mw.on_lint_code_clicked)
        controls_layout.addWidget(mw.lint_code_button)

        mw.correct_code_button = QPushButton("Code Korrektur (KI)")
        if hasattr(mw, "on_correct_code_clicked"):
            mw.correct_code_button.clicked.connect(mw.on_correct_code_clicked)
        controls_layout.addWidget(mw.correct_code_button)

        mw.improve_code_button = QPushButton("KI-Verbesserung")
        mw.improve_code_button.setToolTip(
            "Markierten Codeblock oder gesamten Code (wenn nichts markiert) durch KI verbessern lassen."
        )
        if hasattr(mw, "on_improve_code_clicked"):  # Slot in MainWindow erstellen
            mw.improve_code_button.clicked.connect(mw.on_improve_code_clicked)
        else:
            logger.warning(
                "WorkspaceManager: Slot 'on_improve_code_clicked' nicht in MainWindow gefunden."
            )
            mw.improve_code_button.setEnabled(False)
        controls_layout.addWidget(mw.improve_code_button)

        hf_model_name = mw.runtime_settings.get("hf_base_model_name", "HF Model")
        hf_model_short_name = hf_model_name.split("/")[-1]
        mw.finetune_hf_button = QPushButton(f"Fine-Tune {hf_model_short_name} (QLoRA)")
        if hasattr(mw, "on_finetune_hf_clicked"):
            mw.finetune_hf_button.clicked.connect(mw.on_finetune_hf_clicked)
        controls_layout.addWidget(mw.finetune_hf_button)

        parent_layout.addLayout(controls_layout)
        logger.debug("WorkspaceManager: Kontroll-Panel eingerichtet.")

    def _setup_plot_info_area(self, parent_layout: QVBoxLayout):
        mw = self.main_window
        plot_info_area_layout = QHBoxLayout()
        if MATPLOTLIB_AVAILABLE_WM and FigureCanvas is not None and Figure is not None:
            mw.loss_plot_figure = Figure(figsize=(5, 3), dpi=100)  # Erstellt die Figure
            mw.loss_plot_canvas = FigureCanvas(
                mw.loss_plot_figure
            )  # Erstellt den Canvas mit der Figure
            plot_info_area_layout.addWidget(mw.loss_plot_canvas, stretch=1)
        else:
            no_plot_label = QLabel(
                "Matplotlib nicht gefunden.\nPlot-Anzeige deaktiviert."
            )
            no_plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            plot_info_area_layout.addWidget(no_plot_label, stretch=1)
            mw.loss_plot_canvas = None  # Sicherstellen, dass es None ist

        mw.model_status_label = QLabel("Modellstatus: Initialisiere...")
        mw.model_status_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        mw.model_status_label.setWordWrap(True)
        plot_info_area_layout.addWidget(mw.model_status_label, stretch=0)
        parent_layout.addLayout(plot_info_area_layout)
        logger.debug("WorkspaceManager: Plot- und Info-Bereich eingerichtet.")

    def _setup_results_log_area(self, parent_layout: QVBoxLayout):
        mw = self.main_window
        parent_layout.addWidget(QLabel("Analyse / Vorschläge / Fehler-Log (akt. Tab):"))
        mw.results_display_list = QListWidget()
        monospace_font_log = QFont("monospace")
        monospace_font_log.setPointSize(9)  # Etwas kleiner für mehr Inhalt
        mw.results_display_list.setFont(monospace_font_log)
        if hasattr(mw, "on_project_search_result_activated"):
            mw.results_display_list.itemActivated.connect(
                mw.on_project_search_result_activated
            )
        parent_layout.addWidget(mw.results_display_list, stretch=1)

        log_actions_layout = QHBoxLayout()
        mw.export_log_button = QPushButton("Log Exportieren (.txt)")
        if hasattr(mw, "on_export_log_clicked"):
            mw.export_log_button.clicked.connect(mw.on_export_log_clicked)
        log_actions_layout.addWidget(mw.export_log_button)

        mw.apply_corrections_button = QPushButton("Korrekturen anwenden")
        if hasattr(mw, "on_apply_corrections_clicked"):
            mw.apply_corrections_button.clicked.connect(mw.on_apply_corrections_clicked)
        mw.apply_corrections_button.setToolTip(
            "Wendet den zuletzt erfolgreich analysierten/korrigierten Code im aktuellen Tab an."
        )
        log_actions_layout.addWidget(mw.apply_corrections_button)

        mw.correction_feedback_good_button = QPushButton("👍 Gute Korrektur")
        if hasattr(mw, "on_correction_feedback_good_clicked"):
            mw.correction_feedback_good_button.clicked.connect(
                mw.on_correction_feedback_good_clicked
            )
        mw.correction_feedback_good_button.setToolTip(
            "Markiert die letzte KI-Korrektur im aktuellen Tab als gut."
        )
        log_actions_layout.addWidget(mw.correction_feedback_good_button)

        mw.correction_feedback_bad_button = QPushButton("👎 Schlechte Korrektur")
        if hasattr(mw, "on_correction_feedback_bad_clicked"):
            mw.correction_feedback_bad_button.clicked.connect(
                mw.on_correction_feedback_bad_clicked
            )
        mw.correction_feedback_bad_button.setToolTip(
            "Markiert die letzte KI-Korrektur im aktuellen Tab als schlecht."
        )
        log_actions_layout.addWidget(mw.correction_feedback_bad_button)

        log_actions_layout.addSpacerItem(
            QSpacerItem(
                40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
        )
        parent_layout.addLayout(log_actions_layout)
        logger.debug("WorkspaceManager: Ergebnis-/Log-Bereich eingerichtet.")

    def update_window_title(self):
        mw = self.main_window
        title_parts = ["Signz-Mind Code Analyzer"]

        if mw.tab_manager and hasattr(mw, "tab_widget") and mw.tab_widget:
            current_tab_index = mw.tab_widget.currentIndex()

            if current_tab_index != -1:  # Prüfe, ob ein Tab ausgewählt ist
                tab_title_text = mw.tab_widget.tabText(current_tab_index)
                title_parts.append(f"- {tab_title_text}")
            elif mw.tab_widget.count() == 0:
                title_parts.append("- Unbenannt")

        mw.setWindowTitle(" ".join(title_parts))
        # logger.debug(f"WorkspaceManager: Fenstertitel aktualisiert auf '{mw.windowTitle()}'.") # Zu häufig

    def update_button_states(self):
        mw = self.main_window

        db_ready = mw.data_manager and mw.data_manager.conn is not None
        current_tab_data = (
            mw.tab_manager.get_current_tab_data() if mw.tab_manager else None
        )
        editor: Optional["CodeEditor"] = None
        editor_has_text = False
        is_modified_doc = False
        has_selection = False
        current_file_has_path = False
        no_tabs_open = (
            mw.tab_widget.count() == 0
            if hasattr(mw, "tab_widget") and mw.tab_widget
            else True
        )

        if current_tab_data:
            editor = current_tab_data.get("editor")
            if editor:
                editor_has_text = bool(editor.toPlainText().strip())
                is_modified_doc = editor.document().isModified()
            current_file_has_path = current_tab_data.get("path") is not None

        if hasattr(mw, "load_py_files_button"):
            mw.load_py_files_button.setEnabled(db_ready)
        if hasattr(mw, "action_new"):
            mw.action_new.setEnabled(True)
        if hasattr(mw, "action_open"):
            mw.action_open.setEnabled(True)
        if hasattr(mw, "action_open_folder"):
            mw.action_open_folder.setEnabled(True)

        if hasattr(mw, "action_save"):
            mw.action_save.setEnabled(
                not no_tabs_open and current_file_has_path and is_modified_doc
            )
        if hasattr(mw, "action_save_as"):
            mw.action_save_as.setEnabled(not no_tabs_open)
        if hasattr(mw, "action_close_tab"):
            mw.action_close_tab.setEnabled(not no_tabs_open)
        if hasattr(mw, "action_find_in_project"):
            mw.action_find_in_project.setEnabled(
                mw.project_root_path is not None and mw.project_root_path.is_dir()
            )

        if hasattr(mw, "quality_label_combo"):
            mw.quality_label_combo.setEnabled(
                not no_tabs_open and current_file_has_path
            )

        data_for_training_exists = False
        if db_ready and mw.data_manager:
            try:
                snippet_quality_label = mw.runtime_settings.get(
                    "hf_snippet_training_quality_label", "gut (Training)"
                )
                positive_feedback_types = mw.runtime_settings.get(
                    "hf_positive_feedback_types",
                    ["suggestion_used", "correction_applied", "correction_good"],
                )
                train_with_snippets = mw.runtime_settings.get(
                    "hf_train_with_snippets", True
                )
                train_with_feedback = mw.runtime_settings.get(
                    "hf_train_with_feedback", True
                )

                snippets_text = ""
                if train_with_snippets:
                    snippets_text = (
                        mw.data_manager.get_all_snippets_text(
                            quality_filter=[snippet_quality_label]
                        )
                        or ""
                    )

                feedback_list = []
                if train_with_feedback:
                    feedback_list = mw.data_manager.get_feedback_data_for_training(
                        feedback_types=positive_feedback_types
                    )

                if (snippets_text.strip() and train_with_snippets) or (
                    feedback_list and train_with_feedback
                ):
                    data_for_training_exists = True
            except Exception as e:
                logger.warning(
                    f"Fehler beim Prüfen der Trainingsdaten für Button-Status: {e}",
                    exc_info=False,
                )

        can_finetune_hf = (
            data_for_training_exists
            and (
                hasattr(mw, "HF_LIBRARIES_AVAILABLE_FLAG")
                and mw.HF_LIBRARIES_AVAILABLE_FLAG
            )
            and mw.hf_fine_tuner is not None
            and (
                hasattr(mw, "FineTuneThread_CLASS_REF")
                and mw.FineTuneThread_CLASS_REF is not None
            )  # Prüft Existenz der Klasse
        )

        if hasattr(mw, "finetune_hf_button"):
            mw.finetune_hf_button.setEnabled(can_finetune_hf)

        if hasattr(mw, "results_display_list") and hasattr(mw, "export_log_button"):
            mw.export_log_button.setEnabled(mw.results_display_list.count() > 0)

        if hasattr(mw, "lint_code_button"):
            mw.lint_code_button.setEnabled(
                mw.linter_service is not None and editor_has_text
            )

        if hasattr(mw, "correct_code_button"):
            mw.correct_code_button.setEnabled(
                mw.code_corrector is not None and editor_has_text
            )

        can_format = (
            hasattr(mw, "BLACK_AVAILABLE_FLAG") and mw.BLACK_AVAILABLE_FLAG
        ) and editor_has_text
        if hasattr(mw, "action_format_code") and mw.action_format_code:
            mw.action_format_code.setEnabled(can_format)

        if (
            hasattr(mw, "format_code_button")
            and mw.format_code_button
            and hasattr(mw, "action_format_code")
            and mw.action_format_code
        ):
            mw.format_code_button.setEnabled(mw.action_format_code.isEnabled())

        if hasattr(mw, "improve_code_button"):
            # Button ist aktiv, wenn Text im Editor ist ODER wenn Text markiert ist.
            mw.improve_code_button.setEnabled(editor_has_text or has_selection)

        can_give_correction_feedback = False
        can_apply_correction = False
        if current_tab_data:
            can_apply_correction = (
                current_tab_data.get("last_suggested_code") is not None
            )
            can_give_correction_feedback = (
                can_apply_correction
                and not current_tab_data.get(
                    "feedback_given_for_last_correction", False
                )
            )

        if hasattr(mw, "apply_corrections_button"):
            mw.apply_corrections_button.setEnabled(can_apply_correction)
        if hasattr(mw, "correction_feedback_good_button"):
            mw.correction_feedback_good_button.setEnabled(can_give_correction_feedback)
        if hasattr(mw, "correction_feedback_bad_button"):
            mw.correction_feedback_bad_button.setEnabled(can_give_correction_feedback)

        if (
            mw.search_controller
            and hasattr(mw, "find_replace_widget")
            and mw.find_replace_widget
            and mw.find_replace_widget.isVisible()
        ):
            mw.search_controller._update_find_buttons_state()
        # logger.debug("WorkspaceManager: Button-Zustände aktualisiert.") # Zu häufig

    def set_ui_for_long_operation(self, enabled: bool):
        mw = self.main_window
        if hasattr(mw, "init_db_button"):
            mw.init_db_button.setEnabled(enabled)

        if enabled:
            self.update_button_states()
        else:
            if hasattr(mw, "load_py_files_button"):
                mw.load_py_files_button.setEnabled(False)
            if hasattr(mw, "finetune_hf_button"):
                mw.finetune_hf_button.setEnabled(False)
            if hasattr(mw, "lint_code_button"):
                mw.lint_code_button.setEnabled(False)
            if hasattr(mw, "correct_code_button"):
                mw.correct_code_button.setEnabled(False)
            if hasattr(mw, "improve_code_button"):
                mw.improve_code_button.setEnabled(False)
            if hasattr(mw, "apply_corrections_button"):
                mw.apply_corrections_button.setEnabled(False)
            if hasattr(mw, "correction_feedback_good_button"):
                mw.correction_feedback_good_button.setEnabled(False)
            if hasattr(mw, "correction_feedback_bad_button"):
                mw.correction_feedback_bad_button.setEnabled(False)

            if hasattr(mw, "action_new"):
                mw.action_new.setEnabled(False)
            if hasattr(mw, "action_open"):
                mw.action_open.setEnabled(False)
            if hasattr(mw, "action_open_folder"):
                mw.action_open_folder.setEnabled(False)
            if hasattr(mw, "action_save"):
                mw.action_save.setEnabled(False)
            if hasattr(mw, "action_save_as"):
                mw.action_save_as.setEnabled(False)
            if hasattr(mw, "action_close_tab"):
                mw.action_close_tab.setEnabled(False)
            if hasattr(mw, "action_find_replace"):
                mw.action_find_replace.setEnabled(False)
            if hasattr(mw, "action_find_in_project"):
                mw.action_find_in_project.setEnabled(False)

            if hasattr(mw, "quality_label_combo"):
                mw.quality_label_combo.setEnabled(False)

            if (
                mw.search_controller
                and hasattr(mw, "find_replace_widget")
                and mw.find_replace_widget
            ):
                mw.find_replace_widget.setVisible(False)
        logger.debug(
            f"WorkspaceManager: UI für lange Operation auf enabled={enabled} gesetzt."
        )

    def handle_training_progress(self, value: int):
        mw = self.main_window
        if not hasattr(mw, "progress_bar") or not mw.progress_bar:
            return

        if value == -1:
            mw.progress_bar.setRange(0, 100)
            mw.progress_bar.setValue(0)
            mw.progress_bar.setFormat("Fehler")
            mw.progress_bar.setTextVisible(True)
        elif value == 0 and mw.progress_bar.maximum() == 0:
            mw.progress_bar.setRange(0, 0)
            mw.progress_bar.setTextVisible(False)
        elif 0 <= value <= 100:
            if mw.progress_bar.minimum() == 0 and mw.progress_bar.maximum() == 0:
                mw.progress_bar.setRange(0, 100)
            mw.progress_bar.setValue(value)
            mw.progress_bar.setFormat("%p%")
            mw.progress_bar.setTextVisible(True)
        logger.debug(
            f"WorkspaceManager: Trainingsfortschritt auf {value}% aktualisiert."
        )

    def update_model_info_label(self, status_text: str):
        mw = self.main_window
        if hasattr(mw, "model_status_label") and mw.model_status_label:
            mw.model_status_label.setText(
                f"Modellstatus: {status_text if status_text else 'N/A'}"
            )
        logger.debug(
            f"WorkspaceManager: Modell-Info-Label aktualisiert: '{status_text}'."
        )
