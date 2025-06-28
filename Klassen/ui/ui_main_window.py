# Klassen/ui_main_window.py
# -*- coding: utf-8 -*-

# --- Standard Imports ---
import sys
import os
import logging
from pathlib import Path
from datetime import datetime  # Für Log-Export Dateinamen
import json
from typing import Optional, Tuple, Dict, Any  # Dict hinzugefügt
from PySide6.QtCore import (
    Slot,
    QDir,
    Qt,
    QEvent,
    Signal,
    QTimer,
)

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QComboBox,
    QFileDialog,  # Wird von Filemanager verwendet
    QMessageBox,
    QDialog,  # Für SettingsDialog
    QTabWidget,
    QTreeView,  # Wird von ProjectExplorerManager verwendet
    QFileSystemModel,  # Wird von ProjectExplorerManager verwendet
    QDockWidget,  # Wird von ProjectExplorerManager verwendet
)
from PySide6.QtGui import (
    QAction,
    QKeySequence,
)  # QKeyEvent nicht mehr direkt hier, da EventFilter in AIServiceCoordinator

# --- Matplotlib Backend Setting ---
import matplotlib
from ..core import config as global_config

logger = logging.getLogger(__name__)  # Logger für MainWindow

try:
    matplotlib.use("QtAgg")
    logger.info("Matplotlib backend set to QtAgg.")
except ImportError:
    logger.warning("Could not set Matplotlib backend to QtAgg.")
    pass
except Exception as e:
    logger.warning(f"Error setting Matplotlib backend: {e}")
    pass

try:
    from matplotlib.figure import Figure  # Für Typ-Hinweis
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
    )  # Für Typ-Hinweis
except ImportError:
    Figure = None  # type: ignore
    FigureCanvas = None  # type: ignore


# --- Projektinterne Imports ---

try:
    from ..core.python_highlighter import PythonHighlighter
except ImportError as e:
    logger.error(f"ERROR importing PythonHighlighter: {e}", exc_info=True)
    PythonHighlighter = None
try:
    from .code_editor_widget import CodeEditor
except ImportError as e:
    logger.error(f"ERROR importing CodeEditor: {e}", exc_info=True)
    CodeEditor = None
try:
    from ..tab_manager import TabManager
except ImportError as e:
    logger.error(f"ERROR importing TabManager: {e}", exc_info=True)
    TabManager = None
try:
    from ..core.file_manager import FileManager
except ImportError as e:
    logger.error(f"ERROR importing FileManager: {e}", exc_info=True)
    FileManager = None
FineTuneThread_imported_class_module_level = None
try:
    from ..ui_threads import FineTuneThread as FineTuneThread

    FineTuneThread_imported_class_module_level = FineTuneThread
    if FineTuneThread_imported_class_module_level:
        logger.info("FineTuneThread class successfully imported at module level.")
    else:
        logger.error(
            "FineTuneThread is None after import without ImportError at module level."
        )
except ImportError as e:
    logger.error(f"ERROR importing FineTuneThread at module level: {e}", exc_info=True)
try:
    from ..menu_manager import MenuManager
except ImportError as e:
    logger.error(f"ERROR importing MenuManager: {e}", exc_info=True)
    MenuManager = None
try:
    from ..project_explorer_manager import ProjectExplorerManager
except ImportError as e:
    logger.error(f"ERROR importing ProjectExplorerManager: {e}", exc_info=True)
    ProjectExplorerManager = None
try:
    from ..search_controller import SearchController
except ImportError as e:
    logger.error(f"ERROR importing SearchController: {e}", exc_info=True)
    SearchController = None
try:
    from ..core.ai_service_coordinator import AIServiceCoordinator
except ImportError as e:
    logger.error(f"ERROR importing AIServiceCoordinator: {e}", exc_info=True)
    AIServiceCoordinator = None
try:
    from ..data.data_manager import DataManager
except ImportError as e:
    logger.error(f"ERROR importing DataManager: {e}", exc_info=True)
    DataManager = None
try:
    from ..core.inference_service import (
        InferenceService,
        HF_LIBRARIES_AVAILABLE as HF_LIBS_INF_AVAILABLE,
    )
except ImportError as e:
    logger.error(f"ERROR importing InferenceService: {e}", exc_info=True)
    InferenceService = None
    HF_LIBS_INF_AVAILABLE = False
try:
    from ..core.code_corrector import CodeCorrector
except ImportError as e:
    logger.error(f"ERROR importing CodeCorrector: {e}", exc_info=True)
    CodeCorrector = None
try:
    from ..services.hf_fine_tuner import (
        HFFineTuner,
        HF_LIBRARIES_AVAILABLE as HF_LIBS_TUNER_AVAILABLE,
    )
except ImportError as e:
    logger.error(f"ERROR importing HFFineTuner: {e}", exc_info=True)
    HFFineTuner = None
    HF_LIBS_TUNER_AVAILABLE = False  # type: ignore
try:
    from ..core.linter_service import (
        LinterService,
    )  # LintError wird hier nicht direkt verwendet, aber LinterService
except ImportError as e:
    logger.error(f"ERROR importing LinterService: {e}", exc_info=True)
    LinterService = None
    # LintError = type('LintError', (dict,), {}) # Nicht mehr nötig, da LintError nicht direkt verwendet wird

try:
    from .settings_dialog import SettingsDialog  # Wird für on_open_settings benötigt
except ImportError as e:
    logger.error(f"ERROR importing SettingsDialog: {e}", exc_info=True)
    SettingsDialog = None

# NEU: Importiere WorkspaceManager
try:
    from .workspace_manager import WorkspaceManager
except ImportError as e:
    logger.error(f"ERROR importing WorkspaceManager: {e}", exc_info=True)
    WorkspaceManager = None
try:
    from ..core.code_context_analyzer import CodeContextAnalyzer
except ImportError as e:
    logger.error(f"ERROR importing CodeContextAnalyzer: {e}", exc_info=True)
    CodeContextAnalyzer = None
try:
    from ..server_connector import ServerConnector  # NEU
except ImportError as e:
    logger.error(f"ERROR importing ServerConnector: {e}", exc_info=True)
    ServerConnector = None  # type: ignore
try:
    from ..core.formatting_service import FormattingService, BLACK_AVAILABLE
except ImportError as e:
    logger.error(f"ERROR importing FormattingService: {e}", exc_info=True)
    FormattingService = None
    BLACK_AVAILABLE = False


HF_LIBRARIES_AVAILABLE = HF_LIBS_INF_AVAILABLE and HF_LIBS_TUNER_AVAILABLE


MATPLOTLIB_AVAILABLE = (
    True  # Wird vom WorkspaceManager geprüft, hier nur als Flag für Logik
)
if WorkspaceManager:  # Prüfe, ob WorkspaceManager importiert wurde
    try:
        # Versuche, die Konstante aus WorkspaceManager zu importieren, falls sie dort definiert ist
        from .workspace_manager import MATPLOTLIB_AVAILABLE_WM

        MATPLOTLIB_AVAILABLE = MATPLOTLIB_AVAILABLE_WM
    except ImportError:
        logger.warning(
            "Konnte MATPLOTLIB_AVAILABLE_WM nicht aus WorkspaceManager importieren."
        )
        # Fallback, falls der Import fehlschlägt oder die Konstante dort nicht existiert
        try:
            import matplotlib.figure  # type: ignore
            import matplotlib.backends.backend_qtagg  # type: ignore
        except ImportError:
            MATPLOTLIB_AVAILABLE = False


try:
    from accelerate import Accelerator

    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    Accelerator = None  # type: ignore


class MainWindow(QMainWindow):
    CONFIG_FILE = "app_settings.json"
    modelStatusChanged = Signal(str)  # Signal für Modellstatus-Änderungen
    LINTING_DELAY_MS = 1000
    QUALITY_LABELS = ["neutral", "gut (Training)", "schlecht (Ignorieren)", "in Arbeit"]
    DEFAULT_PROJECT_PATH_KEY = (
        "default_project_path"  # Wird von ProjectExplorerManager verwendet
    )

    EXCLUDE_DIR_PATTERNS_PROJECT_SEARCH = [
        ".git",
        "__pycache__",
        ".venv*",
        "env*",
        "Backup",
        "node_modules",
        ".vs*",
        ".idea",
        "build",
        "dist",
        ".pytest_cache",
        ".mypy_cache",
        "*.egg-info",
    ]
    INCLUDE_EXTENSIONS_PROJECT_SEARCH = {".py"}

    def __init__(self):
        super().__init__()
        logger.info("MainWindow Initialisierung gestartet.")
        self.resize(1200, 850)
        self.output_dir = Path("Ausgabe")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Attribute für Manager und Services ---
        self.runtime_settings = self._load_runtime_settings()
        self.accelerator: Optional["Accelerator"] = None
        self.data_manager: Optional[DataManager] = None
        self.inference_service: Optional[InferenceService] = None
        self.code_corrector: Optional[CodeCorrector] = None
        self.hf_fine_tuner: Optional[HFFineTuner] = None
        self.linter_service: Optional[LinterService] = None
        self.formatting_service: Optional[FormattingService] = None
        self.fine_tune_thread: Optional[FineTuneThread] = None  # Typ FineTuneThread
        self.server_connector: Optional[ServerConnector] = None

        self.tab_manager: Optional[TabManager] = None
        self.file_manager: Optional[FileManager] = None
        self.project_explorer_manager: Optional[ProjectExplorerManager] = None
        self.search_controller: Optional[SearchController] = None
        self.ai_service_coordinator: Optional[AIServiceCoordinator] = None
        self.menu_manager: Optional[MenuManager] = None
        self.workspace_manager: Optional[WorkspaceManager] = None  # NEU
        self.code_context_analyzer: Optional[CodeContextAnalyzer] = None

        self.HF_LIBRARIES_AVAILABLE_FLAG = HF_LIBRARIES_AVAILABLE
        self.FineTuneThread_CLASS_REF = FineTuneThread_imported_class_module_level
        self.BLACK_AVAILABLE_FLAG = BLACK_AVAILABLE

        # --- UI-Elemente, die von Managern erstellt und hier referenziert werden ---
        self.project_root_path: Optional[Path] = None

        self.project_explorer_dock: Optional[QDockWidget] = None
        self.project_explorer_tree: Optional[QTreeView] = None
        self.file_system_model: Optional[QFileSystemModel] = None
        self.action_toggle_project_explorer: Optional[QAction] = None
        self.action_sync_with_server: Optional[QAction] = None
        self.action_format_code: Optional[QAction] = None
        self.find_replace_widget: Optional[QWidget] = None
        from PySide6.QtWidgets import QLineEdit, QCheckBox  # Für Typ-Hinweise

        self.find_input: Optional[QLineEdit] = None
        self.replace_input: Optional[QLineEdit] = None

        self.find_replace_widget: Optional[QWidget] = None
        # Explizite Typen für UI-Elemente, die von anderen Managern erstellt werden,
        # aber in MainWindow referenziert werden könnten (z.B. für Fallbacks oder direkte Zugriffe, obwohl selten).
        from PySide6.QtWidgets import (
            QLineEdit,
            QPushButton,
        )  # Für Typ-Hinweise

        self.find_input: Optional[QLineEdit] = None
        self.replace_input: Optional[QLineEdit] = None
        self.find_next_button: Optional[QPushButton] = None
        self.find_prev_button: Optional[QPushButton] = None
        self.replace_button: Optional[QPushButton] = None
        self.replace_all_button: Optional[QPushButton] = None
        self.case_sensitive_checkbox: Optional[QCheckBox] = None
        self.whole_word_checkbox: Optional[QCheckBox] = None

        self.main_central_layout: Optional[QVBoxLayout] = None
        self.ui_content_widget: Optional[QWidget] = None
        self.tab_widget: Optional[QTabWidget] = None
        self.autocomplete_popup: Optional[QListWidget] = None
        self.quality_label_combo: Optional[QComboBox] = None
        self.progress_bar: Optional[QProgressBar] = None
        self.loss_plot_canvas: Optional[FigureCanvas] = None
        self.loss_plot_figure: Optional[Figure] = None
        self.model_status_label: Optional[QLabel] = None
        self.results_display_list: Optional[QListWidget] = None
        self.init_db_button: Optional[QPushButton] = None
        self.load_py_files_button: Optional[QPushButton] = None
        self.lint_code_button: Optional[QPushButton] = None
        self.correct_code_button: Optional[QPushButton] = None
        self.improve_code_button: Optional[QPushButton] = None
        self.finetune_hf_button: Optional[QPushButton] = None
        self.export_log_button: Optional[QPushButton] = None
        self.apply_corrections_button: Optional[QPushButton] = None
        self.correction_feedback_good_button: Optional[QPushButton] = None
        self.correction_feedback_bad_button: Optional[QPushButton] = None
        self.format_code_button: Optional[QPushButton] = None

        # --- Reihenfolge der Initialisierung ---
        self._initialize_accelerator()
        self._initialize_services()
        self._create_shared_actions()

        critical_managers = {
            "CodeEditor": CodeEditor,
            "TabManager": TabManager,
            "FileManager": FileManager,
            "MenuManager": MenuManager,
            "ProjectExplorerManager": ProjectExplorerManager,
            "SearchController": SearchController,
            "AIServiceCoordinator": AIServiceCoordinator,
            "WorkspaceManager": WorkspaceManager,
            "CodeContextAnalyzer": CodeContextAnalyzer,
        }
        missing_managers_list = [
            name for name, cls in critical_managers.items() if cls is None
        ]
        if missing_managers_list:
            QMessageBox.critical(
                self,
                "Fatal Error",
                f"Kritische Manager-Klassen nicht geladen: {', '.join(missing_managers_list)}. Anwendung kann nicht starten.",
            )
            logger.critical(
                f"Kritische Manager-Klassen nicht geladen: {', '.join(missing_managers_list)}. Anwendung wird beendet."
            )
            sys.exit(1)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.on_tab_close_requested)

        if (
            TabManager
            and DataManager
            and CodeEditor
            and self.tab_widget
            and self.data_manager
        ):
            self.tab_manager = TabManager(
                tab_widget=self.tab_widget,
                data_manager=self.data_manager,
                code_editor_class=CodeEditor,
                python_highlighter_class=PythonHighlighter,
                linting_delay_ms=self.LINTING_DELAY_MS,
                quality_labels=self.QUALITY_LABELS,
                main_window_ref=self,
            )
        else:
            logger.critical(
                "TabManager oder seine Abhängigkeiten konnten nicht initialisiert werden."
            )
            sys.exit(1)  # Kritischer Fehler

        # WorkspaceManager initialisieren und UI aufbauen LASSEN (ERSTELLT autocomplete_popup)
        if WorkspaceManager and self.tab_widget:
            self.workspace_manager = WorkspaceManager(self)
            self.workspace_manager.setup_ui_structure(central_widget)
            self.workspace_manager.setup_ui_components()  # Dies erstellt mw.autocomplete_popup
        else:
            logger.critical("WorkspaceManager konnte nicht initialisiert werden.")
            sys.exit(1)  # Kritischer Fehler

        if FileManager and self.tab_manager and self.data_manager:
            self.file_manager = FileManager(
                main_window=self,
                tab_manager=self.tab_manager,
                data_manager=self.data_manager,
            )
        if ProjectExplorerManager and self.file_manager:
            self.project_explorer_manager = ProjectExplorerManager(
                main_window=self, file_manager=self.file_manager
            )
        if CodeContextAnalyzer:
            self.code_context_analyzer = CodeContextAnalyzer(self.project_root_path)
        else:
            logger.error("CodeContextAnalyzer konnte nicht geladen werden.")
        if SearchController and self.tab_manager and self.file_manager:
            self.search_controller = SearchController(
                main_window=self,
                tab_manager=self.tab_manager,
                file_manager=self.file_manager,
            )

        # AIServiceCoordinator initialisieren (NACHDEM WorkspaceManager das autocomplete_popup erstellt hat)
        if AIServiceCoordinator and self.tab_manager and self.data_manager:
            self.ai_service_coordinator = AIServiceCoordinator(
                main_window=self,
                tab_manager=self.tab_manager,
                data_manager=self.data_manager,
                inference_service=self.inference_service,
                code_corrector=self.code_corrector,
                linter_service=self.linter_service,
                code_context_analyzer=self.code_context_analyzer,
                formatting_service=self.formatting_service,
            )
        else:
            logger.critical(
                "AIServiceCoordinator oder seine Abhängigkeiten konnten nicht initialisiert werden."
            )
            # Fehlerbehandlung oder sys.exit()

        self._connect_tab_manager_signals()

        if MenuManager:
            self.menu_manager = MenuManager(self)

        if self.workspace_manager:
            self.modelStatusChanged.connect(
                self.workspace_manager.update_model_info_label
            )

        if self.inference_service and hasattr(self.inference_service, "get_status"):
            self.modelStatusChanged.emit(self.inference_service.get_status())
        else:
            self.modelStatusChanged.emit(
                "HF Model: N/A (Bibliotheken fehlen oder Ladefehler)"
            )

        if self.workspace_manager:
            self.workspace_manager.update_window_title()
            self.workspace_manager.update_button_states()
        else:
            self.update_window_title_fallback()
            self.update_button_states_fallback()

        if self.tab_manager and self.tab_manager.count() == 0:
            self.tab_manager.add_new_tab()
        elif self.tab_manager:
            current_editor = self.tab_manager.get_current_editor()
            if current_editor and self.ai_service_coordinator:
                self.ai_service_coordinator.install_event_filter_on_editor(
                    current_editor
                )

        if hasattr(self, "results_display_list") and self.results_display_list:
            self.results_display_list.itemActivated.connect(
                self.on_project_search_result_activated
            )

        logger.info("MainWindow Initialisierung abgeschlossen.")

    # --- Methoden, die an WorkspaceManager delegiert werden ---
    def _create_shared_actions(self):
        """Erstellt QAction-Objekte, die von mehreren UI-Komponenten benötigt werden."""
        self.action_format_code = QAction("Code formatieren (black)", self)
        self.action_format_code.setShortcut(QKeySequence("Ctrl+Alt+L"))
        self.action_format_code.setStatusTip(
            "Formatiert den Code im aktuellen Tab mit 'black'."
        )
        self.action_format_code.triggered.connect(self.on_format_code_clicked)
        logger.debug("Gemeinsam genutzte Aktionen (z.B. Formatieren) erstellt.")

    def update_window_title(self):
        if self.workspace_manager:
            self.workspace_manager.update_window_title()
        else:
            self.update_window_title_fallback()

    def update_button_states(self):
        if self.workspace_manager:
            self.workspace_manager.update_button_states()
        else:
            self.update_button_states_fallback()

    def set_buttons_enabled(self, enabled: bool):
        if self.workspace_manager:
            self.workspace_manager.set_ui_for_long_operation(enabled)
        else:
            logger.warning("set_buttons_enabled: WorkspaceManager nicht verfügbar.")
            if hasattr(self, "finetune_hf_button") and self.finetune_hf_button:
                self.finetune_hf_button.setEnabled(enabled)
            if hasattr(self, "load_py_files_button") and self.load_py_files_button:
                self.load_py_files_button.setEnabled(enabled)

    def handle_training_progress(self, value: int):
        if self.workspace_manager:
            self.workspace_manager.handle_training_progress(value)
        elif hasattr(self, "progress_bar") and self.progress_bar:
            self.progress_bar.setValue(value if 0 <= value <= 100 else 0)
            self.progress_bar.setTextVisible(value >= 0)

    # --- Fallback-Methoden, falls WorkspaceManager nicht geladen werden konnte ---
    def update_window_title_fallback(self):
        logger.warning("update_window_title_fallback aufgerufen.")
        self.setWindowTitle("Signz-Mind Code Analyzer (Fallback UI)")

    def update_button_states_fallback(self):
        logger.warning("update_button_states_fallback aufgerufen.")
        if hasattr(self, "action_save") and self.action_save:
            self.action_save.setEnabled(
                self.tab_manager.count() > 0 if self.tab_manager else False
            )
        if hasattr(self, "finetune_hf_button") and self.finetune_hf_button:
            self.finetune_hf_button.setEnabled(False)

    def _connect_tab_manager_signals(self):
        if not self.tab_manager:
            return
        self.tab_manager.current_tab_changed_signal.connect(
            self.on_tab_manager_current_tab_changed
        )
        self.tab_manager.tab_added_signal.connect(self.on_tab_manager_tab_added)
        self.tab_manager.tab_closed_signal.connect(self.on_tab_manager_tab_closed)
        self.tab_manager.last_tab_closed_signal.connect(
            self.on_tab_manager_last_tab_closed
        )
        self.tab_manager.tab_modification_changed_signal.connect(
            self.on_tab_manager_tab_modification_changed
        )
        if (
            self.ai_service_coordinator
        ):  # Sicherstellen, dass ai_service_coordinator existiert
            self.tab_manager.lint_request_for_editor_signal.connect(
                self.ai_service_coordinator.handle_lint_request
            )
        else:
            logger.error(
                "AIServiceCoordinator nicht initialisiert, kann lint_request_for_editor_signal nicht verbinden."
            )
        self.tab_manager.tab_title_updated_signal.connect(
            self.on_tab_manager_tab_title_updated
        )

    def _get_current_tab_data(self) -> Optional[Dict[str, Any]]:
        if hasattr(self, "tab_manager") and self.tab_manager:
            return self.tab_manager.get_current_tab_data()
        return None

    def _get_current_editor(self) -> Optional["CodeEditor"]:
        if hasattr(self, "tab_manager") and self.tab_manager:
            return self.tab_manager.get_current_editor()
        return None

    def _load_runtime_settings(self) -> dict:
        defaults = {
            k: getattr(global_config, k)
            for k in dir(global_config)
            if not k.startswith("__")
            and isinstance(
                getattr(global_config, k), (str, int, float, bool, list, tuple)
            )
        }
        if hasattr(global_config, "BNB_4BIT_COMPUTE_DTYPE"):
            defaults["bnb_4bit_compute_dtype"] = str(
                global_config.BNB_4BIT_COMPUTE_DTYPE
            )

        defaults[self.DEFAULT_PROJECT_PATH_KEY] = QDir.homePath()

        defaults['linter_ignore_codes'] = ["E501"]

        defaults["server_api_url"] = getattr(
            global_config,
            "DEFAULT_SERVER_API_URL",
            "https://example.com/api_placeholder",
        )
        defaults["client_api_key"] = getattr(
            global_config, "DEFAULT_CLIENT_API_KEY", "CONFIGURE_ME_IN_APP_SETTINGS_JSON"
        )
        defaults["client_sync_id"] = getattr(
            global_config, "DEFAULT_CLIENT_SYNC_ID", f"client_{os.urandom(4).hex()}"
        )
        defaults["last_successful_sync_timestamp_utc"] = (
            None  # Wird beim ersten erfolgreichen Sync gesetzt
        )
        defaults["current_model_adapter_version"] = None
        defaults["lora_adapter_path"] = getattr(
            global_config, "LORA_ADAPTER_PATH", "codellama-7b-lora-adapters"
        )

        settings_path = Path(self.CONFIG_FILE)
        if settings_path.exists():
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    loaded_settings = json.load(f)
                for key, value in defaults.items():
                    if key not in loaded_settings:
                        loaded_settings[key] = value
                logger.info(
                    f"Einstellungen aus '{self.CONFIG_FILE}' geladen und mit Defaults ergänzt."
                )
                final_settings = loaded_settings
            except Exception as e:
                logger.warning(
                    f"Fehler beim Laden von '{self.CONFIG_FILE}': {e}. Verwende Defaults.",
                    exc_info=True,
                )
                final_settings = defaults.copy()
        else:
            logger.info(
                f"Keine Einstellungsdatei '{self.CONFIG_FILE}' gefunden. Verwende Defaults und erstelle Datei."
            )
            final_settings = defaults.copy()
            try:
                with open(settings_path, "w", encoding="utf-8") as f:
                    json.dump(final_settings, f, indent=4)
                logger.info(
                    f"Standardeinstellungen in '{self.CONFIG_FILE}' gespeichert."
                )
            except Exception as e_save:
                logger.error(
                    f"Konnte Standardeinstellungen nicht in '{self.CONFIG_FILE}' speichern: {e_save}"
                )

        if "device" not in final_settings and hasattr(global_config, "DEVICE"):  # type: ignore
            final_settings["device"] = global_config.DEVICE  # type: ignore
        return final_settings

    def _save_runtime_settings(self):
        settings_path = Path(self.CONFIG_FILE)
        try:
            save_settings = self.runtime_settings.copy()
            if self.project_root_path:
                save_settings[self.DEFAULT_PROJECT_PATH_KEY] = str(
                    self.project_root_path
                )

            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(save_settings, f, indent=4)
            logger.info(f"Einstellungen in '{self.CONFIG_FILE}' gespeichert.")
            if self.statusBar():
                self.statusBar().showMessage("Einstellungen gespeichert.", 3000)
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Einstellungen: {e}", exc_info=True)
            QMessageBox.warning(
                self,
                "Speicherfehler",
                f"Einstellungen konnten nicht gespeichert werden:\n{e}",
            )
            if self.statusBar():
                self.statusBar().showMessage(
                    f"Fehler beim Speichern der Einstellungen: {e}", 5000
                )

    def _initialize_accelerator(self):
        self.accelerator = None
        if ACCELERATE_AVAILABLE and Accelerator:
            try:
                self.accelerator = Accelerator()
                logger.info(
                    f"Accelerator initialized on device: {self.accelerator.device}"
                )
                self.runtime_settings["device"] = str(self.accelerator.device)
            except Exception as acc_e:
                logger.warning(
                    f"Failed to initialize Accelerator: {acc_e}", exc_info=True
                )
                self.accelerator = None
        else:
            logger.info("Accelerator not available or not used.")

    def _initialize_services(self):
        # 1. Prüfen, ob die DataManager-KLASSE überhaupt importiert wurde
        if DataManager is None:  # Prüft die importierte Klasse DataManager
            QMessageBox.critical(
                self,
                "Fatal Error",
                "DataManager Klasse konnte nicht geladen werden (Importfehler).",
            )
            logger.critical(
                "DataManager Klasse konnte nicht geladen werden (Importfehler). Anwendung wird beendet."
            )
            sys.exit(1)  # Beenden, da eine kritische Komponente fehlt

        # 2. DataManager-INSTANZ initialisieren
        db_path_key = "db_path"
        db_path_default = "code_snippets.db"
        if hasattr(global_config, "DB_PATH"):
            db_path_default = global_config.DB_PATH

        # Erstellen der Instanz
        self.data_manager = DataManager(
            db_path=self.runtime_settings.get(db_path_key, db_path_default)
        )
        if not self.data_manager.initialize_database():
            QMessageBox.warning(
                self,
                "DB Fehler",
                "Datenbank konnte nicht initialisiert werden. Bitte Logs prüfen.",
            )
            logger.error("Datenbank konnte nicht initialisiert werden.")

        self.formatting_service = None
        if FormattingService:
            try:
                self.formatting_service = FormattingService()
            except Exception as e:
                logger.error(
                    f"Failed to initialize FormattingService: {e}", exc_info=True
                )
        else:
            logger.warning("FormattingService class not loaded.")

        # DataManager Instanz erstellen, auch wenn sie None sein könnte
        self.data_manager = DataManager(
            db_path=self.runtime_settings.get(db_path_key, db_path_default)
        )
        if (
            not self.data_manager.initialize_database()
        ):  # initialize_database sollte bool zurückgeben
            QMessageBox.warning(
                self,
                "DB Fehler",
                "Datenbank konnte nicht initialisiert werden. Bitte Logs prüfen.",
            )
            logger.error("Datenbank konnte nicht initialisiert werden.")
            # Hier nicht unbedingt beenden, aber Funktionalität ist eingeschränkt

        self.inference_service = None
        self.code_corrector = None
        self.hf_fine_tuner = None

        if HF_LIBRARIES_AVAILABLE:
            if InferenceService:
                try:
                    self.inference_service = InferenceService(
                        self.runtime_settings, self.accelerator
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to initialize InferenceService: {e}", exc_info=True
                    )
            else:
                logger.warning("InferenceService class not loaded.")

            if CodeCorrector and self.inference_service:
                try:
                    self.code_corrector = CodeCorrector(
                        self.inference_service, self.runtime_settings
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to initialize CodeCorrector: {e}", exc_info=True
                    )
            elif not self.inference_service:
                logger.warning(
                    "Cannot initialize CodeCorrector, InferenceService is missing."
                )
            else:
                logger.warning("CodeCorrector class not loaded.")

            if HFFineTuner and self.data_manager:
                try:
                    self.hf_fine_tuner = HFFineTuner(
                        self.data_manager, self.runtime_settings, self.accelerator
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to initialize HFFineTuner: {e}", exc_info=True
                    )
            elif not self.data_manager:
                logger.warning("Cannot initialize HFFineTuner, DataManager is missing.")
            else:
                logger.warning("HFFineTuner class not loaded.")
        else:
            logger.warning(
                "Hugging Face libraries not available. HF features disabled."
            )

        self.linter_service = None
        if LinterService:
            try:
                self.linter_service = LinterService(self.runtime_settings)
            except Exception as e:
                logger.error(f"Failed to initialize LinterService: {e}", exc_info=True)
        else:
            logger.warning("LinterService class not loaded.")

        self.fine_tune_thread = None

        if (
            ServerConnector is None
        ):  # Prüft, ob die ServerConnector-KLASSE importiert wurde
            logger.warning(
                "ServerConnector Klasse konnte nicht geladen werden (Importfehler). Sync-Funktionalität wird nicht verfügbar sein."
            )
            self.server_connector = (
                None  # Sicherstellen, dass das Attribut existiert, aber None ist
            )
        elif (
            self.data_manager
        ):  # Prüft, ob die DataManager-INSTANZ erfolgreich erstellt wurde
            self.server_connector = ServerConnector(self, self.data_manager)
            logger.info("ServerConnector erfolgreich initialisiert.")
        else:
            # Dieser Fall sollte eigentlich nicht eintreten, wenn die DataManager-Instanziierung oben robust ist.
            logger.error(
                "ServerConnector konnte nicht initialisiert werden, da die DataManager-Instanz fehlt."
            )
            self.server_connector = None

    # --- TabManager Signal Slots ---
    @Slot(int, object)
    def on_tab_manager_current_tab_changed(
        self, index: int, tab_data: Optional[Dict[str, Any]]
    ):
        logger.debug(
            f"MainWindow: Slot on_tab_manager_current_tab_changed, Index: {index}"
        )
        editor_for_filter: Optional[CodeEditor] = None
        if tab_data:
            if hasattr(self, "quality_label_combo") and self.quality_label_combo:
                self.quality_label_combo.setEnabled(tab_data["path"] is not None)
                self.quality_label_combo.blockSignals(True)
                self.quality_label_combo.setCurrentText(tab_data["quality_label"])
                self.quality_label_combo.blockSignals(False)

            editor = tab_data.get("editor")
            editor_for_filter = editor
            if editor:
                lint_timer = tab_data.get("lint_delay_timer")
                if editor.toPlainText().strip() and lint_timer:
                    lint_timer.start()
                elif lint_timer:
                    lint_timer.stop()
                if (
                    hasattr(self, "results_display_list")
                    and self.results_display_list
                    and not editor.toPlainText().strip()
                ):
                    self.results_display_list.clear()
                if (
                    hasattr(editor, "clear_lint_errors")
                    and not editor.toPlainText().strip()
                ):
                    editor.clear_lint_errors()
                editor.setFocus()
        else:
            if hasattr(self, "quality_label_combo") and self.quality_label_combo:
                self.quality_label_combo.setEnabled(False)
                self.quality_label_combo.setCurrentIndex(0)
            if hasattr(self, "results_display_list") and self.results_display_list:
                self.results_display_list.clear()

        if self.ai_service_coordinator and editor_for_filter:
            self.ai_service_coordinator.install_event_filter_on_editor(
                editor_for_filter
            )

        self.update_window_title()
        self.update_button_states()
        if (
            self.search_controller
            and hasattr(self, "find_replace_widget")
            and self.find_replace_widget
            and self.find_replace_widget.isVisible()
        ):
            self.search_controller._update_find_buttons_state()

    @Slot(int, object)
    def on_tab_manager_tab_added(self, index: int, tab_data: Dict[str, Any]):
        logger.debug(f"MainWindow: Slot on_tab_manager_tab_added, Index: {index}")
        editor = tab_data.get("editor")
        if editor and self.ai_service_coordinator:
            self.ai_service_coordinator.install_event_filter_on_editor(editor)
        self.update_window_title()
        self.update_button_states()

    @Slot(object)
    def on_tab_manager_tab_closed(self, closed_tab_path_obj: object):
        logger.debug(
            f"MainWindow: Slot on_tab_manager_tab_closed, Objekt: {closed_tab_path_obj}"
        )
        self.update_window_title()
        self.update_button_states()

    @Slot()
    def on_tab_manager_last_tab_closed(self):
        logger.debug("MainWindow: Slot on_tab_manager_last_tab_closed.")
        self.update_window_title()
        self.update_button_states()

    @Slot(int, bool)
    def on_tab_manager_tab_modification_changed(self, index: int, is_modified: bool):
        logger.debug(
            f"MainWindow: Slot on_tab_manager_tab_modification_changed, Index: {index}, Modified: {is_modified}"
        )
        if (
            hasattr(self, "tab_widget")
            and self.tab_widget
            and self.tab_widget.currentIndex() == index
        ):
            self.update_window_title()
            self.update_button_states()

    @Slot(int, str)
    def on_tab_manager_tab_title_updated(self, index: int, new_title: str):
        if (
            hasattr(self, "tab_widget")
            and self.tab_widget
            and self.tab_widget.currentIndex() == index
        ):
            self.update_window_title()

    # --- Dateioperationen (delegiert an FileManager) ---
    @Slot()
    def on_file_new(self):
        if self.file_manager:
            self.file_manager.handle_new_file()
        else:
            logger.error("FileManager nicht initialisiert für on_file_new.")

    @Slot()
    def on_file_open(self):
        if self.file_manager:
            self.file_manager.handle_open_files()
        else:
            logger.error("FileManager nicht initialisiert für on_file_open.")

    @Slot()
    def on_file_save(self) -> bool:
        if self.file_manager:
            return self.file_manager.handle_save_file()
        logger.error("FileManager nicht initialisiert für on_file_save.")
        return False

    @Slot()
    def on_file_save_as(self) -> bool:
        if self.file_manager:
            return self.file_manager.handle_save_file_as()
        logger.error("FileManager nicht initialisiert für on_file_save_as.")
        return False

    @Slot()
    def on_close_current_tab_action(self):
        if self.tab_manager:
            current_index = self.tab_manager.get_current_editor_index()
            if current_index != -1:
                self.on_tab_close_requested(current_index)

    @Slot(int)
    def on_tab_close_requested(self, index: int):
        if not self.tab_manager or not self.file_manager:
            return

        if hasattr(self, "tab_widget") and self.tab_widget:
            self.tab_widget.setCurrentIndex(index)

        if self.file_manager.maybe_save_current_tab():
            self.tab_manager.remove_tab(index)

    # --- Projekt-Explorer Operationen (delegiert an ProjectExplorerManager) ---
    @Slot()
    def on_open_folder_clicked(self):
        if self.project_explorer_manager:
            self.project_explorer_manager.handle_open_folder()
            # NEU: CodeContextAnalyzer aktualisieren, nachdem der Projektordner geändert wurde
            if self.code_context_analyzer and self.project_root_path:
                logger.info(
                    f"Aktualisiere CodeContextAnalyzer mit neuem Projektpfad: {self.project_root_path}"
                )
                self.code_context_analyzer.update_project_root(self.project_root_path)
            elif self.code_context_analyzer:  # Falls kein Projektpfad mehr gesetzt ist
                self.code_context_analyzer.update_project_root(None)
        else:
            logger.error(
                "ProjectExplorerManager nicht initialisiert für on_open_folder_clicked."
            )
            QMessageBox.warning(
                self, "Fehler", "ProjectExplorerManager ist nicht initialisiert."
            )

    # --- Suchen/Ersetzen Operationen (delegiert an SearchController) ---
    @Slot()
    def show_find_replace_bar(self):
        if self.search_controller:
            self.search_controller.show_bar()
        else:
            logger.error(
                "SearchController nicht initialisiert für show_find_replace_bar."
            )

    @Slot()
    def on_find_in_project_triggered(self):
        if self.search_controller:
            self.search_controller.trigger_project_search()
        else:
            logger.error(
                "SearchController nicht initialisiert für on_find_in_project_triggered."
            )

    @Slot(QListWidgetItem)
    def on_project_search_result_activated(self, item: QListWidgetItem):
        data = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(data, dict):
            file_path_str = data.get("file_path")
            line_number = data.get("line_number")
            column_number = data.get("column", 0)

            if file_path_str and line_number is not None:
                target_path = Path(file_path_str)
                if self.file_manager:
                    self.file_manager._open_single_file_in_tab(target_path)
                QTimer.singleShot(
                    100,
                    lambda: self._goto_line_in_current_tab(line_number, column_number),
                )

    def _goto_line_in_current_tab(self, line_number: int, column_number: int):
        current_editor = self._get_current_editor()
        if current_editor and hasattr(current_editor, "goto_line"):
            logger.debug(
                f"Springe zu Zeile {line_number}, Spalte {column_number} im aktuellen Editor."
            )
            current_editor.goto_line(
                line_number, select_line=True, column=column_number
            )
            current_editor.setFocus()

    # --- Einstellungsdialog ---
    @Slot()
    def on_open_settings(self):
        if SettingsDialog is None:
            QMessageBox.warning(
                self, "Fehler", "SettingsDialog konnte nicht geladen werden."
            )
            return

        dialog = SettingsDialog(self.runtime_settings, self)
        if dialog.exec() == QDialog.Accepted:
            new_settings = dialog.get_settings()
            if new_settings != self.runtime_settings:
                logger.info("Einstellungen werden aktualisiert...")
                self.runtime_settings.update(new_settings)

                if self.inference_service:
                    self.inference_service.config_dict = self.runtime_settings
                if self.code_corrector:
                    self.code_corrector.config_dict = self.runtime_settings
                if self.hf_fine_tuner:
                    self.hf_fine_tuner.config_dict = self.runtime_settings
                if self.linter_service:
                    self.linter_service.config_dict = self.runtime_settings
                    self.linter_service.ignore_codes = self.runtime_settings.get("linter_ignore_codes", [])
                    logger.info(f"LinterService ignoriert jetzt: {self.linter_service.ignore_codes}")

                logger.info("Service-Konfigurationen aktualisiert.")
                self._save_runtime_settings()

                if self.inference_service and hasattr(
                    self.inference_service, "_load_inference_model"
                ):
                    logger.info("Reloading inference model due to settings change...")
                    self.inference_service._load_inference_model()
                    if hasattr(self.inference_service, "get_status"):
                        self.modelStatusChanged.emit(
                            self.inference_service.get_status()
                        )
                self.update_button_states()
            else:
                logger.debug("Einstellungsänderungen verworfen (keine Änderungen).")
        else:
            logger.debug("Einstellungsdialog abgebrochen.")

    # --- Kontroll-Button Slots (DB, Laden, KI-Aktionen) ---
    @Slot()
    def on_init_db_clicked(self):
        if not self.data_manager:
            if self.statusBar():
                self.statusBar().showMessage("DataManager nicht initialisiert.", 3000)
            logger.warning("on_init_db_clicked: DataManager nicht initialisiert.")
            return
        if self.data_manager.initialize_database():
            db_path_key = "db_path"
            db_path_default = "code_snippets.db"
            if hasattr(global_config, "DB_PATH"):
                db_path_default = global_config.DB_PATH
            db_path = self.runtime_settings.get(db_path_key, db_path_default)
            msg = f"Datenbank '{db_path}' initialisiert/geöffnet."
            if self.statusBar():
                self.statusBar().showMessage(msg, 3000)
            logger.info(msg)
        else:
            if self.statusBar():
                self.statusBar().showMessage(
                    "Fehler bei der Datenbankinitialisierung.", 3000
                )
            logger.error(
                "Fehler bei der Datenbankinitialisierung in on_init_db_clicked."
            )
        self.update_button_states()

    @Slot()
    def on_load_py_files_clicked(self):
        if self.file_manager:
            self.file_manager.handle_load_py_files_to_db()
        else:
            logger.error(
                "FileManager nicht initialisiert für on_load_py_files_clicked."
            )

    @Slot()
    def on_finetune_hf_clicked(self):
        # Verwende das Instanzattribut, das die importierte Klasse referenziert
        if self.FineTuneThread_CLASS_REF is None:
            if self.statusBar():
                self.statusBar().showMessage(
                    "FineTuneThread Klasse nicht geladen.", 3000
                )
            logger.warning(
                "on_finetune_hf_clicked: FineTuneThread Klasse nicht geladen (Attribut FineTuneThread_CLASS_REF ist None)."
            )
            return

        if (
            self.fine_tune_thread and self.fine_tune_thread.isRunning()
        ):  # Prüft die Instanz des laufenden Threads
            if self.statusBar():
                self.statusBar().showMessage("Fine-Tuning läuft bereits.", 3000)
            logger.info("on_finetune_hf_clicked: Fine-Tuning läuft bereits.")
            return

        if not self.hf_fine_tuner:
            if self.statusBar():
                self.statusBar().showMessage("HFFineTuner Service nicht bereit.", 3000)
            logger.warning("on_finetune_hf_clicked: HFFineTuner Service nicht bereit.")
            return

        # Verwende das Instanzattribut für HF_LIBRARIES_AVAILABLE_FLAG
        if not self.HF_LIBRARIES_AVAILABLE_FLAG:
            if self.statusBar():
                self.statusBar().showMessage(
                    "Erforderliche HF-Bibliotheken nicht verfügbar.", 3000
                )
            logger.warning(
                "on_finetune_hf_clicked: Erforderliche HF-Bibliotheken nicht verfügbar."
            )
            return

        can_load_snippets = self.runtime_settings.get("hf_train_with_snippets", True)
        can_load_feedback = self.runtime_settings.get("hf_train_with_feedback", True)
        if not (can_load_snippets or can_load_feedback):
            QMessageBox.warning(
                self,
                "Keine Trainingsdatenquellen",
                "Bitte konfiguriere in den Einstellungen, ob Snippets und/oder Feedback-Daten für das Training verwendet werden sollen.",
            )
            logger.info(
                "on_finetune_hf_clicked: Keine Trainingsdatenquellen konfiguriert."
            )
            return

        if self.data_manager:
            snippet_quality_label = self.runtime_settings.get(
                "hf_snippet_training_quality_label", "gut (Training)"
            )
            positive_feedback_types = self.runtime_settings.get(
                "hf_positive_feedback_types",
                ["suggestion_used", "correction_applied", "correction_good"],
            )

            test_snippets = (
                self.data_manager.get_all_snippets_text(
                    quality_filter=[snippet_quality_label]
                )
                if can_load_snippets
                else ""
            )
            test_feedback = (
                self.data_manager.get_feedback_data_for_training(
                    feedback_types=positive_feedback_types
                )
                if can_load_feedback
                else []
            )

            if not ((test_snippets and test_snippets.strip()) or test_feedback):
                QMessageBox.warning(
                    self,
                    "Keine Trainingsdaten",
                    "Es wurden keine Daten (weder 'gut (Training)'-Snippets noch positives Feedback) für das Fine-Tuning gefunden.",
                )
                logger.info(
                    "on_finetune_hf_clicked: Keine tatsächlichen Trainingsdaten in der DB gefunden."
                )
                return

        self.set_buttons_enabled(False)
        hf_model_name = self.runtime_settings.get("hf_base_model_name", "HF Model")
        hf_model_short_name = hf_model_name.split("/")[-1]
        status_message = f"Starte {hf_model_short_name} QLoRA Fine-Tuning..."
        if self.statusBar():
            self.statusBar().showMessage(status_message)
        logger.info(status_message)
        if hasattr(self, "progress_bar") and self.progress_bar:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setTextVisible(False)

        if self.hf_fine_tuner:  # Sicherstellen, dass hf_fine_tuner existiert
            # Hier wird die Klasse (gespeichert in self.FineTuneThread_CLASS_REF) zur Instanziierung verwendet
            self.fine_tune_thread = self.FineTuneThread_CLASS_REF(
                hf_fine_tuner_instance=self.hf_fine_tuner
            )
            self.fine_tune_thread.progressUpdated.connect(self.handle_training_progress)
            self.fine_tune_thread.finished.connect(self.on_finetune_hf_finished)
            self.fine_tune_thread.start()
            logger.debug("FineTuneThread gestartet.")
        else:
            logger.error(
                "HFFineTuner ist None, FineTuneThread kann nicht gestartet werden."
            )
            self.set_buttons_enabled(True)  # UI wieder aktivieren

    def on_finetune_hf_finished(self, success: bool, result_obj: object):
        logger.debug(f"FineTuneThread finished Signal empfangen. Erfolg: {success}")
        self.set_buttons_enabled(True)

        if hasattr(self, "progress_bar") and self.progress_bar:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100 if success else 0)
            self.progress_bar.setTextVisible(True)
            self.progress_bar.setFormat(
                "Abgeschlossen" if success else "Fehlgeschlagen"
            )

        result_message = "Fine-Tuning Prozess beendet."
        if isinstance(result_obj, dict) and "message" in result_obj:
            result_message = result_obj["message"]
        elif isinstance(result_obj, str):
            result_message = result_obj

        if success:
            if self.statusBar():
                self.statusBar().showMessage(result_message, 7000)
            logger.info(f"QLoRA Fine-Tuning erfolgreich: {result_message}")
            if self.inference_service and hasattr(
                self.inference_service, "_load_inference_model"
            ):
                logger.info("Reloading model in InferenceService after fine-tuning...")
                self.inference_service._load_inference_model()
                if hasattr(self.inference_service, "get_status"):
                    self.modelStatusChanged.emit(self.inference_service.get_status())
            else:
                logger.warning(
                    "InferenceService not available or cannot reload model after fine-tuning."
                )
        else:
            if self.statusBar():
                self.statusBar().showMessage(
                    f"QLoRA Fine-Tuning Fehlgeschlagen: {result_message}", 7000
                )
            logger.error(f"QLoRA Fine-Tuning Fehlgeschlagen: {result_message}")

        self.fine_tune_thread = None
        logger.debug("Fine-Tuning UI wieder aktiviert.")

    @Slot()
    def on_export_log_clicked(self):
        if (
            not hasattr(self, "results_display_list")
            or not self.results_display_list
            or self.results_display_list.count() == 0
        ):
            if self.statusBar():
                self.statusBar().showMessage("Kein Log zum Exportieren.", 3000)
            logger.info("on_export_log_clicked: Kein Log zum Exportieren.")
            return

        log_content = [
            self.results_display_list.item(i).text()
            for i in range(self.results_display_list.count())
            if self.results_display_list.item(i)
        ]
        full_log_text = "\n".join(log_content)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = self.output_dir / f"analysis_log_{timestamp}.txt"

        file_path_tuple: Tuple[str, str] = QFileDialog.getSaveFileName(
            self,
            "Analyse-Log Speichern",
            str(default_filename),
            "Textdateien (*.txt);;Alle Dateien (*)",
        )
        filePath: str = file_path_tuple[0]

        if filePath:
            try:
                with open(filePath, "w", encoding="utf-8") as f:
                    f.write(full_log_text)
                if self.statusBar():
                    self.statusBar().showMessage(
                        f"Log exportiert nach: {filePath}", 5000
                    )
                logger.info(f"Log exportiert nach {filePath}")
            except IOError as e:
                QMessageBox.critical(
                    self, "Exportfehler", f"Export fehlgeschlagen:\n{e}"
                )
                if self.statusBar():
                    self.statusBar().showMessage(f"Fehler Export: {e}", 5000)
                logger.error(f"Fehler beim Exportieren des Logs: {e}", exc_info=True)
            except Exception as e:
                QMessageBox.critical(self, "Unerwarteter Exportfehler", f"Fehler:\n{e}")
                if self.statusBar():
                    self.statusBar().showMessage(f"Unerwarteter Fehler: {e}", 5000)
                logger.error(
                    f"Unerwarteter Fehler beim Exportieren des Logs: {e}", exc_info=True
                )
        else:
            if self.statusBar():
                self.statusBar().showMessage("Log-Export abgebrochen.", 3000)
            logger.info("Log-Export abgebrochen.")

    # --- Slots für KI-Aktionen (delegiert an AIServiceCoordinator) ---
    @Slot()
    def on_lint_code_clicked(
        self,
        triggered_by_timer: bool = False,
        editor_instance: Optional["CodeEditor"] = None,
    ):
        if self.ai_service_coordinator:
            self.ai_service_coordinator.handle_lint_request(
                editor_instance if editor_instance else self._get_current_editor(),
                triggered_by_timer,
            )
        else:
            logger.error(
                "AIServiceCoordinator nicht initialisiert für on_lint_code_clicked."
            )

    @Slot()
    def on_correct_code_clicked(self):
        if self.ai_service_coordinator:
            self.ai_service_coordinator.handle_code_correction_request()
        else:
            logger.error(
                "AIServiceCoordinator nicht initialisiert für on_correct_code_clicked."
            )

    @Slot()
    def on_format_code_clicked(self):
        """Delegates the code formatting request to the AIServiceCoordinator."""
        if self.ai_service_coordinator:
            self.ai_service_coordinator.handle_format_code_request()
        else:
            logger.error(
                "AIServiceCoordinator nicht initialisiert für on_format_code_clicked."
            )
            QMessageBox.warning(
                self, "Fehler", "Formatierungsdienst ist nicht verfügbar."
            )

    @Slot()
    def on_improve_code_clicked(self):
        """Wird aufgerufen, wenn der 'KI-Verbesserung'-Button geklickt wird."""
        logger.debug("MainWindow: 'on_improve_code_clicked' Slot aufgerufen.")
        current_editor = (
            self._get_current_editor()
        )  # Nimmt den Editor aus dem aktuellen Tab
        if not current_editor:
            self.statusBar().showMessage(
                "Kein aktiver Editor für Code-Verbesserung.", 3000
            )
            return

        selected_text = current_editor.textCursor().selectedText()
        code_to_improve = ""
        is_selection = False

        if selected_text:
            code_to_improve = selected_text
            is_selection = True
            logger.info(
                f"KI-Verbesserung für markierten Text angefordert (Länge: {len(code_to_improve)})."
            )
        else:
            code_to_improve = current_editor.toPlainText()
            if not code_to_improve.strip():
                self.statusBar().showMessage(
                    "Kein Code im Editor für Verbesserung vorhanden.", 3000
                )
                return
            logger.info(
                "KI-Verbesserung für gesamten Code im aktuellen Tab angefordert."
            )

        if self.ai_service_coordinator:
            # Eine neue Methode im AIServiceCoordinator wird benötigt
            self.ai_service_coordinator.handle_code_improvement_request(
                code_to_improve, is_selection
            )
        else:
            logger.error(
                "AIServiceCoordinator nicht initialisiert für on_improve_code_clicked."
            )
            QMessageBox.warning(
                self, "Fehler", "AI Service Coordinator ist nicht verfügbar."
            )

    @Slot()
    def on_apply_corrections_clicked(self):
        if self.ai_service_coordinator:
            self.ai_service_coordinator.handle_apply_corrections()
        else:
            logger.error(
                "AIServiceCoordinator nicht initialisiert für on_apply_corrections_clicked."
            )

    @Slot()
    def on_correction_feedback_good_clicked(self):
        if self.ai_service_coordinator:
            self.ai_service_coordinator.handle_correction_feedback_good()
        else:
            logger.error(
                "AIServiceCoordinator nicht initialisiert für on_correction_feedback_good_clicked."
            )

    @Slot()
    def on_correction_feedback_bad_clicked(self):
        if self.ai_service_coordinator:
            self.ai_service_coordinator.handle_correction_feedback_bad()
        else:
            logger.error(
                "AIServiceCoordinator nicht initialisiert für on_correction_feedback_bad_clicked."
            )

    # --- Qualitätslabel ---
    @Slot(str)
    def on_quality_label_changed_ui(self, new_label: str):
        if not self.tab_manager:
            return
        current_index = self.tab_manager.get_current_editor_index()
        if current_index != -1:
            self.tab_manager.set_quality_label(current_index, new_label)
            logger.debug(
                f"Qualitätslabel UI geändert für Tab {current_index} zu '{new_label}'. "
                "Wird beim Speichern in die DB übernommen."
            )
            editor = self.tab_manager.get_editor(current_index)
            if editor:
                editor.document().setModified(True)

    @Slot()
    def on_sync_with_server_clicked(self):
        """Wird aufgerufen, wenn der Sync-Button/Menüeintrag geklickt wird."""
        if not self.server_connector:
            QMessageBox.warning(
                self,
                "Synchronisation Fehler",
                "ServerConnector nicht initialisiert. Bitte Konfiguration prüfen und ggf. neu starten.",
            )
            logger.warning(
                "on_sync_with_server_clicked: ServerConnector nicht verfügbar."
            )
            return

        logger.info("Starte Synchronisation mit dem Server...")
        # Deaktiviere den Sync-Button während des Vorgangs, um doppelte Klicks zu verhindern
        if self.action_sync_with_server:
            self.action_sync_with_server.setEnabled(False)
        # Ggf. auch einen Button im UI deaktivieren, falls vorhanden
        # self.sync_button_ui_element.setEnabled(False) # Beispiel

        self.statusBar().showMessage("Synchronisiere Daten mit Server...", 0)
        QApplication.processEvents()  # UI aktualisieren

        # Führe den Upload in einem separaten Thread aus, um die GUI nicht zu blockieren.
        # Für Phase 1 ist es noch direkt, aber für später ist ein Thread besser.
        # Hier erstmal direkt für Einfachheit in Phase 1.
        # TODO: Für längere Operationen in einen QThread auslagern.
        success, message = self.server_connector.upload_data_batch_to_server()

        if success:
            self.statusBar().showMessage(
                f"Synchronisation: {message}", 7000
            )  # Längere Anzeigezeit
            logger.info(f"Synchronisation: {message}")
        else:
            self.statusBar().showMessage(
                f"Synchronisationsfehler: {message}", 10000
            )  # Längere Anzeigezeit
            QMessageBox.critical(self, "Synchronisationsfehler", message)
            logger.error(f"Synchronisationsfehler: {message}")

        if self.action_sync_with_server:
            self.action_sync_with_server.setEnabled(True)
        # self.sync_button_ui_element.setEnabled(True) # Beispiel
        self.update_button_states()

    @Slot()
    def on_check_for_model_update_clicked(self):
        """Prüft auf neue Modellversionen auf dem Server und bietet Download an."""
        if not self.server_connector:
            QMessageBox.warning(
                self, "Modell-Update Fehler", "ServerConnector nicht initialisiert."
            )
            return

        self.statusBar().showMessage("Prüfe auf Modell-Updates...", 0)
        QApplication.processEvents()

        model_info, msg = self.server_connector.get_latest_model_info_from_server()

        if not model_info or not model_info.get("latest_model_version"):
            QMessageBox.information(
                self,
                "Modell-Update",
                f"Fehler beim Abrufen der Modellinformationen: {msg}",
            )
            self.statusBar().showMessage(f"Fehler Modell-Info: {msg}", 5000)
            return

        latest_server_version = model_info["latest_model_version"]
        metadata = model_info.get("metadata")
        current_local_version = self.runtime_settings.get(
            "current_model_adapter_version"
        )

        info_text = f"Neueste Version auf Server: {latest_server_version}\n"
        if metadata:
            info_text += f"  Trainingsdatum: {metadata.get('training_date', 'N/A')}\n"
            info_text += (
                f"  Hochgeladen am: {metadata.get('upload_timestamp_utc', 'N/A')}\n"
            )
            if metadata.get("notes"):
                info_text += f"  Notizen: {metadata['notes']}\n"

        if current_local_version:
            info_text += f"\nAktuell verwendete Version: {current_local_version}\n"
        else:
            info_text += "\nKeine lokale Versionsinformation gefunden.\n"

        if latest_server_version == current_local_version:
            info_text += "\nSie verwenden bereits die neueste Version."
            QMessageBox.information(self, "Modell-Update", info_text)
            self.statusBar().showMessage("Modell ist aktuell.", 3000)
            return

        info_text += f"\nMöchten Sie die Version '{latest_server_version}' herunterladen und anwenden?"
        reply = QMessageBox.question(
            self,
            "Modell-Update verfügbar",
            info_text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply == QMessageBox.StandardButton.Yes:
            lora_adapter_base_path_str = self.runtime_settings.get(
                "lora_adapter_path", "codellama-7b-lora-adapters"
            )
            # Das Zielverzeichnis für den Adapter ist der konfigurierte Pfad.
            # Der ServerConnector wird dieses Verzeichnis (falls vorhanden) leeren und neu befüllen.
            target_adapter_dir = Path(lora_adapter_base_path_str).resolve()

            # Sicherstellen, dass das übergeordnete Verzeichnis für die temporäre ZIP existiert
            if not target_adapter_dir.parent.exists():
                try:
                    target_adapter_dir.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e_mkdir:
                    QMessageBox.critical(
                        self,
                        "Fehler",
                        f"Konnte übergeordnetes Verzeichnis für Download nicht erstellen: {e_mkdir}",
                    )
                    self.statusBar().showMessage(
                        "Fehler bei Download-Vorbereitung.", 5000
                    )
                    return

            self.statusBar().showMessage(
                f"Starte Download von Modell '{latest_server_version}'...", 0
            )
            QApplication.processEvents()

            success, dl_msg, adapter_path = (
                self.server_connector.download_model_from_server(
                    latest_server_version,
                    target_adapter_dir,  # Das ist das Verzeichnis, in das direkt entpackt wird
                )
            )

            if success and adapter_path:
                QMessageBox.information(
                    self,
                    "Modell-Update",
                    f"Modell '{latest_server_version}' erfolgreich heruntergeladen nach:\n{adapter_path}\n"
                    "Der KI-Dienst wird nun neu geladen, um das Update anzuwenden.",
                )
                self.statusBar().showMessage(
                    f"Modell '{latest_server_version}' angewendet.", 5000
                )

                # KI-Dienst (InferenceService) anweisen, das Modell neu zu laden
                if self.inference_service and hasattr(
                    self.inference_service, "_load_inference_model"
                ):
                    logger.info("Lade Inferenzmodell nach Update neu...")
                    self.inference_service._load_inference_model()  # Diese Methode muss den Adapter neu laden
                    if hasattr(self.inference_service, "get_status"):
                        self.modelStatusChanged.emit(
                            self.inference_service.get_status()
                        )
                else:
                    logger.warning(
                        "InferenceService nicht verfügbar oder kann Modell nicht neu laden. Ggf. Neustart erforderlich."
                    )
                    QMessageBox.information(
                        self,
                        "Hinweis",
                        "InferenceService konnte nicht automatisch neu geladen werden. Bitte starten Sie die Anwendung ggf. neu, um das Update zu aktivieren.",
                    )
            else:
                QMessageBox.critical(
                    self, "Modell-Download Fehler", f"Download fehlgeschlagen: {dl_msg}"
                )
                self.statusBar().showMessage(
                    f"Modell-Download fehlgeschlagen: {dl_msg}", 7000
                )
        else:
            self.statusBar().showMessage("Modell-Update abgebrochen.", 3000)

    def closeEvent(self, event: QEvent):
        all_saved = True
        if hasattr(self, "tab_widget") and self.tab_widget and self.file_manager:
            for i in range(self.tab_widget.count()):  # Iterate through existing tabs
                self.tab_widget.setCurrentIndex(i)
                if not self.file_manager.maybe_save_current_tab():
                    all_saved = False
                    break
        elif self.file_manager:
            pass
        else:
            all_saved = True

        if not all_saved:
            event.ignore()
            return

        logger.info("Schließe Anwendung...")
        if self.fine_tune_thread and self.fine_tune_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Fine-Tuning Läuft",
                "Fine-Tuning läuft. Abbrechen und Schließen?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                logger.info("Stoppe Fine-Tuning Thread...")
                self.fine_tune_thread.stop()
                if not self.fine_tune_thread.wait(5000):
                    logger.warning("Fine-Tuning Thread nicht beendet. Terminiere...")
                    self.fine_tune_thread.terminate()
                    self.fine_tune_thread.wait()
                logger.info("Fine-Tuning Thread gestoppt.")
            else:
                event.ignore()
                return

        if hasattr(self, "data_manager") and self.data_manager:
            self.data_manager.close_connection()
            logger.info("DB geschlossen.")

        logger.info("Anwendung beendet.")
        super().closeEvent(event)


# --- Standalone Test Block ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
