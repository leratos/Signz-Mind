# cronjob/tower_daily_workflow.py
# -*- coding: utf-8 -*-

"""
Main orchestration script for the Tower PC's daily automated workflow.

This script is intended to be run as a scheduled task (e.g., a cron job or
Windows Task Scheduler task). It performs the following sequence of operations:
1. Fetches and processes new data batches from the central server.
2. If new data is available or daily training is forced, it initiates the
   AI model fine-tuning process using the HFFineTuner.
3. If a new model adapter is successfully trained, it packages and uploads
   the adapter back to the central server.
"""

import logging
from pathlib import Path
import datetime
import json
import sys
import torch
from typing import Optional

# Ensure the project root is in the system path to allow for package imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import necessary local modules and project classes
try:
    from tower_data_processor import (
        process_new_data_from_server,
        load_tower_config as load_shared_config,
    )
    from tower_model_uploader import create_zip_archive, upload_model_to_server
except ImportError as e:
    logging.critical(
        f"Konnte lokale Module (tower_data_processor, tower_model_uploader) nicht importieren: {e}. "
        "Stellen Sie sicher, dass das 'Starten in'-Verzeichnis der Aufgabe korrekt ist oder der cronjob-Ordner im PYTHONPATH liegt."
    )
    sys.exit(1)

try:
    from Klassen.data.data_manager import DataManager
    from Klassen.services.hf_fine_tuner import HFFineTuner
    from Klassen import config as global_model_config
except ImportError as e:
    logging.critical(
        f"Konnte Klassen-Module (DataManager, HFFineTuner, config) nicht importieren: {e}"
    )
    sys.exit(1)

# Configure logging to file and console
LOG_FILE_PATH = Path(__file__).resolve().parent / "tower_daily_workflow.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH, mode="a"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Define default paths relative to the project root
DEFAULT_ADAPTER_OUTPUT_BASE_DIR = PROJECT_ROOT / "trained_adapters_output"
DEFAULT_ADAPTER_OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_TOWER_DB_PATH_WORKFLOW = PROJECT_ROOT / "tower_code_snippets.db"


def run_fine_tuning_on_tower(
    config_dict: dict, data_manager: DataManager
) -> Optional[Path]:
    """
    Initializes and runs the HFFineTuner to train a new model adapter.

    Args:
        config_dict: A dictionary containing runtime configurations for the tower.
        data_manager: An initialized DataManager instance for the tower's database.

    Returns:
        The path to the newly trained adapter directory if successful, otherwise None.
    """
    if (
        HFFineTuner is None
        or global_model_config is None
        or DataManager is None
        or torch is None
    ):
        logger.error(
            "Eine oder mehrere Kernkomponenten für das Fine-Tuning sind nicht verfügbar."
        )
        return None

    logger.info("Starte Fine-Tuning Prozess auf dem Tower...")
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    adapter_version_name_for_path = f"adapter_ft_{timestamp}"
    adapter_output_dir = DEFAULT_ADAPTER_OUTPUT_BASE_DIR / adapter_version_name_for_path
    adapter_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Prepare Fine-Tuner Configuration ---
    # Merge global defaults with specific overrides from the tower's config file.
    bnb_compute_dtype_str = config_dict.get(
        "bnb_4bit_compute_dtype_str_override",
        str(getattr(global_model_config, "BNB_4BIT_COMPUTE_DTYPE", "torch.float16")),
    )
    bnb_4bit_compute_dtype = (
        torch.bfloat16
        if bnb_compute_dtype_str == "torch.bfloat16"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    fine_tuner_runtime_config = {
        # Model and Training Hyperparameters
        "hf_base_model_name": config_dict.get(
            "hf_base_model_name_override",
            getattr(global_model_config, "HF_BASE_MODEL_NAME"),
        ),
        "hf_num_epochs": int(
            config_dict.get(
                "hf_num_epochs_override", getattr(global_model_config, "HF_NUM_EPOCHS")
            )
        ),
        "hf_batch_size": int(
            config_dict.get(
                "hf_batch_size_override", getattr(global_model_config, "HF_BATCH_SIZE")
            )
        ),
        "hf_grad_accum": int(
            config_dict.get(
                "hf_grad_accum_override", getattr(global_model_config, "HF_GRAD_ACCUM")
            )
        ),
        "hf_max_length": int(
            config_dict.get(
                "hf_max_length_override", getattr(global_model_config, "HF_MAX_LENGTH")
            )
        ),
        "hf_learning_rate": float(
            config_dict.get(
                "hf_learning_rate_override",
                getattr(global_model_config, "HF_LEARNING_RATE"),
            )
        ),
        # Quantization and LoRA settings
        "bnb_load_in_4bit": config_dict.get(
            "bnb_load_in_4bit_override",
            getattr(global_model_config, "BNB_LOAD_IN_4BIT"),
        ),
        "bnb_4bit_quant_type": config_dict.get(
            "bnb_4bit_quant_type_override",
            getattr(global_model_config, "BNB_4BIT_QUANT_TYPE"),
        ),
        "bnb_4bit_compute_dtype": bnb_4bit_compute_dtype,
        "bnb_4bit_use_double_quant": config_dict.get(
            "bnb_4bit_use_double_quant_override",
            getattr(global_model_config, "BNB_4BIT_USE_DOUBLE_QUANT"),
        ),
        "lora_r": int(
            config_dict.get("lora_r_override", getattr(global_model_config, "LORA_R"))
        ),
        "lora_alpha": int(
            config_dict.get(
                "lora_alpha_override", getattr(global_model_config, "LORA_ALPHA")
            )
        ),
        "lora_dropout": float(
            config_dict.get(
                "lora_dropout_override", getattr(global_model_config, "LORA_DROPOUT")
            )
        ),
        "lora_target_modules": config_dict.get(
            "lora_target_modules_override",
            getattr(global_model_config, "LORA_TARGET_MODULES"),
        ),
        # Paths and Data Sources
        "device": config_dict.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        ),
        "lora_adapter_path": str(adapter_output_dir.resolve()),
        "hf_train_with_snippets": config_dict.get("hf_train_with_snippets", True),
        "hf_snippet_training_quality_label": config_dict.get(
            "hf_snippet_training_quality_label", "gut (Training)"
        ),
        "hf_train_with_feedback": config_dict.get("hf_train_with_feedback", True),
        "hf_positive_feedback_types": config_dict.get(
            "hf_positive_feedback_types",
            ["suggestion_used", "correction_applied", "correction_good"],
        ),
    }
    logger.info(
        f"HFFineTuner wird mit folgender Konfiguration initialisiert: {json.dumps(fine_tuner_runtime_config, default=str, indent=2)}"
    )

    fine_tuner = HFFineTuner(
        data_manager=data_manager, config_dict=fine_tuner_runtime_config
    )  # accelerator wird intern geholt

    def training_progress_callback(progress_percent: int):
        if progress_percent == -1:
            logger.error("Trainingsfehler gemeldet.")
        else:
            logger.info(f"Trainingsfortschritt: {progress_percent}%")

    logger.info(f"Speichere trainierten Adapter in: {adapter_output_dir}")
    success, message_or_result = fine_tuner.run_fine_tuning(
        progress_callback=training_progress_callback
    )

    if success:
        logger.info(
            f"Fine-Tuning erfolgreich. Adapter gespeichert unter: {adapter_output_dir}"
        )
        return adapter_output_dir
    else:
        logger.error(f"Fine-Tuning fehlgeschlagen: {message_or_result}")
        return None


def main_workflow():
    """The main entry point for the daily scheduled workflow."""
    logger.info(
        f"=== Starte täglichen Workflow auf dem Tower-PC ({datetime.datetime.now()}) ==="
    )

    if (
        load_shared_config is None
        or process_new_data_from_server is None
        or create_zip_archive is None
        or upload_model_to_server is None
        or DataManager is None
        or HFFineTuner is None
        or global_model_config is None
    ):
        logger.critical(
            "Ein oder mehrere Kernmodule/Funktionen konnten nicht importiert werden. Workflow kann nicht gestartet werden."
        )
        return

    tower_settings = load_shared_config()
    server_api_url = tower_settings.get("server_api_url")
    tower_api_key = tower_settings.get("tower_api_key")
    tower_db_path_str = tower_settings.get(
        "tower_db_path", str(DEFAULT_TOWER_DB_PATH_WORKFLOW.resolve())
    )
    always_run_training_daily = tower_settings.get("always_run_training_daily", True)

    if (
        not all([server_api_url, tower_api_key, tower_db_path_str])
        or tower_api_key == "BITTE_API_KEY_IN_CONFIG_SETZEN"
    ):
        logger.error(
            "Wichtige Konfigurationswerte fehlen in app_settings_tower.json. Workflow abgebrochen."
        )
        return

    tower_db_path = Path(tower_db_path_str)

    logger.info("--- Schritt 1: Verarbeite neue Daten vom Server ---")
    new_data_processed_this_run = process_new_data_from_server()

    run_training_now = always_run_training_daily or new_data_processed_this_run
    trained_adapter_output_path: Optional[Path] = None

    if run_training_now:
        logger.info("--- Schritt 2: Starte Modell-Training ---")
        dm_for_training = DataManager(db_path=tower_db_path)
        if dm_for_training.initialize_database():
            trained_adapter_output_path = run_fine_tuning_on_tower(
                tower_settings, dm_for_training
            )
            dm_for_training.close_connection()
        else:
            logger.error(
                f"Konnte Datenbank '{tower_db_path}' für Training nicht initialisieren."
            )
    else:
        logger.info(
            "Keine neuen Daten verarbeitet oder tägliches Training nicht erzwungen. Überspringe Training für heute."
        )

    if trained_adapter_output_path and trained_adapter_output_path.exists():
        logger.info("--- Schritt 3: Lade neues Modell zum Server hoch ---")
        model_version_name_for_upload = trained_adapter_output_path.name
        temp_zip_file_for_upload = (
            Path(__file__).resolve().parent
            / f"{model_version_name_for_upload}_tower_upload.zip"
        )

        if create_zip_archive(trained_adapter_output_path, temp_zip_file_for_upload):
            upload_success = upload_model_to_server(
                server_url_base=server_api_url,
                api_key=tower_api_key,
                model_zip_path=temp_zip_file_for_upload,
                model_version=model_version_name_for_upload,
                training_date_iso=datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%d"
                ),
                notes=f"Automatisch trainiert ({model_version_name_for_upload}) und hochgeladen vom Tower-PC.",
            )
            if upload_success:
                logger.info(
                    f"Modell '{model_version_name_for_upload}' erfolgreich zum Server hochgeladen."
                )
            else:
                logger.error(
                    f"Upload des Modells '{model_version_name_for_upload}' fehlgeschlagen."
                )
            temp_zip_file_for_upload.unlink(missing_ok=True)
        else:
            logger.error(
                f"Konnte kein ZIP-Archiv für Adapter '{trained_adapter_output_path}' erstellen."
            )
    elif run_training_now and not trained_adapter_output_path:
        logger.warning(
            "Training wurde ausgeführt (oder war geplant), aber kein gültiger Adapterpfad wurde zurückgegeben. Upload übersprungen."
        )

    logger.info(
        f"=== Täglicher Workflow auf dem Tower-PC beendet ({datetime.datetime.now()}) ==="
    )


if __name__ == "__main__":
    main_workflow()
