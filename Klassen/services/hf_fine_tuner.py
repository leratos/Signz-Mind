# Klassen/hf_fine_tuner.py
# -*- coding: utf-8 -*-

"""
Verantwortlich für das Fine-Tuning eines Hugging Face Modells mit QLoRA.
Diese Klasse kapselt die Logik zum Laden des Modells, Vorbereiten der Daten,
Durchführen des Trainings und Speichern des trainierten Adapters.
NEU: Nutzt Feedback-Daten aus dem DataManager und "gut (Training)" Snippets für das Training.
"""

import gc
import logging
from pathlib import Path
from typing import (
    Optional,
    Type,
    Callable,
    List,
    Tuple,
    Union,
    Dict,
    Any,
    TYPE_CHECKING,
)

# Importiere Konfiguration und abhängige Dienste/Klassen
from ..core import config as global_config

if TYPE_CHECKING:
    from ..data.data_manager import DataManager
    from accelerate import Accelerator
    from transformers import (
        AutoModelForCausalLM as HFAutoModelForCausalLM_typing,
        AutoTokenizer as HFAutoTokenizer_typing,
        Trainer as Trainer_typing,
    )
    from peft import PeftModel as HFPeftModel_typing
    from torch.utils.data import Dataset as TorchDataset_typing
    import torch as torch_typing


# Initialisiere den Logger für dieses Modul
logger = logging.getLogger(__name__)

# --- Optionale Bibliotheken für Typ-Hinweise und Laufzeitprüfungen ---
HF_LIBRARIES_AVAILABLE = False
torch: Optional[Type] = None
TorchDataset: Optional[Type] = None
HFAutoModelForCausalLM: Optional[Type] = None
HFAutoTokenizer: Optional[Type] = None
Trainer: Optional[Type] = None
TrainingArguments: Optional[Type] = None
BitsAndBytesConfig: Optional[Type] = None
DataCollatorForLanguageModeling: Optional[Type] = None
LoraConfig: Optional[Type] = None
get_peft_model: Optional[Callable] = None
prepare_model_for_kbit_training: Optional[Callable] = None
HFPeftModel: Optional[Type] = None


try:
    import torch
    from torch.utils.data import Dataset as TorchDataset_import

    TorchDataset = TorchDataset_import

    from transformers import (
        AutoModelForCausalLM as HFAutoModelForCausalLM_import,
        AutoTokenizer as HFAutoTokenizer_import,
        Trainer as Trainer_import,
        TrainingArguments as TrainingArguments_import,
        BitsAndBytesConfig as BitsAndBytesConfig_import,
        DataCollatorForLanguageModeling as DataCollatorForLanguageModeling_import,
    )

    HFAutoModelForCausalLM = HFAutoModelForCausalLM_import
    HFAutoTokenizer = HFAutoTokenizer_import
    Trainer = Trainer_import
    TrainingArguments = TrainingArguments_import
    BitsAndBytesConfig = BitsAndBytesConfig_import
    DataCollatorForLanguageModeling = DataCollatorForLanguageModeling_import

    from peft import (
        LoraConfig as LoraConfig_import,
        get_peft_model as get_peft_model_import,
        prepare_model_for_kbit_training as prepare_model_for_kbit_training_import,
        PeftModel as HFPeftModel_import,
    )

    LoraConfig = LoraConfig_import
    get_peft_model = get_peft_model_import
    prepare_model_for_kbit_training = prepare_model_for_kbit_training_import
    HFPeftModel = HFPeftModel_import

    import bitsandbytes

    HF_LIBRARIES_AVAILABLE = True
    logger.info(
        "Alle Hugging Face relevanten Bibliotheken (inkl. bitsandbytes) wurden erfolgreich geladen."
    )
except ImportError as e:
    logger.warning(
        f"Eine oder mehrere kritische Bibliotheken konnten nicht importiert werden: {e}. "
        "Funktionalitäten, die auf Transformers, PEFT oder bitsandbytes basieren, werden nicht verfügbar sein."
    )

logger.debug(
    f"hf_fine_tuner.py loaded. HF Libraries Available: {HF_LIBRARIES_AVAILABLE}"
)


class HFFineTuner:
    """
    Kapselt die Logik für das QLoRA Fine-Tuning von Hugging Face Modellen.
    """

    def __init__(
        self,
        data_manager: "DataManager",
        config_dict: dict,
        accelerator: Optional["Accelerator"] = None,
    ):
        """
        Initialisiert den HFFineTuner.

        Args:
            data_manager: Der DataManager zum Laden der Trainingsdaten.
            config_dict: Das Konfigurationsdictionary.
            accelerator: Der Accelerator, falls verwendet.
        """
        if (
            not HF_LIBRARIES_AVAILABLE
            or not torch
            or not TorchDataset
            or not HFAutoModelForCausalLM
            or not HFAutoTokenizer
            or not Trainer
            or not TrainingArguments
            or not BitsAndBytesConfig
            or not DataCollatorForLanguageModeling
            or not LoraConfig
            or not get_peft_model
            or not prepare_model_for_kbit_training
            or not HFPeftModel
        ):
            logger.error(
                "HFFineTuner: Nicht alle erforderlichen Bibliothekskomponenten sind verfügbar. "
                "Fine-Tuning wird nicht möglich sein."
            )

        self.data_manager = data_manager
        self.config_dict = config_dict
        self.accelerator = accelerator
        self.tokenizer: Optional["HFAutoTokenizer_typing"] = None
        self.peft_model_for_training: Optional["HFPeftModel_typing"] = None
        logger.info("HFFineTuner initialisiert.")

    def _get_compute_dtype(self) -> Optional["torch_typing.dtype"]:
        """Ermittelt den torch.dtype aus der Konfiguration."""
        if not torch:
            return None
        compute_dtype_str = str(
            self.config_dict.get(
                "bnb_4bit_compute_dtype",
                str(getattr(global_config, "BNB_4BIT_COMPUTE_DTYPE", "torch.float16")),
            )
        )
        compute_dtype_name = compute_dtype_str.split(".")[-1]
        return getattr(torch, compute_dtype_name, torch.float16)

    def _ensure_tokenizer_pad_token(self):
        """Stellt sicher, dass der Tokenizer einen Pad-Token hat."""
        if self.tokenizer and self.tokenizer.pad_token is None:
            logger.info("Setze pad_token des Tokenizers auf eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token is None:
                logger.warning(
                    "Tokenizer hat keinen eos_token, um ihn als pad_token zu verwenden. "
                    "Setze pad_token_id auf 0. Dies könnte zu unerwartetem Verhalten führen."
                )
                self.tokenizer.pad_token_id = 0

    def _prepare_training_data(self) -> Optional[List[str]]:
        """
        Lädt und bereitet Trainingsdaten aus Snippets und KI-Feedback vor.
        """
        logger.info(
            "Bereite Trainingsdaten vor (kombiniert aus Snippets und Feedback)..."
        )
        texts_list: List[str] = []

        # --- 1. Lade "gut (Training)" Snippets ---
        # Die Konfigurationsschlüssel hier sind Beispiele und können angepasst werden.
        use_snippets_data = self.config_dict.get("hf_train_with_snippets", True)

        if use_snippets_data:
            snippet_quality_label = self.config_dict.get(
                "hf_snippet_training_quality_label", "gut (Training)"
            )
            logger.info(f"Lade Code-Snippets mit Qualität '{snippet_quality_label}'...")

            snippets_joined_text = self.data_manager.get_all_snippets_text(
                quality_filter=[snippet_quality_label]
            )
            if snippets_joined_text and snippets_joined_text.strip():
                # Annahme: get_all_snippets_text liefert einen String, der durch "\n\n" getrennt ist
                # oder eine Liste von Strings, je nach Implementierung.
                # Hier gehen wir von einem durch "\n\n" getrennten String aus.
                snippet_items = [
                    s.strip() for s in snippets_joined_text.split("\n\n") if s.strip()
                ]
                texts_list.extend(snippet_items)
                logger.info(
                    f"{len(snippet_items)} Snippets zur Trainingsdatenliste hinzugefügt."
                )
            else:
                logger.info(
                    f"Keine Snippets mit Qualität '{snippet_quality_label}' gefunden."
                )
        else:
            logger.info(
                "Training mit Code-Snippets ist deaktiviert (hf_train_with_snippets=False)."
            )

        # --- 2. Lade Feedback-Daten ---
        use_feedback_data = self.config_dict.get("hf_train_with_feedback", True)

        if use_feedback_data:
            positive_feedback_types = self.config_dict.get(
                "hf_positive_feedback_types",
                ["suggestion_used", "correction_applied", "correction_good"],
            )
            logger.info(f"Lade Feedback-Daten mit Typen '{positive_feedback_types}'...")

            feedback_data_dicts = self.data_manager.get_feedback_data_for_training(
                feedback_types=positive_feedback_types
                # action_types könnten hier auch gefiltert werden, falls gewünscht
            )

            if feedback_data_dicts:
                feedback_texts_count = 0
                for item in feedback_data_dicts:
                    prompt = item.get("prompt", "")
                    completion = item.get("completion", "")
                    if prompt and completion:
                        # Kombiniere Prompt und Completion zu einem einzelnen Trainingsbeispiel.
                        # Das Modell lernt, die Completion nach dem Prompt vorherzusagen.
                        texts_list.append(f"{prompt}{completion}")
                        feedback_texts_count += 1
                    else:
                        logger.debug(
                            f"Überspringe Feedback-Eintrag aufgrund fehlendem Prompt oder Completion: {item}"
                        )
                logger.info(
                    f"{feedback_texts_count} Feedback-basierte Texte zur Trainingsdatenliste hinzugefügt."
                )
            else:
                logger.info(
                    f"Keine Feedback-Daten mit Typen '{positive_feedback_types}' gefunden."
                )
        else:
            logger.info(
                "Training mit Feedback-Daten ist deaktiviert (hf_train_with_feedback=False)."
            )

        if not texts_list:
            logger.warning(
                "Keine Trainingsdaten (weder Snippets noch Feedback) für das Fine-Tuning gefunden."
            )
            return None

        logger.info(
            f"Insgesamt {len(texts_list)} Textsegmente für das Training vorbereitet."
        )
        return texts_list

    def _load_tokenizer(self) -> bool:
        """Lädt den Tokenizer für das Basismodell."""
        if not HFAutoTokenizer:
            return False
        base_model_name = self.config_dict.get(
            "hf_base_model_name", global_config.HF_BASE_MODEL_NAME
        )
        try:
            logger.info(f"Lade Tokenizer für Basemodell: '{base_model_name}'...")
            self.tokenizer = HFAutoTokenizer.from_pretrained(
                base_model_name, trust_remote_code=True
            )
            self._ensure_tokenizer_pad_token()
            logger.info("Tokenizer erfolgreich geladen.")
            return True
        except Exception as e:
            logger.error(
                f"Fehler beim Laden des Tokenizers '{base_model_name}': {e}",
                exc_info=True,
            )
            self.tokenizer = None
            return False

    def _load_base_model(
        self, compute_dtype: Optional["torch_typing.dtype"]
    ) -> Optional["HFAutoModelForCausalLM_typing"]:
        """Lädt das Basismodell mit Quantisierungskonfiguration."""
        if not HFAutoModelForCausalLM or not BitsAndBytesConfig or not compute_dtype:
            logger.error(
                "Benötigte Klassen (HFAutoModelForCausalLM, BitsAndBytesConfig) oder compute_dtype nicht verfügbar."
            )
            return None

        base_model_name = self.config_dict.get(
            "hf_base_model_name", global_config.HF_BASE_MODEL_NAME
        )

        bnb_quant_config = BitsAndBytesConfig(
            load_in_4bit=self.config_dict.get(
                "bnb_load_in_4bit", global_config.BNB_LOAD_IN_4BIT
            ),
            bnb_4bit_quant_type=self.config_dict.get(
                "bnb_4bit_quant_type", global_config.BNB_4BIT_QUANT_TYPE
            ),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.config_dict.get(
                "bnb_4bit_use_double_quant", global_config.BNB_4BIT_USE_DOUBLE_QUANT
            ),
        )

        device_map_load = "auto"
        if self.accelerator and self.accelerator.distributed_type != "NO":
            device_map_load = {"": self.accelerator.device}
            logger.info(
                f"Using device_map: {device_map_load} due to distributed Accelerator."
            )
        elif self.accelerator:
            logger.info(
                "Accelerator detected (non-distributed), device_map='auto' will be used for model loading."
            )

        logger.info(
            f"Lade Basemodell: '{base_model_name}' mit device_map='{device_map_load}' und Quantisierung..."
        )
        try:
            model = HFAutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_quant_config,
                trust_remote_code=True,
                device_map=device_map_load,
            )
            logger.info("Basismodell erfolgreich geladen.")
            return model
        except Exception as e:
            logger.error(
                f"Fehler beim Laden des Basismodells '{base_model_name}': {e}",
                exc_info=True,
            )
            return None

    def _prepare_model_for_peft(
        self, model: "HFAutoModelForCausalLM_typing"
    ) -> Optional["HFPeftModel_typing"]:
        """Bereitet das Modell für PEFT (QLoRA) vor."""
        if (
            not prepare_model_for_kbit_training
            or not LoraConfig
            or not get_peft_model
            or not HFPeftModel
        ):
            logger.error("Benötigte PEFT-Funktionen/Klassen nicht verfügbar.")
            return None
        try:
            logger.info("Bereite Modell für k-bit Training vor (PEFT)...")
            model.gradient_checkpointing_enable()
            model_prepared = prepare_model_for_kbit_training(model)

            lora_config_peft = LoraConfig(
                r=self.config_dict.get("lora_r", global_config.LORA_R),
                lora_alpha=self.config_dict.get("lora_alpha", global_config.LORA_ALPHA),
                target_modules=self.config_dict.get(
                    "lora_target_modules", global_config.LORA_TARGET_MODULES
                ),
                lora_dropout=self.config_dict.get(
                    "lora_dropout", global_config.LORA_DROPOUT
                ),
                bias=self.config_dict.get("lora_bias", global_config.LORA_BIAS),
                task_type=self.config_dict.get(
                    "lora_task_type", global_config.LORA_TASK_TYPE
                ),
            )
            logger.info("Erstelle PEFT-Modell...")
            peft_model = get_peft_model(model_prepared, lora_config_peft)
            peft_model.print_trainable_parameters()
            logger.info("PEFT-Modell erfolgreich erstellt.")
            return peft_model
        except Exception as e:
            logger.error(
                f"Fehler bei der Vorbereitung des Modells mit PEFT: {e}", exc_info=True
            )
            return None

    def _create_hf_dataset(
        self, texts_list: List[str], max_length: int
    ) -> Optional["TorchDataset_typing"]:
        """Erstellt ein Hugging Face Dataset aus den Textdaten."""
        if not self.tokenizer or not TorchDataset:
            logger.error(
                "Tokenizer oder PyTorch Dataset-Klasse nicht verfügbar für Dataset-Erstellung."
            )
            return None

        class HFTextDatasetForTuning(TorchDataset):
            def __init__(self, tokenizer_param, texts_param, max_length_param):
                self.tokenizer = tokenizer_param
                self.texts = texts_param
                self.max_length = max_length_param
                self.examples: List[Dict[str, Any]] = []
                logger.debug(
                    f"HFTextDatasetForTuning: Tokenisiere {len(self.texts)} Texte..."
                )
                for i, text_content in enumerate(self.texts):
                    if not text_content.strip():
                        continue
                    tokenized_output = self.tokenizer(
                        text_content,
                        truncation=True,
                        max_length=self.max_length,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    self.examples.append(
                        {
                            "input_ids": tokenized_output["input_ids"].squeeze(0),
                            "attention_mask": tokenized_output[
                                "attention_mask"
                            ].squeeze(0),
                            "labels": tokenized_output["input_ids"].squeeze(0).clone(),
                        }
                    )
                    if (i + 1) % 200 == 0:
                        logger.debug(
                            f"HFTextDatasetForTuning: Tokenisiert {i+1}/{len(self.texts)}"
                        )
                logger.info(
                    f"HFTextDatasetForTuning: Tokenisierung abgeschlossen. {len(self.examples)} Beispiele erstellt."
                )

            def __len__(self):
                return len(self.examples)

            def __getitem__(self, i):
                return self.examples[i]

        try:
            hf_dataset = HFTextDatasetForTuning(self.tokenizer, texts_list, max_length)
            if not hf_dataset or len(hf_dataset) == 0:
                logger.warning(
                    "Erstelltes Hugging Face Dataset ist leer oder Erstellung fehlgeschlagen."
                )
                return None
            logger.info(
                f"Hugging Face Dataset mit {len(hf_dataset)} Beispielen erstellt."
            )
            return hf_dataset
        except Exception as e:
            logger.error(
                f"Fehler bei Tokenisierung oder HF Dataset Erstellung: {e}",
                exc_info=True,
            )
            return None

    def _setup_trainer(
        self,
        model_to_train: Union["HFAutoModelForCausalLM_typing", "HFPeftModel_typing"],
        hf_dataset: "TorchDataset_typing",
        compute_dtype: Optional["torch_typing.dtype"],
    ) -> Optional["Trainer_typing"]:
        """Konfiguriert und erstellt die Trainer-Instanz."""
        if (
            not self.tokenizer
            or not TrainingArguments
            or not Trainer
            or not DataCollatorForLanguageModeling
            or not torch
        ):
            logger.error("Benötigte Klassen für Trainer-Setup nicht verfügbar.")
            return None

        adapter_path = Path(
            self.config_dict.get("lora_adapter_path", global_config.LORA_ADAPTER_PATH)
        )

        dataset_len = len(hf_dataset) if hf_dataset else 1
        batch_size_eff = self.config_dict.get(
            "hf_batch_size", global_config.HF_BATCH_SIZE
        ) * self.config_dict.get("hf_grad_accum", global_config.HF_GRAD_ACCUM)
        if batch_size_eff == 0:
            batch_size_eff = 1

        training_args_dict = {
            "output_dir": str(adapter_path),
            "num_train_epochs": self.config_dict.get(
                "hf_num_epochs", global_config.HF_NUM_EPOCHS
            ),
            "per_device_train_batch_size": self.config_dict.get(
                "hf_batch_size", global_config.HF_BATCH_SIZE
            ),
            "gradient_accumulation_steps": self.config_dict.get(
                "hf_grad_accum", global_config.HF_GRAD_ACCUM
            ),
            "learning_rate": self.config_dict.get(
                "hf_learning_rate", global_config.HF_LEARNING_RATE
            ),
            "logging_steps": max(1, int(dataset_len / (batch_size_eff * 10))),
            "save_strategy": "epoch",
            "report_to": "none",
            "save_total_limit": 2,
        }

        if not self.accelerator:
            if compute_dtype == torch.bfloat16 and torch.cuda.is_bf16_supported():
                training_args_dict["bf16"] = True
                logger.info(
                    "Verwende bfloat16 für Training (TrainingArguments, kein Accelerator)."
                )
            elif compute_dtype == torch.float16:
                training_args_dict["fp16"] = True
                logger.info(
                    "Verwende float16 für Training (TrainingArguments, kein Accelerator)."
                )
        else:
            logger.info(
                "Accelerator ist vorhanden. FP16/BF16-Einstellung wird vom Accelerator erwartet."
            )

        training_args = TrainingArguments(**training_args_dict)
        logger.debug(f"TrainingArguments: {training_args}")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        trainer = Trainer(
            model=model_to_train,
            args=training_args,
            train_dataset=hf_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        logger.info("Trainer erfolgreich konfiguriert.")
        return trainer

    def _train_and_save_model(
        self,
        trainer: "Trainer_typing",
        model_to_save_after_training: Union[
            "HFAutoModelForCausalLM_typing", "HFPeftModel_typing"
        ],
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Tuple[bool, Union[dict, str]]:
        """Führt das Training durch und speichert das Modell/Adapter."""
        adapter_path = Path(
            self.config_dict.get("lora_adapter_path", global_config.LORA_ADAPTER_PATH)
        )
        try:
            if adapter_path.exists() and any(adapter_path.iterdir()):
                logger.warning(
                    f"Adapter-Pfad {adapter_path} ist nicht leer. Bestehende Dateien könnten überschrieben werden."
                )

            logger.info("Starte Trainer.train()...")
            trainer.train()
            logger.info("Fine-Tuning abgeschlossen.")

            logger.info("Speichere Modell/Adapter...")
            final_model_to_save = model_to_save_after_training
            if self.accelerator and final_model_to_save is not None:
                unwrapped_model = self.accelerator.unwrap_model(final_model_to_save)
                if unwrapped_model is not None:
                    final_model_to_save = unwrapped_model
                else:
                    logger.warning(
                        "accelerator.unwrap_model returned None. Saving original model instance."
                    )

            if final_model_to_save is not None and hasattr(
                final_model_to_save, "save_pretrained"
            ):
                adapter_path.mkdir(parents=True, exist_ok=True)
                final_model_to_save.save_pretrained(str(adapter_path))
                logger.info(f"LoRA-Adapter erfolgreich in {adapter_path} gespeichert.")

                if self.tokenizer:
                    self.tokenizer.save_pretrained(str(adapter_path))
                    logger.info(f"Tokenizer ebenfalls in {adapter_path} gespeichert.")

                if progress_callback:
                    progress_callback(100)
                return True, {
                    "message": f"QLoRA Fine-Tuning abgeschlossen. Adapter gespeichert in {adapter_path}.",
                    "adapter_saved": True,
                }
            else:
                msg = "Modell zum Speichern ist None oder hat keine save_pretrained Methode."
                logger.error(msg)
                if progress_callback:
                    progress_callback(-1)
                return False, msg

        except Exception as e:
            msg = f"Fehler während Trainer.train() oder beim Speichern: {e}"
            logger.error(msg, exc_info=True)
            if progress_callback:
                progress_callback(-1)
            return False, msg

    def run_fine_tuning(
        self, progress_callback: Optional[Callable[[int], None]] = None
    ) -> Tuple[bool, Union[dict, str]]:
        """
        Führt den QLoRA Fine-Tuning Prozess durch. Orchestriert die einzelnen Schritte.
        """
        logger.info("Starte Fine-Tuning Prozess...")
        if progress_callback:
            progress_callback(0)

        if not HF_LIBRARIES_AVAILABLE:
            msg = "Fine-Tuning nicht möglich: Erforderliche Bibliotheken fehlen."
            logger.error(msg)
            if progress_callback:
                progress_callback(-1)
            return False, msg

        # 1. Daten vorbereiten (jetzt potenziell kombiniert)
        texts_list = self._prepare_training_data()
        if texts_list is None or not texts_list:
            if progress_callback:
                progress_callback(-1)
            return (
                False,
                "Fehler bei der Vorbereitung der Trainingsdaten oder keine Daten (Snippets/Feedback) gefunden.",
            )

        # 2. Tokenizer laden
        if not self._load_tokenizer() or not self.tokenizer:
            if progress_callback:
                progress_callback(-1)
            return False, "Fehler beim Laden des Tokenizers."
        if progress_callback:
            progress_callback(10)

        # 3. Basismodell laden
        compute_dtype = self._get_compute_dtype()
        if compute_dtype is None:
            if progress_callback:
                progress_callback(-1)
            return False, "Fehler: PyTorch nicht verfügbar für compute_dtype."

        base_model = self._load_base_model(compute_dtype)
        if base_model is None:
            if progress_callback:
                progress_callback(-1)
            return False, "Fehler beim Laden des Basismodells."
        if progress_callback:
            progress_callback(25)

        # 4. Modell für PEFT vorbereiten
        self.peft_model_for_training = self._prepare_model_for_peft(base_model)
        if self.peft_model_for_training is None:
            if progress_callback:
                progress_callback(-1)
            return False, "Fehler bei der Vorbereitung des Modells für PEFT."
        model_to_pass_to_trainer = self.peft_model_for_training
        if progress_callback:
            progress_callback(40)

        # 5. Dataset erstellen
        max_length = self.config_dict.get("hf_max_length", global_config.HF_MAX_LENGTH)
        hf_dataset = self._create_hf_dataset(texts_list, max_length)
        if hf_dataset is None:
            if progress_callback:
                progress_callback(-1)
            return False, "Fehler bei der Erstellung des Hugging Face Datasets."
        if progress_callback:
            progress_callback(55)

        # 6. Trainer einrichten
        trainer = self._setup_trainer(
            model_to_pass_to_trainer, hf_dataset, compute_dtype
        )
        if trainer is None:
            if progress_callback:
                progress_callback(-1)
            return False, "Fehler beim Einrichten des Trainers."
        if progress_callback:
            progress_callback(70)

        # 7. Training durchführen und Modell speichern
        success, result = False, "Initialer Fehler vor Trainingsstart"
        try:
            success, result = self._train_and_save_model(
                trainer, model_to_pass_to_trainer, progress_callback
            )
        finally:
            logger.debug("Ressourcenbereinigung nach Fine-Tuning Versuch...")
            del texts_list, base_model, hf_dataset, trainer
            if hasattr(self, "peft_model_for_training"):
                del self.peft_model_for_training
                self.peft_model_for_training = None

            gc.collect()
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug("Ressourcenbereinigung abgeschlossen.")

        return success, result
