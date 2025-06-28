# Klassen/inference_service.py
# -*- coding: utf-8 -*-

"""
Verwaltet das Laden von Hugging Face Modellen (Basis und PEFT-Adapter)
für die Inferenz und generiert Code-Vervollständigungsvorschläge.
"""

import traceback
import re
import ast
from pathlib import Path
import logging  # Logging-Modul importiert
from typing import Optional, Tuple, List, Union, Type, Dict, Any

# Importiere Konfiguration und optionale Bibliotheken
from ..core import config as global_config
from .code_corrector import CodeCorrector

# Initialisiere den Logger für dieses Modul
logger = logging.getLogger(__name__)

# --- Optionale Bibliotheken für Typ-Hinweise und Laufzeitprüfungen ---
HF_LIBRARIES_AVAILABLE = False
# Definiere Platzhalter für Typ-Hinweise, wenn die Bibliotheken fehlen
# Diese werden überschrieben, wenn die Bibliotheken erfolgreich importiert werden.
HFAutoModelForCausalLM = type("HFAutoModelForCausalLM", (object,), {})
HFAutoTokenizer = type("HFAutoTokenizer", (object,), {})
BitsAndBytesConfig = type("BitsAndBytesConfig", (object,), {})
HFPeftModel = type("HFPeftModel", (object,), {})
torch: Optional[Type] = None  # Wird später importiert

try:
    import torch
    from transformers import (
        AutoModelForCausalLM as HFAutoModelForCausalLM_import,
        AutoTokenizer as HFAutoTokenizer_import,
        BitsAndBytesConfig as BitsAndBytesConfig_import,
    )
    from peft import PeftModel as HFPeftModel_import

    # Überschreibe die Platzhalter mit den echten Klassen
    HFAutoModelForCausalLM = HFAutoModelForCausalLM_import
    HFAutoTokenizer = HFAutoTokenizer_import
    BitsAndBytesConfig = BitsAndBytesConfig_import
    HFPeftModel = HFPeftModel_import

    HF_LIBRARIES_AVAILABLE = True
    logger.info("Hugging Face Bibliotheken (transformers, peft) erfolgreich geladen.")
except ImportError as e:
    logger.warning(
        f"Eine oder mehrere Hugging Face Bibliotheken konnten nicht importiert werden: {e}. "
        "Inferenz-Funktionalität wird eingeschränkt sein."
    )

try:
    from accelerate import Accelerator

    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    Accelerator = None  # type: ignore für den Fall, dass es nicht importiert wird

logger.debug(
    f"inference_service.py loaded. HF Libraries Available: {HF_LIBRARIES_AVAILABLE}"
)


class InferenceService:
    """
    Verwaltet das Laden von Modellen für die Inferenz und generiert Vorschläge.
    """

    def __init__(self, config_dict: dict, accelerator: Optional["Accelerator"] = None):
        """
        Initialisiert den InferenceService.

        Args:
            config_dict (dict): Das Konfigurationsdictionary.
            accelerator (Optional[Accelerator]): Der Accelerator, falls verwendet.
        """
        if not HF_LIBRARIES_AVAILABLE:
            logger.error(
                "Erforderliche Hugging Face Bibliotheken (transformers, peft) sind nicht installiert. "
                "InferenceService kann nicht korrekt initialisiert werden."
            )
            # Man könnte hier eine Exception werfen oder den Service in einem degradierten Modus lassen.
            # Fürs Erste setzen wir den Status und machen weiter, aber die Funktionalität ist stark eingeschränkt.

        self.config_dict = config_dict
        self.accelerator = accelerator
        self.hf_base_model: Optional[HFAutoModelForCausalLM] = None
        self.hf_peft_model: Optional[HFPeftModel] = None
        self.tokenizer: Optional[HFAutoTokenizer] = None
        self.model_load_status: str = "Nicht geladen"

        logger.info("InferenceService initialisiert.")
        self._load_inference_model()  # Modelle direkt beim Initialisieren laden

    def _ensure_tokenizer_pad_token(self):
        """Stellt sicher, dass der Tokenizer einen Pad-Token hat."""
        if self.tokenizer and self.tokenizer.pad_token is None:
            logger.info("Setze pad_token des Tokenizers auf eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token is None:
                # Fallback, wenn auch eos_token None ist
                logger.warning(
                    "Tokenizer hat keinen eos_token, um ihn als pad_token zu verwenden. "
                    "Setze pad_token_id auf 0. Dies könnte zu unerwartetem Verhalten führen."
                )
                self.tokenizer.pad_token_id = 0

    def _get_compute_dtype(self) -> Optional[torch.dtype]:
        """Ermittelt den torch.dtype aus der Konfiguration."""
        if not torch:  # Sicherstellen, dass torch importiert wurde
            logger.error("PyTorch ist nicht verfügbar, um compute_dtype zu bestimmen.")
            return None

        compute_dtype_str = str(
            self.config_dict.get(
                "bnb_4bit_compute_dtype",
                str(getattr(global_config, "BNB_4BIT_COMPUTE_DTYPE", "torch.float16")),
            )
        )
        compute_dtype_name = compute_dtype_str.split(".")[-1]  # z.B. 'float16'
        return getattr(torch, compute_dtype_name, torch.float16)  # Fallback auf float16

    def _load_base_model_and_tokenizer(
        self, base_model_name: str, quant_config: Optional[BitsAndBytesConfig]
    ) -> Tuple[Optional[HFAutoModelForCausalLM], Optional[HFAutoTokenizer]]:
        """Lädt das Basismodell und den zugehörigen Tokenizer."""
        try:
            logger.info(f"Lade Tokenizer für Basemodell: '{base_model_name}'...")
            tokenizer = HFAutoTokenizer.from_pretrained(
                base_model_name, trust_remote_code=True
            )
            self._ensure_tokenizer_pad_token()  # self.tokenizer wird hier implizit gesetzt
            self.tokenizer = tokenizer  # Explizit setzen für den Rest der Klasse
            logger.info("Tokenizer erfolgreich geladen.")

            device_map_load: Union[str, Dict[str, Any]] = "auto"
            if self.accelerator and hasattr(self.accelerator, "device"):
                device_map_load = {"": str(self.accelerator.device)}

            logger.info(
                f"Lade Basemodell: '{base_model_name}' mit device_map='{device_map_load}'..."
            )
            model = HFAutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quant_config,  # type: ignore
                device_map=device_map_load,
                trust_remote_code=True,
            )
            model.eval()  # In den Evaluationsmodus setzen
            logger.info(
                "Basismodell erfolgreich geladen und in den Eval-Modus versetzt."
            )
            return model, tokenizer
        except Exception as e:
            logger.error(
                f"Fehler beim Laden des Basismodells '{base_model_name}' oder Tokenizers: {e}",
                exc_info=True,
            )
            return None, None

    def _apply_peft_adapter(
        self, base_model: HFAutoModelForCausalLM, adapter_path_str: str
    ) -> Optional[HFPeftModel]:
        """Wendet einen PEFT-Adapter auf ein geladenes Basismodell an."""
        adapter_path = Path(adapter_path_str)
        adapter_config_file = adapter_path / "adapter_config.json"

        if (
            adapter_path.exists()
            and adapter_path.is_dir()
            and adapter_config_file.exists()
        ):
            logger.info(
                f"PEFT-Adapter gefunden unter '{adapter_path}'. Lade und verbinde mit Basismodell..."
            )
            try:
                peft_model = HFPeftModel.from_pretrained(
                    base_model,  # type: ignore
                    str(adapter_path),
                    is_trainable=False,  # Wichtig für Inferenz
                )
                peft_model.eval()  # In den Evaluationsmodus setzen
                logger.info(
                    "PEFT-Adapter erfolgreich geladen und mit Basismodell verbunden."
                )
                return peft_model
            except Exception as e:
                logger.error(
                    f"Fehler beim Laden des PEFT-Adapters von '{adapter_path}': {e}",
                    exc_info=True,
                )
                return None
        else:
            logger.info(
                f"Kein gültiger PEFT-Adapter unter '{adapter_path}' gefunden. "
                "Verwende nur das Basismodell."
            )
            return None

    def _load_inference_model(self) -> None:
        """
        Lädt das Basismodell und optional den LoRA-Adapter für die Inferenz.
        Aktualisiert self.hf_base_model, self.hf_peft_model, self.tokenizer und self.model_load_status.
        """
        self.hf_base_model = None
        self.hf_peft_model = None
        self.tokenizer = None  # Wird in _load_base_model_and_tokenizer gesetzt
        self.model_load_status = "Lade..."
        logger.info("Starte Ladevorgang für Inferenzmodell...")

        if not HF_LIBRARIES_AVAILABLE or not torch:
            self.model_load_status = (
                "Fehler: Kritische Bibliotheken (HF/PyTorch) fehlen."
            )
            logger.error(self.model_load_status)
            return

        base_model_name: str = self.config_dict.get(
            "hf_base_model_name",
            getattr(global_config, "HF_BASE_MODEL_NAME", "default/model"),
        )
        adapter_path_str: str = self.config_dict.get(
            "lora_adapter_path",
            getattr(global_config, "LORA_ADAPTER_PATH", "default/adapter"),
        )

        compute_dtype = self._get_compute_dtype()
        if compute_dtype is None:
            self.model_load_status = (
                "Fehler: PyTorch nicht verfügbar für compute_dtype."
            )
            logger.error(self.model_load_status)
            return

        logger.debug(f"Verwende compute_dtype für Inferenz: {compute_dtype}")

        bnb_quant_config: Optional[BitsAndBytesConfig] = None
        if self.config_dict.get(
            "bnb_load_in_4bit", getattr(global_config, "BNB_LOAD_IN_4BIT", False)
        ):
            bnb_quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config_dict.get(
                    "bnb_4bit_quant_type",
                    getattr(global_config, "BNB_4BIT_QUANT_TYPE", "nf4"),
                ),
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.config_dict.get(
                    "bnb_4bit_use_double_quant",
                    getattr(global_config, "BNB_4BIT_USE_DOUBLE_QUANT", True),
                ),
            )
            logger.info(
                f"BitsAndBytes Quantisierungskonfiguration wird verwendet: {bnb_quant_config}"
            )
        else:
            logger.info(
                "Keine BitsAndBytes Quantisierungskonfiguration verwendet (load_in_4bit ist False)."
            )

        base_model, tokenizer_loaded = self._load_base_model_and_tokenizer(
            base_model_name, bnb_quant_config
        )

        if base_model and tokenizer_loaded:
            self.hf_base_model = base_model
            # self.tokenizer wurde bereits in _load_base_model_and_tokenizer gesetzt

            peft_model_loaded = self._apply_peft_adapter(base_model, adapter_path_str)
            if peft_model_loaded:
                self.hf_peft_model = peft_model_loaded
                self.model_load_status = f"{base_model_name.split('/')[-1]} + LoRA"
            else:
                self.model_load_status = f"{base_model_name.split('/')[-1]} (Basis)"
            logger.info(f"Inferenzmodell geladen. Status: {self.model_load_status}")
        else:
            self.model_load_status = (
                "Fehler beim Laden des Basismodells oder Tokenizers."
            )
            logger.error(self.model_load_status)
            # Sicherstellen, dass alles zurückgesetzt ist bei Fehler
            self.hf_base_model = None
            self.hf_peft_model = None
            self.tokenizer = None

    def get_loaded_model_and_tokenizer(
        self,
    ) -> Tuple[
        Optional[Union[HFAutoModelForCausalLM, HFPeftModel]], Optional[HFAutoTokenizer]
    ]:
        """
        Gibt das aktuell geladene Modell (PEFT oder Basis) und den Tokenizer zurück.
        Priorisiert das PEFT-Modell, wenn verfügbar.
        """
        active_model = self.hf_peft_model if self.hf_peft_model else self.hf_base_model

        if active_model and self.tokenizer:
            return active_model, self.tokenizer
        else:
            logger.warning(
                "Kein Modell oder Tokenizer geladen. Versuche erneutes Laden."
            )
            self._load_inference_model()  # Erneuter Ladeversuch
            active_model_after_reload = (
                self.hf_peft_model if self.hf_peft_model else self.hf_base_model
            )
            if active_model_after_reload and self.tokenizer:
                return active_model_after_reload, self.tokenizer
            else:
                logger.error(
                    "Erneuter Ladeversuch für Modell und Tokenizer fehlgeschlagen."
                )
                return None, None

    def _process_generated_text_for_suggestions(self, generated_text: str) -> List[str]:
        """Verarbeitet den roh generierten Text, um eine Liste von Vorschlägen zu extrahieren."""
        logger.debug(f"Roh generierter Text für Vorschläge: '{generated_text}'")
        processed_text = generated_text.strip()
        if not processed_text:
            logger.debug("Verarbeiteter Text für Vorschläge ist leer.")
            return []

        lines = processed_text.splitlines()
        first_meaningful_line = ""
        for line_content in lines:
            if line_content.strip():
                first_meaningful_line = line_content.strip()
                break

        logger.debug(
            f"Erste bedeutsame Zeile für Vorschläge: '{first_meaningful_line}'"
        )
        if not first_meaningful_line:
            logger.debug("Keine bedeutsame Zeile im generierten Text gefunden.")
            return []

        # Regex für Trennzeichen (maskierter Bindestrich)
        delimiters = r"[\s\(\)\{\}\[\],;:=<>!+/*%&|^~.\-]+"

        potential_items = [
            item for item in re.split(delimiters, first_meaningful_line) if item.strip()
        ]
        logger.debug(f"Potenzielle Items nach Regex-Split: {potential_items}")

        if not potential_items and first_meaningful_line:
            potential_items = [first_meaningful_line.split(" ", 1)[0]]
            logger.debug(f"Fallback für potenzielle Items: {potential_items}")

        valid_suggestions = [
            s.strip()
            for s in potential_items
            if s.strip() and not s.strip().isnumeric()  # Entferne reine Zahlen
        ]
        logger.debug(
            f"Valide (nicht-numerische, gestrippte) Vorschläge: {valid_suggestions}"
        )

        seen = set()
        unique_suggestions = [
            x for x in valid_suggestions if not (x in seen or seen.add(x))
        ]

        max_sugg_count = self.config_dict.get(
            "max_suggestion_count",
            getattr(
                global_config, "MAX_SUGGESTION_COUNT", 5
            ),  # Fallback auf global_config oder Hardcoded
        )
        final_suggestions = unique_suggestions[:max_sugg_count]
        logger.debug(
            f"Finale, einzigartige Vorschläge (max {max_sugg_count}): {final_suggestions}"
        )
        return final_suggestions

    def get_suggestions(
        self, prefix: str, context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        logger.info(
            f"get_suggestions mit Prefix (Länge {len(prefix)}): '{prefix[:50]}...' und Kontext: {bool(context)}"
        )
        model, tokenizer = self.get_loaded_model_and_tokenizer()
        if not model or not tokenizer or not torch:
            return ["[Fehler: Modell nicht bereit]"]

        prompt_parts = []
        # System-Nachricht oder Rollen-Definition am Anfang
        prompt_parts.append(
            "You are a helpful Python coding assistant. Complete the following Python code."
        )

        if context:
            prompt_parts.append("\n### Relevant Code Context from the Current File:")
            imports_ctx_str = context.get("imports_context", "")
            if imports_ctx_str and imports_ctx_str != "Imports: []":
                try:
                    actual_imports_list_str = imports_ctx_str.replace(
                        "Imports: ", ""
                    ).strip()
                    if actual_imports_list_str != "[]":
                        prompt_parts.append(
                            "```python\n# Available imports in the file:"
                        )
                        parsed_list = ast.literal_eval(actual_imports_list_str)
                        for imp_statement in parsed_list:
                            prompt_parts.append(
                                imp_statement.strip()
                            )  # Remove leading/trailing spaces from statement
                        prompt_parts.append("```")
                except Exception as e:
                    logger.debug(
                        f"Konnte imports_context für Suggestion-Prompt nicht parsen: {e}"
                    )
                    prompt_parts.append(f"# Raw Imports: {imports_ctx_str}")

            defs_ctx_str = context.get("definitions_context", "")
            if defs_ctx_str and defs_ctx_str != "Definitions: []":
                try:
                    actual_defs_list_str = defs_ctx_str.replace(
                        "Definitions: ", ""
                    ).strip()
                    if actual_defs_list_str != "[]":
                        prompt_parts.append(
                            "```python\n# Known definitions in this file (summary):"
                        )
                        parsed_defs_list = ast.literal_eval(actual_defs_list_str)
                        for def_repr in parsed_defs_list:
                            match_name = re.search(r"name='([^']*)'", def_repr)
                            match_type = re.search(r"type='([^']*)'", def_repr)
                            match_sig = re.search(r"signature='([^']*)'", def_repr)
                            if match_name and match_type:
                                type_val = match_type.group(1)
                                name_val = match_name.group(1)
                                sig_preview = ""
                                if match_sig and match_sig.group(1) != "None":
                                    # Extrahiere nur den Parameterteil der Signatur
                                    param_part_match = re.search(
                                        r"\((.*)\)", match_sig.group(1)
                                    )
                                    sig_preview = f"({param_part_match.group(1) if param_part_match else '...'})"
                                elif type_val in ["function", "class"]:
                                    sig_preview = "(...)"
                                prompt_parts.append(
                                    f"# {type_val} {name_val}{sig_preview}"
                                )
                            else:
                                prompt_parts.append(
                                    f"# {def_repr[:120].strip()}..."
                                )  # Gekürzt
                        prompt_parts.append("```")
                except Exception as e:
                    logger.debug(
                        f"Konnte definitions_context für Suggestion-Prompt nicht parsen: {e}"
                    )
                    prompt_parts.append(f"# Raw Definitions: {defs_ctx_str}")

            if (
                len(prompt_parts) > 1
            ):  # Check if context was actually added (mehr als nur die System-Nachricht)
                prompt_parts.append("### End of Context")

        prompt_parts.append(
            "\nComplete the following Python code snippet. Provide only the next logical token(s) or a short, relevant code completion.\n```python"
        )
        prompt_parts.append(prefix)
        # KEIN schließendes ``` hier, da das Modell den Code fortsetzen soll.
        # Wir könnten einen Stop-Token wie "\n```" oder "\n\n" verwenden, wenn das Modell dazu neigt, zu viel zu generieren.

        final_prompt = "\n".join(prompt_parts)
        # logger.debug(f"Finaler Prompt für Suggestion (Länge {len(final_prompt)}):\n{final_prompt}") # Für Debugging einkommentieren

        try:
            device = next(model.parameters()).device
            model_max_len = self.config_dict.get("hf_max_length", global_config.HF_MAX_LENGTH)  # type: ignore
            max_new_tokens_suggest = self.config_dict.get(
                "max_new_tokens_suggest", 20
            )  # Oft sind kurze Vervollständigungen besser
            max_prompt_len = model_max_len - max_new_tokens_suggest - 50

            inputs = tokenizer(
                final_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_len,
            ).to(device)
            if inputs["input_ids"].nelement() == 0:
                logger.warning(
                    "Leere input_ids nach Tokenisierung für Suggestion-Prompt."
                )
                return []

            model.eval()
            with torch.no_grad():
                pad_token_id = (
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                )
                eos_token_id = (
                    tokenizer.eos_token_id
                )  # Wichtig, damit das Modell weiß, wann es aufhören soll

                if pad_token_id is None:
                    pad_token_id = 0  # Absoluter Fallback
                # Wenn kein expliziter EOS-Token, könnte das Modell weiter generieren.
                # Manchmal ist es besser, hier keinen Fallback zu pad_token_id zu machen,
                # sondern auf das Modelltraining zu vertrauen oder Stop-Sequenzen zu nutzen.
                # Fürs Erste lassen wir es, aber das ist ein Tuning-Parameter.
                if eos_token_id is None:
                    eos_token_id = pad_token_id

                num_return_sequences = self.config_dict.get(
                    "num_suggestion_sequences", 3
                )
                temperature = self.config_dict.get("suggestion_temperature", 0.2)
                top_k = self.config_dict.get("suggestion_top_k", 10)
                top_p = self.config_dict.get("suggestion_top_p", 0.95)

                output_sequences = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens_suggest,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,  # type: ignore
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    early_stopping=True,  # Stoppt, sobald EOS generiert wird
                    # repetition_penalty=1.1 # Kann helfen, Wiederholungen zu vermeiden
                )

            suggestions = []
            input_length = inputs["input_ids"].shape[1]
            for seq_idx in range(output_sequences.shape[0]):
                generated_tokens = output_sequences[seq_idx, input_length:]
                suggestion_text_raw = tokenizer.decode(
                    generated_tokens, skip_special_tokens=False
                )
                logger.debug(
                    f"Rohe KI-Sequenz {seq_idx} für Suggestion: {repr(suggestion_text_raw)}"
                )

                # Entferne alles nach dem ersten Newline, da wir meist nur eine kurze Vervollständigung wollen
                suggestion_text_cleaned = tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                ).split("\n")[0]

                # Entferne den Prefix, falls die KI ihn wiederholt hat
                if suggestion_text_cleaned.startswith(prefix) and len(
                    suggestion_text_cleaned
                ) > len(prefix):
                    suggestion_text_cleaned = suggestion_text_cleaned[len(prefix):]

                # Entferne gängige Stop-Phrasen oder unvollständige Ausgaben
                stop_phrases = [
                    "```",
                    "...",
                    "<|endoftext|>",
                    tokenizer.eos_token if tokenizer.eos_token else "XXXXXX",
                ]
                for phrase in stop_phrases:
                    if phrase and phrase in suggestion_text_cleaned:
                        suggestion_text_cleaned = suggestion_text_cleaned.split(phrase)[
                            0
                        ]

                suggestion_text_cleaned = suggestion_text_cleaned.strip()

                if suggestion_text_cleaned:
                    suggestions.append(suggestion_text_cleaned)

            unique_suggestions = []
            seen_suggestions = set()
            for sug in suggestions:
                if sug and sug not in seen_suggestions:
                    unique_suggestions.append(sug)
                    seen_suggestions.add(sug)

            max_sugg_to_show = self.config_dict.get("max_suggestion_count_display", 5)
            logger.debug(
                f"Finale einzigartige Vorschläge für Autocomplete: {unique_suggestions[:max_sugg_to_show]}"
            )
            return unique_suggestions[:max_sugg_to_show]

        except Exception as e:
            logger.error(
                f"Fehler während der HF-Vorschlagsgenerierung: {e}", exc_info=True
            )
            return ["[Fehler bei HF Vorschlägen]"]

    def improve_code_snippet(
        self, code_snippet: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        logger.info(
            f"improve_code_snippet für Snippet (Länge {len(code_snippet)}). Kontext vorhanden: {bool(context)}"
        )

        model, tokenizer = self.get_loaded_model_and_tokenizer()
        if not model or not tokenizer or not torch:
            return "[Error: Model not ready]"

        prompt_parts = []
        task_description = "Rewrite the provided Python 'Code to Improve' to be more Pythonic, efficient, robust, or clear."
        file_path_from_context: Optional[str] = None
        json_imported = False
        csv_imported = False  # Hinzugefügt für CSV-Kontext

        if context:
            file_path_from_context = context.get("file_path")
            imports_ctx_str = context.get(
                "imports_context", ""
            )  # z.B. "Imports: [ImportInfo(import=['json'], line=3)]"

            # Überprüfung, ob 'json' oder 'csv' importiert wurde
            # Diese Logik muss möglicherweise robuster sein, abhängig vom genauen Format von imports_ctx_str
            if (
                re.search(r"import=\['json'\]", imports_ctx_str)
                or "'json'" in imports_ctx_str
            ):
                json_imported = True
            if (
                re.search(r"import=\['csv'\]", imports_ctx_str)
                or "'csv'" in imports_ctx_str
            ):
                csv_imported = True  # Hinzugefügt

            if (
                file_path_from_context
                and file_path_from_context.lower().endswith(".json")
                and json_imported
            ):
                task_description = (
                    "The 'Code to Improve' likely deals with a JSON file (as 'json' is imported and filename ends with .json). "
                    "Rewrite the code to use appropriate JSON handling functions (e.g., json.load(), json.dump()) if it's currently using generic file read/write. "
                    "Ensure it's also Pythonic, efficient, and clear."
                )
                logger.info("Tailoring improvement prompt for JSON file handling.")
            elif (
                file_path_from_context
                and file_path_from_context.lower().endswith(".csv")
                and csv_imported
            ):  # Hinzugefügt
                task_description = (
                    "The 'Code to Improve' likely deals with a CSV file (as 'csv' is imported and filename ends with .csv). "
                    "Rewrite the code to use appropriate CSV handling functions (e.g., csv.reader, csv.writer) if it's currently using generic file read/write. "
                    "Ensure it's also Pythonic, efficient, and clear."
                )
                logger.info("Tailoring improvement prompt for CSV file handling.")

        prompt_parts.append(f"TASK: {task_description}")
        prompt_parts.append(
            "IMPORTANT: Your response MUST be ONLY the improved Python code. NO explanations, NO markdown, ONLY code."
        )

        if context:
            prompt_parts.append("\n### File Context (for information only):")
            if file_path_from_context:
                prompt_parts.append(f"# Filename: {file_path_from_context}")

            imports_ctx_str = context.get("imports_context", "")
            if (
                imports_ctx_str and "[]" not in imports_ctx_str
            ):  # Nur hinzufügen, wenn nicht leer
                try:
                    actual_imports_list_str = imports_ctx_str.replace(
                        "Imports: ", ""
                    ).strip()
                    formatted_imports_for_prompt = []
                    if actual_imports_list_str.startswith(
                        "["
                    ) and actual_imports_list_str.endswith("]"):
                        try:
                            eval_list = ast.literal_eval(actual_imports_list_str)
                            for (
                                item_repr_str
                            ) in (
                                eval_list
                            ):  # item_repr_str ist bereits ein String wie "ImportInfo(...)"
                                imp_match = re.search(
                                    r"import=\['([^']*)'\]", item_repr_str
                                )
                                from_imp_match = re.search(
                                    r"from='([^']*)'.*import=\[(.*?)\]", item_repr_str
                                )
                                if imp_match:
                                    formatted_imports_for_prompt.append(
                                        f"import {imp_match.group(1)}"
                                    )
                                elif from_imp_match:
                                    from_module = from_imp_match.group(1)
                                    imported_items = (
                                        from_imp_match.group(2)
                                        .replace("'", "")
                                        .replace('"', "")
                                    )  # Namen aus 'name' oder ('name', 'alias') extrahieren
                                    # Einfache Annahme für die Darstellung:
                                    processed_items = re.sub(
                                        r"\([^)]*\)",
                                        lambda m: m.group(0).split(",")[0][1:].strip(),
                                        imported_items,
                                    )  # ('name', 'alias') -> name
                                    formatted_imports_for_prompt.append(
                                        f"from {from_module} import {processed_items}"
                                    )
                                else:
                                    if len(item_repr_str) < 70:
                                        formatted_imports_for_prompt.append(
                                            f"# {item_repr_str}"
                                        )  # Fallback: rohes, gekürztes Item
                        except Exception:
                            if len(actual_imports_list_str) < 150:
                                formatted_imports_for_prompt.append(
                                    f"# {actual_imports_list_str}"
                                )  # Fallback bei Parsefehler
                    else:  # Kein Listenformat
                        if len(actual_imports_list_str) < 150:
                            formatted_imports_for_prompt.append(
                                f"# {actual_imports_list_str}"
                            )

                    if formatted_imports_for_prompt:
                        prompt_parts.append(
                            f"# Imports: {', '.join(formatted_imports_for_prompt[:3])}{'...' if len(formatted_imports_for_prompt) > 3 else ''}"
                        )

                except Exception as e:
                    logger.debug(f"Minor error formatting imports for prompt: {e}")
                    if len(imports_ctx_str) < 150:
                        prompt_parts.append(
                            f"# Raw Imports Context: {imports_ctx_str}..."
                        )

            defs_ctx_str = context.get("definitions_context", "")
            if (
                defs_ctx_str and "[]" not in defs_ctx_str
            ):  # Nur hinzufügen, wenn nicht leer
                try:
                    actual_defs_list_str = defs_ctx_str.replace(
                        "Definitions: ", ""
                    ).strip()
                    formatted_defs_for_prompt = []
                    if actual_defs_list_str.startswith(
                        "["
                    ) and actual_defs_list_str.endswith("]"):
                        try:
                            eval_list = ast.literal_eval(
                                actual_defs_list_str
                            )  # eval_list enthält Strings wie "DefinitionInfo(...)"
                            for i, def_item_repr_str in enumerate(eval_list):
                                if (
                                    i >= 2 and len(eval_list) > 3
                                ):  # Max 2-3 Definitionen für Kürze
                                    formatted_defs_for_prompt.append("...")
                                    break
                                name_m = re.search(r"name='([^']*)'", def_item_repr_str)
                                type_m = re.search(r"type='([^']*)'", def_item_repr_str)
                                sig_m = re.search(
                                    r"signature='([^']*)'", def_item_repr_str
                                )
                                if name_m and type_m:
                                    sig_preview = ""
                                    if sig_m and sig_m.group(1) != "None":
                                        s_val = sig_m.group(1)
                                        param_match = re.search(r"\((.*?)\)", s_val)
                                        if param_match:
                                            sig_preview = f"({param_match.group(1)})"
                                    formatted_defs_for_prompt.append(
                                        f"{type_m.group(1)} {name_m.group(1)}{sig_preview}"
                                    )
                        except Exception:  # Fallback bei Parsefehler der Liste
                            if len(actual_defs_list_str) < 200:
                                formatted_defs_for_prompt.append(
                                    f"# {actual_defs_list_str}"
                                )
                    else:  # Kein Listenformat
                        if len(actual_defs_list_str) < 200:
                            formatted_defs_for_prompt.append(
                                f"# {actual_defs_list_str}"
                            )

                    if formatted_defs_for_prompt:
                        prompt_parts.append(
                            f"# Definitions: {', '.join(formatted_defs_for_prompt)}"
                        )
                except Exception as e:
                    logger.debug(f"Minor error formatting definitions for prompt: {e}")
                    if len(defs_ctx_str) < 200:
                        prompt_parts.append(
                            f"# Raw Definitions Context: {defs_ctx_str}..."
                        )
            # prompt_parts.append("### End Context") # Eingespart für Kürze

        prompt_parts.append("\n### Code to Improve:\n```python")
        prompt_parts.append(code_snippet.rstrip())
        prompt_parts.append("```")
        prompt_parts.append(
            "\n### Improved Python Code ONLY (do not add explanations or markdown formatting like ```python):\n"
        )

        final_prompt = "\n".join(prompt_parts)

        tokenizer_for_log = tokenizer if tokenizer else None
        try:  # Fallback Tokenizer für Logging, falls Haupt-Tokenizer noch nicht initialisiert
            if not tokenizer_for_log and HF_LIBRARIES_AVAILABLE:
                tokenizer_for_log = HFAutoTokenizer.from_pretrained(self.config_dict.get("hf_base_model_name", getattr(global_config, "HF_BASE_MODEL_NAME", "gpt2")), trust_remote_code=True)  # type: ignore
        except Exception:
            tokenizer_for_log = None  # Sicherstellen, dass es None ist, falls der Ladeversuch fehlschlägt

        prompt_token_count = -1
        if tokenizer_for_log:
            try:
                prompt_token_count = len(tokenizer_for_log.encode(final_prompt))
            except Exception:
                pass  # Ignoriere Fehler beim Encodieren für Logging

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Vollständiger Prompt für Code-Verbesserung (Länge {prompt_token_count} Tokens):\n{final_prompt}"
            )
        else:
            logger.info(
                f"Prompt für Code-Verbesserung erstellt (Länge ca. {prompt_token_count} Tokens)."
            )

        try:
            model.eval()
            device = next(model.parameters()).device  # type: ignore
            model_max_len = self.config_dict.get(
                "hf_max_length", getattr(global_config, "HF_MAX_LENGTH", 1024)
            )

            estimated_snippet_tokens = (
                len(tokenizer.encode(code_snippet))
                if tokenizer
                else len(code_snippet) // 3
            )  # Grobe Schätzung
            max_new_tokens_improve_base = self.config_dict.get(
                "max_new_tokens_improve_base", 25
            )
            max_new_tokens_improve_factor = self.config_dict.get(
                "max_new_tokens_improve_factor", 1.5
            )
            max_new_tokens = int(
                max_new_tokens_improve_base
                + (estimated_snippet_tokens * max_new_tokens_improve_factor)
            )
            max_new_tokens = min(
                max_new_tokens, self.config_dict.get("max_new_tokens_improve_cap", 200)
            )
            max_new_tokens = max(
                max_new_tokens, self.config_dict.get("min_new_tokens_improve", 15)
            )
            logger.debug(
                f"Calculated max_new_tokens for improvement: {max_new_tokens} (snippet tokens: {estimated_snippet_tokens})"
            )

            prompt_buffer_tokens = self.config_dict.get(
                "prompt_buffer_tokens", 70
            )  # Etwas mehr Puffer
            max_prompt_len_for_tokenizer = (
                model_max_len - max_new_tokens - prompt_buffer_tokens
            )

            current_prompt_tokens_val = prompt_token_count if prompt_token_count != -1 and tokenizer else len(tokenizer.encode(final_prompt)) if tokenizer else -1  # type: ignore

            if (
                current_prompt_tokens_val != -1
                and current_prompt_tokens_val > max_prompt_len_for_tokenizer
            ):
                logger.warning(
                    f"Prompt zu lang ({current_prompt_tokens_val} > {max_prompt_len_for_tokenizer}). Versuche Kontext aggressiv zu kürzen."
                )
                # Aggressives Kürzen: Entferne allen Kontext außer Dateiname und Aufgabenbeschreibung
                new_prompt_parts = [
                    prompt_parts[0],
                    prompt_parts[1],
                ]  # TASK und IMPORTANT
                if file_path_from_context:
                    new_prompt_parts.insert(1, f"# Filename: {file_path_from_context}")

                # Füge Code-Snippet und Aufforderung wieder hinzu
                code_to_improve_section_idx = -1
                for i, part_content in enumerate(prompt_parts):
                    if "### Code to Improve:" in part_content:
                        code_to_improve_section_idx = i
                        break
                if code_to_improve_section_idx != -1:
                    new_prompt_parts.extend(prompt_parts[code_to_improve_section_idx:])

                final_prompt = "\n".join(new_prompt_parts)
                current_prompt_tokens_val = len(tokenizer.encode(final_prompt)) if tokenizer else -1  # type: ignore
                logger.info(
                    f"Prompt nach aggressiver Kontextentfernung (Länge {current_prompt_tokens_val} Tokens)."
                )

                if (
                    current_prompt_tokens_val != -1
                    and current_prompt_tokens_val > max_prompt_len_for_tokenizer
                ):
                    logger.error(
                        f"Prompt auch nach aggressiver Kürzung zu lang ({current_prompt_tokens_val} / {max_prompt_len_for_tokenizer})."
                    )
                    return f"[Error: Prompt too long ({current_prompt_tokens_val}/{max_prompt_len_for_tokenizer}) for model to process]"

            inputs = tokenizer(final_prompt, return_tensors="pt", max_length=max_prompt_len_for_tokenizer, truncation=True).to(device)  # type: ignore
            if inputs["input_ids"].nelement() == 0:
                return "[Error: Empty input after tokenization]"  # type: ignore

            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id  # type: ignore
            if pad_id is None:
                pad_id = 0
            eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id  # type: ignore

            with torch.no_grad():  # type: ignore
                output_sequences = model.generate(  # type: ignore
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    do_sample=self.config_dict.get("improvement_do_sample", True),
                    temperature=self.config_dict.get(
                        "improvement_temperature", 0.45
                    ),  # Etwas erhöht für mehr Kreativität
                    top_k=self.config_dict.get("improvement_top_k", 25),
                    top_p=self.config_dict.get("improvement_top_p", 0.92),
                    pad_token_id=pad_id,  # type: ignore
                    eos_token_id=eos_id,  # type: ignore
                    early_stopping=True,
                    repetition_penalty=self.config_dict.get(
                        "improvement_repetition_penalty", 1.05
                    ),
                )

            generated_ids = output_sequences[0][inputs["input_ids"].shape[-1]:]  # type: ignore
            raw_output = tokenizer.decode(generated_ids, skip_special_tokens=False)  # type: ignore
            logger.debug(f"KI Rohausgabe (Verbesserung):\n---\n{raw_output}\n---")

            temp_log_list_for_extraction: List[str] = []
            improved_code = CodeCorrector._extract_code_from_ai_response(raw_output, tokenizer, temp_log_list_for_extraction)  # type: ignore

            for log_entry in temp_log_list_for_extraction:
                logger.debug(f"(Extraktionslog improve_code): {log_entry}")

            # Zusätzliche Bereinigungsschritte
            if improved_code.strip().startswith("```python"):
                improved_code = re.sub(
                    r"^```python\n?", "", improved_code.strip(), count=1
                )
            if improved_code.strip().startswith("```"):
                improved_code = re.sub(
                    r"^```[a-zA-Z]*\n?", "", improved_code.strip(), count=1
                )
            if improved_code.strip().endswith("```"):
                improved_code = improved_code.strip()[:-3].rstrip()

            # Heuristik zum Entfernen von Erklärungen, falls die KI sie trotzdem einfügt
            if (
                "###" in improved_code
                or "Explanation:" in improved_code
                or "Note:" in improved_code
                or "Here's the improved code:" in improved_code
            ):
                logger.debug(
                    "AI response for improvement might contain explanations. Attempting to strip."
                )
                lines = improved_code.splitlines()
                code_lines_only = []
                non_code_keywords = [
                    "Explanation:",
                    "Note:",
                    "Here's the improved code:",
                    "This version",
                ]
                for line_content in lines:
                    stripped_line_lower = line_content.strip().lower()
                    if any(
                        keyword.lower() in stripped_line_lower
                        for keyword in non_code_keywords
                    ):
                        if (
                            len(code_lines_only) > 0
                        ):  # Nur stoppen, wenn schon Code gesammelt wurde
                            break  # Stoppe bei der ersten Erklärungslinie, wenn bereits Code vorhanden ist
                        else:  # Wenn Erklärung am Anfang, überspringe
                            continue
                    if stripped_line_lower.startswith("###"):
                        continue  # Überspringe Markdown-Überschriften

                    code_lines_only.append(line_content)
                potential_clean_code = "\n".join(code_lines_only).strip()

                if (
                    potential_clean_code or not improved_code.strip()
                ):  # Nur verwenden, wenn Ergebnis sinnvoll oder Original leer war
                    improved_code = potential_clean_code

            logger.info(
                f"KI-Vorschlag für Verbesserung (bereinigt, Länge {len(improved_code)}):\n{improved_code}"
            )

            if not improved_code.strip():
                logger.warning("KI-Verbesserungsvorschlag ist nach Bereinigung leer.")
                return "[Info: AI did not suggest a significant change or the suggestion was empty after cleaning.]"

            # Spezifische Überprüfung für JSON-Fall: Wenn f.read() immer noch da ist, aber json.load erwartet wurde.
            if (
                file_path_from_context
                and file_path_from_context.lower().endswith(".json")
                and json_imported
            ):
                if (
                    "f.read()" in improved_code
                    and "json.load" not in improved_code.lower()
                ):
                    logger.warning(
                        "KI hat für JSON-Datei und Import 'f.read()' beibehalten statt 'json.load()' vorzuschlagen. Möglicherweise ist der Prompt immer noch nicht spezifisch genug oder das Modell bevorzugt die kürzere Variante."
                    )
                    # Hier könnte man entscheiden, ob man den Vorschlag trotzdem annimmt oder eine spezifischere Fehlermeldung gibt.
                    # Fürs Erste wird der Vorschlag der KI zurückgegeben.

            return improved_code

        except Exception as e:
            logger.error(
                f"Fehler während der HF-Verbesserungsgenerierung: {e}", exc_info=True
            )
            # Detailliertere Fehlermeldung, falls Traceback verfügbar
            tb_str = traceback.format_exc()
            return f"[Error during AI improvement: {e}\nTraceback:\n{tb_str[:500]}...]"

    def get_status(self) -> str:
        return self.model_load_status
