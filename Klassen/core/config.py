# Klassen/config.py
# -*- coding: utf-8 -*-

"""
Zentrale Konfigurationswerte für die Signz-Mind Anwendung.
"""

import torch
import logging

logger = logging.getLogger(__name__)
# --- Pfade ---
DB_PATH = "code_snippets.db"
# TINYLM_CHECKPOINT_PATH = "tiny_model.pt" # Entfernt
LORA_ADAPTER_PATH = "codellama-7b-lora-adapters"

# --- Hugging Face Code Llama Konfiguration ---
HF_BASE_MODEL_NAME = "codellama/CodeLlama-7b-hf"

# --- Hugging Face QLoRA Fine-Tuning Parameter ---
HF_NUM_EPOCHS = 1
HF_BATCH_SIZE = 1  # Klein halten für Laptops, ggf. anpassen
HF_GRAD_ACCUM = 4  # Effektive Batch-Größe = HF_BATCH_SIZE * HF_GRAD_ACCUM
HF_MAX_LENGTH = 1024  # Maximale Sequenzlänge für das Training
HF_LEARNING_RATE = 2e-4

# QLoRA / bitsandbytes Konfiguration
BNB_LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"  # "nf4" (NormalFloat4) oder "fp4" (FloatPoint4)
BNB_4BIT_COMPUTE_DTYPE = (
    torch.float16
)  # torch.bfloat16 (wenn unterstützt) oder torch.float16
BNB_4BIT_USE_DOUBLE_QUANT = True  # Doppelte Quantisierung für bessere Präzision

# LoRA Konfiguration
LORA_R = 16  # Rang der LoRA-Matrizen
LORA_ALPHA = 32  # Alpha-Skalierungsfaktor für LoRA
LORA_DROPOUT = 0.05  # Dropout-Rate in LoRA-Layern
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]  # Zielmodule für LoRA-Adapter, modellspezifisch
LORA_BIAS = "none"  # "none", "all", oder "lora_only"
LORA_TASK_TYPE = "CAUSAL_LM"  # Aufgabe des Modells

# --- Geräte-Konfiguration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Linter Konfiguration ---
FLAKE8_PATH = "flake8"  # Pfad zur Flake8-Executable oder einfach "flake8", wenn im PATH

DEFAULT_SERVER_API_URL = "https://api.last-strawberry.com"
DEFAULT_CLIENT_API_KEY = "PLEASE_CONFIGURE_CLIENT_API_KEY"
DEFAULT_CLIENT_SYNC_ID = "default_client"

logger.info(f"DEBUG: config.py loaded. DEVICE set to: {DEVICE}")
logger.info(f"DEBUG: Using Base Model: {HF_BASE_MODEL_NAME}")
logger.info(f"DEBUG: Using BNB Compute Dtype: {BNB_4BIT_COMPUTE_DTYPE}")
