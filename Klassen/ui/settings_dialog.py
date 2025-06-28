# Klassen/settings_dialog.py
# -*- coding: utf-8 -*-

"""
Definiert den Einstellungsdialog für die Anwendung.
Ermöglicht das Anpassen von Laufzeitparametern für verschiedene Komponenten.
"""

import logging
from typing import Dict, Any, Optional

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QDialogButtonBox,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QWidget,
    QScrollArea,
)

# Qt wird nicht direkt benötigt, aber QDialogButtonBox.StandardButton wird verwendet
# from PySide6.QtCore import Qt

logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """
    Ein Dialog zum Anzeigen und Ändern von Anwendungseinstellungen.
    Die Einstellungen werden in einem Dictionary verwaltet.
    """

    def __init__(self, current_settings: Dict[str, Any], parent=None):
        """
        Initialisiert den Dialog.

        Args:
            current_settings: Ein Dictionary mit den aktuellen Einstellungen.
                              Änderungen werden in eine Kopie dieses Dictionaries geschrieben.
            parent: Das übergeordnete Widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Einstellungen")
        self.setMinimumWidth(450)
        logger.debug("SettingsDialog initialisiert.")

        # Kopie der Einstellungen, um Änderungen verwerfen zu können
        self.settings = current_settings.copy()
        self.widgets: Dict[str, QWidget] = {}  # Zum Speichern der Eingabe-Widgets

        main_layout = QVBoxLayout(self)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        form_widget = QWidget()
        self.form_layout = QFormLayout(
            form_widget
        )  # Formularlayout als Attribut speichern
        scroll_area.setWidget(form_widget)
        main_layout.addWidget(scroll_area)

        self._add_setting_widgets()  # Widgets dynamisch hinzufügen

        # OK / Abbrechen Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    def _add_setting_widgets(self):
        """Fügt die Widgets für die einzelnen Einstellungen zum Formularlayout hinzu."""
        # Beispielhafte Struktur für das Hinzufügen von Einstellungen.
        # Dies könnte aus einer Konfigurationsdatei oder einer Liste von Tupeln generiert werden.

        # Hugging Face Fine-Tuning Parameter
        self.form_layout.addRow(QLabel("<b><u>Hugging Face Fine-Tuning:</u></b>"))
        self._add_double_spin_box_setting(
            "hf_learning_rate",
            "Lernrate (HF):",
            decimals=6,
            step=0.000001,
            min_val=0.000001,
            max_val=0.1,
        )
        self._add_spin_box_setting(
            "hf_num_epochs", "Epochen (HF):", min_val=1, max_val=100
        )
        self._add_spin_box_setting(
            "hf_batch_size", "Batch Size pro Gerät (HF):", min_val=1, max_val=64
        )
        self._add_spin_box_setting(
            "hf_grad_accum", "Gradient Accumulation Steps (HF):", min_val=1, max_val=128
        )

        # LoRA Parameter
        self.form_layout.addRow(QLabel("<b><u>LoRA Parameter:</u></b>"))
        self._add_spin_box_setting(
            "lora_r", "LoRA Rank (r):", min_val=4, max_val=256, step=4
        )
        self._add_spin_box_setting(
            "lora_alpha", "LoRA Alpha:", min_val=8, max_val=512, step=8
        )
        self._add_double_spin_box_setting(
            "lora_dropout",
            "LoRA Dropout:",
            decimals=3,
            step=0.005,
            min_val=0.0,
            max_val=0.5,
        )

        # Linter Einstellungen
        self.form_layout.addRow(QLabel("<b><u>Linter (Flake8):</u></b>"))
        self._add_line_edit_setting(
            "flake8_path",
            "Pfad zu Flake8:",
            tooltip="Pfad zur Flake8-Executable (z.B. 'flake8' oder '/usr/bin/flake8').",
        )

        self._add_line_edit_setting(
            "linter_ignore_codes",
            "Ignorierte Fehler (kommagetrennt):",
            tooltip="Eine kommagetrennte Liste von Flake8-Fehlercodes, die ignoriert werden sollen (z.B. E501,W292).",
        )
        # Weitere Einstellungen könnten hier hinzugefügt werden
        # self.form_layout.addRow(QLabel("<b><u>Weitere Einstellungen:</u></b>"))
        # self._add_line_edit_setting('hf_base_model_name', "HF Basismodell Name:")
        # self._add_line_edit_setting('lora_adapter_path', "Pfad für LoRA Adapter:")

    def _add_line_edit_setting(
        self, key: str, label_text: str, tooltip: Optional[str] = None
    ):
        """Hilfsmethode zum Hinzufügen eines QLineEdit-basierten Einstellungsfeldes."""
        widget = QLineEdit()
        value = self.settings.get(key, "")
        if isinstance(value, list):
            # Wandle eine Liste ['E501', 'W292'] in den String "E501, W292" um.
            # Nicht str(value), was "['E501', 'W292']" ergeben würde.
            widget.setText(", ".join(value))
        else:
            widget.setText(str(value))
        if tooltip:
            widget.setToolTip(tooltip)
        self.form_layout.addRow(label_text, widget)
        self.widgets[key] = widget

    def _add_spin_box_setting(
        self,
        key: str,
        label_text: str,
        min_val: int,
        max_val: int,
        step: int = 1,
        tooltip: Optional[str] = None,
    ):
        """Hilfsmethode zum Hinzufügen eines QSpinBox-basierten Einstellungsfeldes."""
        widget = QSpinBox()
        widget.setRange(min_val, max_val)
        widget.setSingleStep(step)
        widget.setValue(int(self.settings.get(key, min_val)))

        if tooltip:
            widget.setToolTip(tooltip)
        self.form_layout.addRow(label_text, widget)
        self.widgets[key] = widget

    def _add_double_spin_box_setting(
        self,
        key: str,
        label_text: str,
        decimals: int,
        step: float,
        min_val: float,
        max_val: float,
        tooltip: Optional[str] = None,
    ):
        """Hilfsmethode zum Hinzufügen eines QDoubleSpinBox-basierten Einstellungsfeldes."""
        widget = QDoubleSpinBox()
        widget.setDecimals(decimals)
        widget.setSingleStep(step)
        widget.setRange(min_val, max_val)
        widget.setValue(float(self.settings.get(key, min_val)))
        if tooltip:
            widget.setToolTip(tooltip)
        self.form_layout.addRow(label_text, widget)
        self.widgets[key] = widget

    # Weitere Helper-Methoden für QCheckBox, QComboBox etc. könnten hier folgen

    def accept(self):
        """Wird aufgerufen, wenn OK geklickt wird. Speichert die Einstellungen."""
        logger.info("Speichere Einstellungen aus dem Dialog...")
        for key, widget in self.widgets.items():
            try:
                if key == "linter_ignore_codes" and isinstance(widget, QLineEdit):
                    codes_str = widget.text()
                    codes_list = [
                        code.strip().upper()
                        for code in codes_str.split(",")
                        if code.strip()
                    ]
                    self.settings[key] = codes_list
                elif isinstance(widget, QSpinBox):
                    self.settings[key] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    self.settings[key] = widget.value()
                elif isinstance(widget, QLineEdit):
                    self.settings[key] = widget.text()
                elif isinstance(
                    widget, QCheckBox
                ):  # Beispiel, falls CheckBoxen verwendet werden
                    self.settings[key] = widget.isChecked()
                elif isinstance(
                    widget, QComboBox
                ):  # Beispiel, falls ComboBoxen verwendet werden
                    self.settings[key] = widget.currentText()
                # Füge hier weitere Widget-Typen hinzu, falls benötigt
                logger.debug(f"  Einstellung '{key}' gesetzt auf: {self.settings[key]}")
            except Exception as e:
                logger.error(
                    f"Fehler beim Auslesen des Widgets für Schlüssel '{key}': {e}",
                    exc_info=True,
                )

        super().accept()  # Schließt den Dialog mit QDialog.Accepted

    def get_settings(self) -> Dict[str, Any]:
        """
        Gibt das (potenziell modifizierte) Einstellungs-Dictionary zurück.
        Wird normalerweise nach dialog.exec() == QDialog.Accepted aufgerufen.
        """
        return self.settings
