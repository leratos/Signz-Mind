# Klassen/ui_threads.py
# -*- coding: utf-8 -*-

"""
Definiert QThread-basierte Klassen für Hintergrundaufgaben der UI,
wie z.B. das Fine-Tuning von ML-Modellen.

Einbindung:
-------------
from Klassen.ui_threads import FineTuneThread
from Klassen.hf_fine_tuner import HFFineTuner # Importiere den neuen Tuner

# Innerhalb Ihrer UI-Klasse (z.B. MainWindow):
# # Annahme: self.hf_fine_tuner ist eine Instanz von HFFineTuner
# self.fine_tune_thread = FineTuneThread(self.hf_fine_tuner) # Übergib den Tuner
# self.fine_tune_thread.progressUpdated.connect(self.handle_progress)
# self.fine_tune_thread.finished.connect(self.on_finetune_finished)
# self.fine_tune_thread.start()
"""

from PySide6.QtCore import QThread, Signal
import traceback

# Importiere HFFineTuner nur für Typ-Hinweise, vermeide zirkuläre Abhängigkeiten
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # This avoids circular imports at runtime but allows type checkers
    from .services.hf_fine_tuner import HFFineTuner


class FineTuneThread(QThread):
    """Runs the HF model fine-tuning process using HFFineTuner in a separate thread."""

    # Signal für Fortschritt (0-100, oder -1 für Fehler, 0 für unbestimmt/Start)
    progressUpdated = Signal(int)
    # Signal für Abschluss (Erfolg, Ergebnis-Dict/Fehlermeldung)
    # Verwende object als Typ für das Ergebnis, da es dict oder str sein kann
    finished = Signal(bool, object)

    def __init__(self, hf_fine_tuner_instance: "HFFineTuner", parent=None):
        """
        Initializes the thread.
        Args:
            hf_fine_tuner_instance: An instance of the HFFineTuner class.
            parent: The parent QObject.
        """
        super().__init__(parent)
        # Store the HFFineTuner instance passed from MainWindow
        self.hf_fine_tuner = hf_fine_tuner_instance
        # Internal flag to help manage graceful stopping if needed
        self._is_running = False
        print("DEBUG: FineTuneThread initialized with HFFineTuner instance.")

    def run(self) -> None:
        """Executes the fine-tuning task using HFFineTuner."""
        self._is_running = True
        print("DEBUG: FineTuneThread run() started.")
        # Ensure HFFineTuner instance is valid before proceeding
        if not self.hf_fine_tuner:
            # Emit finished signal only if the thread wasn't stopped externally
            if self._is_running:
                print("ERROR: FineTuneThread: HFFineTuner instance is None.")
                self.finished.emit(False, "HFFineTuner not available in thread.")
            self._is_running = False  # Mark as not running
            return  # Exit the run method

        try:
            # Emit progress signal to indicate start
            if self._is_running:
                self.progressUpdated.emit(0)

            # Call the fine-tuning method of the HFFineTuner instance
            # Pass a lambda function as the progress callback
            # Check the _is_running flag inside the callback
            print("DEBUG: FineTuneThread: Calling hf_fine_tuner.run_fine_tuning...")
            success, result = self.hf_fine_tuner.run_fine_tuning(
                progress_callback=lambda p: self.progressUpdated.emit(
                    p if self._is_running else -1
                )
            )

            # Emit the finished signal only if the thread is still supposed to be running
            if self._is_running:
                self.finished.emit(success, result)
            print(f"DEBUG: FineTuneThread run() finished. Success: {success}")

        except Exception as e:
            # Catch any unhandled exceptions during the fine-tuning process
            print(f"ERROR: Unhandled exception in FineTuneThread: {e}")
            traceback.print_exc()
            # Emit error signals only if the thread is still supposed to be running
            if self._is_running:
                self.progressUpdated.emit(-1)  # Signal an error in progress
                self.finished.emit(False, f"Unhandled exception in thread: {e}")
        finally:
            # Ensure the running flag is set to False when the run method exits
            self._is_running = False
            print("DEBUG: FineTuneThread run() method exited.")

    def stop(self) -> None:
        """Signals the thread to stop gracefully."""
        print("DEBUG: FineTuneThread stop() called.")
        self._is_running = False
        # Note: True interruption of the underlying training process is complex
        # and might require modifications within HFFineTuner or the Hugging Face Trainer.
        # This flag primarily prevents emitting signals after stop() is called.

    def isRunning(self) -> bool:
        """Overrides QThread.isRunning() to potentially use our internal flag."""
        return self._is_running and super().isRunning()
