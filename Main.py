# main.py
import sys
from PySide6.QtWidgets import QApplication

import logging
logger = logging.getLogger(__name__)
if True:
    # Setze den Logger auf DEBUG-Level
    logging.getLogger().setLevel(logging.DEBUG)
    # Konfiguriere den Logger
    logging.basicConfig(
        level=logging.DEBUG,  # Zeige alle Nachrichten ab DEBUG-Level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Importiere die Hauptfenster-Klasse aus dem neuen Modul im Klassen-Ordner
# Wir nennen die Klasse dort vielleicht "MainWindow" statt "CodeAnalyzerApp"
try:
    from Klassen.ui.ui_main_window import MainWindow
except ImportError as e:
    logger.info(f"Fehler beim Importieren von MainWindow: {e}")
    logger.info("Stelle sicher, dass der Ordner 'Klassen' existiert," /
                "eine '__init__.py' enthält")
    logger.info("und 'ui_main_window.py' die Klasse 'MainWindow' definiert.")
    sys.exit(1)  # Beenden, wenn Import fehlschlägt

# Hauptausführungsblock
if __name__ == "__main__":
    # Erstelle die QApplication-Instanz
    app = QApplication(sys.argv)

    # Erstelle eine Instanz des Hauptfensters
    try:
        main_window = MainWindow()
        main_window.show()  # Zeige das Fenster an
    except Exception as e:
        print(f"Fehler beim Initialisieren des Hauptfensters: {e}")
        # Optional: Detaillierteren Traceback anzeigen
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Starte die Qt Event Loop
    sys.exit(app.exec())