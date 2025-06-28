# Klassen/utils.py
# -*- coding: utf-8 -*-

"""
Allgemeine Hilfsfunktionen für die Anwendung.
"""

import logging

logger = logging.getLogger(__name__)

logger.info("utils.py geladen.")


def decode_token(idx: int) -> str:
    """
    Dekodiert eine Token-ID in ein druckbares Zeichen oder einen Platzhalter.
    Dies ist eine sehr einfache Implementierung, primär für ASCII.

    Args:
        idx (int): Die zu dekodierende Token-ID.

    Returns:
        str: Das dekodierte Zeichen oder ein Platzhalter (z.B. '<128>').
    """
    # Einfacher ASCII-Dekodierer, Platzhalter für andere
    if 32 <= idx < 127:  # Druckbare ASCII-Zeichen
        return chr(idx)

    # Platzhalter für nicht druckbare ASCII-Zeichen oder Nicht-ASCII-Bytes,
    # falls vocab_size > 127 wäre.
    # Eine komplexere Zuordnung wäre nötig, wenn das Vokabular mehr enthält.
    logger.debug(f"Dekodiere Token-ID {idx} zu Platzhalter '<{idx}>'.")
    return f"<{idx}>"


# Hier könnten später weitere allgemeine Hilfsfunktionen hinzukommen,
# z.B. für das Laden/Speichern von JSON-Konfigurationen, Datumsformatierung etc.
