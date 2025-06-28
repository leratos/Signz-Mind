# Klassen/core/formatting_service.py
# -*- coding: utf-8 -*-

"""
Provides a service for formatting Python code using the 'black' library.
"""

import logging
from typing import Tuple

# Attempt to import black, handle if not available
try:
    import black

    BLACK_AVAILABLE = True
except ImportError:
    BLACK_AVAILABLE = False
    # Define placeholder types if black is not installed
    FileMode = type("FileMode", (object,), {})
    InvalidInput = type("InvalidInput", (Exception,), {})

logger = logging.getLogger(__name__)


class FormattingService:
    """
    A service to format Python code snippets or entire files using black.
    """

    def __init__(self):
        """Initializes the FormattingService."""
        if not BLACK_AVAILABLE:
            logger.warning(
                "The 'black' library is not installed. Formatting functionality will be disabled. "
                "Please install it using: pip install black"
            )
        else:
            logger.info("FormattingService initialized successfully with 'black'.")

    def format_code(self, code_string: str) -> Tuple[bool, str]:
        """
        Formats a given string of Python code using black.

        Args:
            code_string (str): The Python code to format.

        Returns:
            A tuple containing:
            - bool: True if formatting was successful or not needed, False on error.
            - str: The formatted code, or an error message if formatting failed.
        """
        if not BLACK_AVAILABLE:
            return False, "Fehler: Die 'black' Bibliothek ist nicht installiert."

        if not code_string.strip():
            return True, ""  # Nothing to format

        try:
            # Create a black file mode object (can be configured in the future)
            mode = black.FileMode()

            # Format the string
            formatted_code = black.format_str(code_string, mode=mode)

            # black.format_str might add a trailing newline, which is usually desired.
            return True, formatted_code

        except black.InvalidInput as e:
            error_message = f"Formatierungsfehler: Der Code enthält einen Syntaxfehler, den 'black' nicht auflösen kann. Bitte beheben Sie den Fehler zuerst. Details: {e}"
            logger.warning(error_message)
            return False, error_message
        except Exception as e:
            error_message = f"Ein unerwarteter Fehler ist bei der Code-Formatierung aufgetreten: {e}"
            logger.error(error_message, exc_info=True)
            return False, error_message
