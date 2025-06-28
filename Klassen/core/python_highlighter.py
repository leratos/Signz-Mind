# Klassen/python_highlighter.py
# -*- coding: utf-8 -*-

"""
Definiert die PythonHighlighter-Klasse für Syntaxhervorhebung in QPlainTextEdit,
basierend auf einer an VS Code Dark+ angelehnten Farbgebung.
Optimierte Version für mehrzeilige Strings.
"""

import logging
from typing import List, Tuple
from PySide6.QtCore import QRegularExpression
from PySide6.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor, QFont

logger = logging.getLogger(__name__)

# Farbdefinitionen für bessere Lesbarkeit und Wartbarkeit
COLOR_KEYWORD = QColor("#569CD6")  # Blau für Keywords
COLOR_DECORATOR = QColor("#DCDCAA")  # Gelblich für Dekoratoren und Funktionsnamen
COLOR_STRING = QColor("#CE9178")  # Orange für Strings
COLOR_NUMBER = QColor("#B5CEA8")  # Grünlich für Zahlen
COLOR_COMMENT = QColor("#6A9955")  # Grün für Kommentare
COLOR_FUNCTION_DEF = QColor("#DCDCAA")  # Gelblich für Funktions-/Klassendefinitionen
COLOR_FUNCTION_CALL = QColor("#DCDCAA")  # Gelblich für Funktionsaufrufe


class PythonHighlighter(QSyntaxHighlighter):
    """
    Ein Syntax-Highlighter für Python-Code, der verschiedene Sprachelemente
    farblich hervorhebt.
    """

    # Zustände für mehrzeilige Strings
    NoState = 0
    InTripleSingleQuoteString = 1
    InTripleDoubleQuoteString = 2

    def __init__(self, parent=None):
        """
        Initialisiert den Highlighter und definiert die Hervorhebungsregeln.
        """
        super().__init__(parent)
        logger.debug("PythonHighlighter initialisiert.")

        self.highlighting_rules: List[Tuple[QRegularExpression, QTextCharFormat]] = []

        # 1. Keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(COLOR_KEYWORD)
        keyword_format.setFontWeight(QFont.Weight.Bold)
        keywords = [
            r"\bFalse\b",
            r"\bNone\b",
            r"\bTrue\b",
            r"\band\b",
            r"\bas\b",
            r"\bassert\b",
            r"\basync\b",
            r"\bawait\b",
            r"\bbreak\b",
            r"\bclass\b",
            r"\bcontinue\b",
            r"\bdef\b",
            r"\bdel\b",
            r"\belif\b",
            r"\belse\b",
            r"\bexcept\b",
            r"\bfinally\b",
            r"\bfor\b",
            r"\bfrom\b",
            r"\bglobal\b",
            r"\bif\b",
            r"\bimport\b",
            r"\bin\b",
            r"\bis\b",
            r"\blambda\b",
            r"\bnonlocal\b",
            r"\bnot\b",
            r"\bor\b",
            r"\bpass\b",
            r"\braise\b",
            r"\breturn\b",
            r"\btry\b",
            r"\bwhile\b",
            r"\bwith\b",
            r"\byield\b",
            r"\bself\b",
            r"\bcls\b",
        ]
        self.keyword_patterns = [QRegularExpression(word) for word in keywords]
        for pattern in self.keyword_patterns:
            self.highlighting_rules.append((pattern, keyword_format))

        # 2. Dekoratoren (z.B. @my_decorator)
        decorator_format = QTextCharFormat()
        decorator_format.setForeground(COLOR_DECORATOR)
        decorator_pattern = QRegularExpression(r"^\s*@\w+")
        self.highlighting_rules.append((decorator_pattern, decorator_format))

        # 3. Strings (einfache und doppelte Anführungszeichen - einzeilig)
        self.single_line_string_format = QTextCharFormat()
        self.single_line_string_format.setForeground(COLOR_STRING)
        self.highlighting_rules.append(
            (QRegularExpression(r"'([^'\\]|\\.)*'"), self.single_line_string_format)
        )
        self.highlighting_rules.append(
            (QRegularExpression(r'"([^"\\]|\\.)*"'), self.single_line_string_format)
        )

        # 4. Zahlen (Integer, Float, Hex)
        number_format = QTextCharFormat()
        number_format.setForeground(COLOR_NUMBER)
        number_pattern = QRegularExpression(
            r"\b([0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?|0[xX][0-9a-fA-F]+)\b"
        )
        self.highlighting_rules.append((number_pattern, number_format))

        # 5. Funktions- und Klassendefinitionen (Namen nach 'def'/'class')
        self.definition_format = QTextCharFormat()
        self.definition_format.setForeground(COLOR_FUNCTION_DEF)
        self.definition_format.setFontWeight(QFont.Weight.Bold)
        self.func_class_def_pattern = QRegularExpression(
            r"\b(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)"
        )

        # 6. Funktionsaufrufe (Heuristik: Wort gefolgt von einer öffnenden Klammer)
        self.call_format = QTextCharFormat()
        self.call_format.setForeground(COLOR_FUNCTION_CALL)
        self.call_pattern = QRegularExpression(r"\b[A-Za-z_][A-Za-z0-9_]*(?=\()")

        # 7. Kommentare (#...)
        self.comment_format = QTextCharFormat()
        self.comment_format.setForeground(COLOR_COMMENT)
        self.comment_format.setFontItalic(True)
        self.highlighting_rules.append(
            (QRegularExpression(r"#[^\n]*"), self.comment_format)
        )

        # 8. Mehrzeilige Strings ('''...''' und """...""")
        self.tri_single_quote_format = QTextCharFormat()
        self.tri_single_quote_format.setForeground(COLOR_STRING)
        self.tri_double_quote_format = QTextCharFormat()
        self.tri_double_quote_format.setForeground(COLOR_STRING)

        # Reguläre Ausdrücke für Start und Ende von mehrzeiligen Strings
        # QRegularExpressionOption.MultilineOption ist hier nicht nötig, da wir zeilenweise arbeiten
        self.tri_single_start_expression = QRegularExpression(r"'''")
        self.tri_single_end_expression = QRegularExpression(r"'''")
        self.tri_double_start_expression = QRegularExpression(r'"""')
        self.tri_double_end_expression = QRegularExpression(r'"""')

    def highlightBlock(self, text: str) -> None:
        """
        Hebt einen einzelnen Textblock (typischerweise eine Zeile) hervor.
        """
        # Zuerst die Regeln anwenden, die keine Zustandsverwaltung benötigen
        # und nicht von mehrzeiligen Strings überdeckt werden sollten.
        for pattern, text_format in self.highlighting_rules:
            match_iterator = pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                # Nur formatieren, wenn nicht bereits durch einen mehrzeiligen String formatiert
                if (
                    self.format(match.capturedStart()) != self.tri_single_quote_format
                    and self.format(match.capturedStart())
                    != self.tri_double_quote_format
                ):
                    self.setFormat(
                        match.capturedStart(), match.capturedLength(), text_format
                    )

        # Spezielle Behandlung für Funktions-/Klassendefinitionen
        match_iterator_def = self.func_class_def_pattern.globalMatch(text)
        while match_iterator_def.hasNext():
            match = match_iterator_def.next()
            if (
                self.format(match.capturedStart(2)) != self.tri_single_quote_format
                and self.format(match.capturedStart(2)) != self.tri_double_quote_format
            ):
                self.setFormat(
                    match.capturedStart(2),
                    match.capturedLength(2),
                    self.definition_format,
                )

        # Spezielle Behandlung für Funktionsaufrufe
        match_iterator_call = self.call_pattern.globalMatch(text)
        while match_iterator_call.hasNext():
            match = match_iterator_call.next()
            is_keyword = any(
                kw_pattern.match(text, match.capturedStart()).hasMatch()
                and kw_pattern.pattern() == rf"\b{match.captured(0)}\b"
                for kw_pattern in self.keyword_patterns
            )
            if (
                not is_keyword
                and self.format(match.capturedStart()) != self.tri_single_quote_format
                and self.format(match.capturedStart()) != self.tri_double_quote_format
            ):
                self.setFormat(
                    match.capturedStart(), match.capturedLength(), self.call_format
                )

        # --- Zustandsbehaftetes Highlighting für mehrzeilige Strings ---
        current_block_state = self.NoState
        previous_state = self.previousBlockState()

        start_offset = 0
        if previous_state == self.InTripleSingleQuoteString:
            match_end = self.tri_single_end_expression.match(text, start_offset)
            if match_end.hasMatch():
                end_offset = match_end.capturedStart() + match_end.capturedLength()
                self.setFormat(0, end_offset, self.tri_single_quote_format)
                start_offset = end_offset
                current_block_state = self.NoState
            else:
                self.setFormat(0, len(text), self.tri_single_quote_format)
                current_block_state = self.InTripleSingleQuoteString
        elif previous_state == self.InTripleDoubleQuoteString:
            match_end = self.tri_double_end_expression.match(text, start_offset)
            if match_end.hasMatch():
                end_offset = match_end.capturedStart() + match_end.capturedLength()
                self.setFormat(0, end_offset, self.tri_double_quote_format)
                start_offset = end_offset
                current_block_state = self.NoState
            else:
                self.setFormat(0, len(text), self.tri_double_quote_format)
                current_block_state = self.InTripleDoubleQuoteString

        # Suche nach neuen Anfängen von mehrzeiligen Strings im Rest des Blocks
        while start_offset < len(text):
            match_single_start = self.tri_single_start_expression.match(
                text, start_offset
            )
            match_double_start = self.tri_double_start_expression.match(
                text, start_offset
            )

            start_single = (
                match_single_start.capturedStart()
                if match_single_start.hasMatch()
                else -1
            )
            start_double = (
                match_double_start.capturedStart()
                if match_double_start.hasMatch()
                else -1
            )

            if start_single != -1 and (
                start_double == -1 or start_single < start_double
            ):
                # ''' beginnt
                start_expr_match = match_single_start
                end_expr = self.tri_single_end_expression
                multi_format = self.tri_single_quote_format
                next_state_if_unterminated = self.InTripleSingleQuoteString
            elif start_double != -1:
                # """ beginnt
                start_expr_match = match_double_start
                end_expr = self.tri_double_end_expression
                multi_format = self.tri_double_quote_format
                next_state_if_unterminated = self.InTripleDoubleQuoteString
            else:
                break  # Kein weiterer mehrzeiliger String-Anfang in diesem Block

            string_start_offset = start_expr_match.capturedStart()
            # Suche das Ende ab dem Start des aktuellen mehrzeiligen Strings
            match_end = end_expr.match(
                text, string_start_offset + start_expr_match.capturedLength()
            )

            if match_end.hasMatch():
                string_end_offset = (
                    match_end.capturedStart() + match_end.capturedLength()
                )
                self.setFormat(
                    string_start_offset,
                    string_end_offset - string_start_offset,
                    multi_format,
                )
                start_offset = string_end_offset
                current_block_state = self.NoState  # String endet in dieser Zeile
            else:
                self.setFormat(
                    string_start_offset, len(text) - string_start_offset, multi_format
                )
                current_block_state = next_state_if_unterminated  # String geht weiter
                break  # Rest des Blocks ist Teil dieses mehrzeiligen Strings

        self.setCurrentBlockState(current_block_state)
