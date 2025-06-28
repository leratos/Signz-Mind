# Klassen/code_editor_widget.py
# -*- coding: utf-8 -*-

import re
from PySide6.QtCore import Qt, QRect, QSize
from PySide6.QtWidgets import QWidget, QPlainTextEdit, QTextEdit
from PySide6.QtGui import (
    QPainter,
    QColor,
    QFont,
    QTextFormat,
    QPaintEvent,
    QResizeEvent,
    QKeyEvent,
    QTextCursor,
    QTextCharFormat,
    QTextDocument,
)  # QTextDocument für FindFlags
from typing import List, Optional, Dict


# --- Line Number Area Widget ---
class LineNumberArea(QWidget):
    def __init__(self, editor: "CodeEditor"):
        super().__init__(editor)
        self.codeEditor = editor
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(palette.ColorRole.Window, QColor("#252526"))
        self.setPalette(palette)

    def sizeHint(self) -> QSize:
        return QSize(self.codeEditor.lineNumberAreaWidth(), 0)

    def paintEvent(self, event: QPaintEvent) -> None:
        self.codeEditor.lineNumberAreaPaintEvent(event)


# --- Code Editor Widget ---
class CodeEditor(QPlainTextEdit):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.lineNumberArea = LineNumberArea(self)
        self.lint_error_selections: List[QTextEdit.ExtraSelection] = []

        self.blockCountChanged.connect(self.updateLineNumberAreaWidth)
        self.updateRequest.connect(self.updateLineNumberArea)
        self.cursorPositionChanged.connect(self.highlightCurrentLine)

        self.updateLineNumberAreaWidth()
        self.highlightCurrentLine()

        font = QFont("monospace")
        font.setPointSize(10)
        self.setFont(font)
        self.tab_width_spaces = 4
        self.indent_line_color = QColor("#404040")

    def lineNumberAreaWidth(self) -> int:
        digits = 1
        count = max(1, self.blockCount())
        while count >= 10:
            count //= 10
            digits += 1
        space = 10 + self.fontMetrics().horizontalAdvance("9") * digits
        return space

    def updateLineNumberAreaWidth(self, newBlockCount: int = 0) -> None:
        self.setViewportMargins(self.lineNumberAreaWidth(), 0, 0, 0)

    def updateLineNumberArea(self, rect: QRect, dy: int) -> None:
        if dy:
            self.lineNumberArea.scroll(0, dy)
        else:
            self.lineNumberArea.update(
                0, rect.y(), self.lineNumberArea.width(), rect.height()
            )
        if rect.contains(self.viewport().rect()):
            self.updateLineNumberAreaWidth()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.lineNumberArea.setGeometry(
            QRect(cr.left(), cr.top(), self.lineNumberAreaWidth(), cr.height())
        )

    def lineNumberAreaPaintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self.lineNumberArea)
        block = self.firstVisibleBlock()
        blockNumber = block.blockNumber()
        top = int(
            self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        )
        bottom = top + int(self.blockBoundingRect(block).height())
        number_color = QColor("#858585")
        current_line_number_color = QColor("#C6C6C6")

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(blockNumber + 1)
                is_current_line = self.textCursor().blockNumber() == blockNumber
                painter.setPen(
                    current_line_number_color if is_current_line else number_color
                )
                painter.setFont(self.font())
                painter.drawText(
                    0,
                    top,
                    self.lineNumberArea.width() - 5,
                    self.fontMetrics().height(),
                    Qt.AlignmentFlag.AlignRight,
                    number,
                )
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            blockNumber += 1

    def highlightCurrentLine(self) -> None:
        extraSelections: List[QTextEdit.ExtraSelection] = []
        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()
            lineColor = QColor("#333337")
            selection.format.setBackground(lineColor)
            selection.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            extraSelections.append(selection)
        extraSelections.extend(self.lint_error_selections)
        self.setExtraSelections(extraSelections)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        cursor = self.textCursor()
        current_block = cursor.block()
        text = current_block.text()
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            leading_whitespace = ""
            match = re.match(r"^(\s*).*", text)
            if match:
                leading_whitespace = match.group(1)
            cursor.insertBlock()
            cursor.insertText(leading_whitespace)
            if text.strip().endswith(":"):
                cursor.insertText(" " * self.tab_width_spaces)
            self.ensureCursorVisible()
            event.accept()
        elif event.key() == Qt.Key.Key_Tab:
            cursor.insertText(" " * self.tab_width_spaces)
            event.accept()
        elif event.key() == Qt.Key.Key_Backspace:
            if (
                cursor.positionInBlock() > 0
                and cursor.positionInBlock() % self.tab_width_spaces == 0
                and text.startswith(" " * cursor.positionInBlock())
                and text[
                    cursor.positionInBlock()
                    - self.tab_width_spaces: cursor.positionInBlock()
                ]
                == " " * self.tab_width_spaces
            ):
                for _ in range(self.tab_width_spaces):
                    cursor.deletePreviousChar()
                event.accept()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        metrics = self.fontMetrics()
        space_width = metrics.horizontalAdvance(" ")
        tab_pixel_width = max(1, space_width * self.tab_width_spaces)
        visible_rect = self.viewport().rect()
        painter.setPen(self.indent_line_color)
        block = self.firstVisibleBlock()
        while (
            block.isValid()
            and self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
            <= visible_rect.bottom()
        ):
            if (
                block.isVisible()
                and self.blockBoundingGeometry(block)
                .translated(self.contentOffset())
                .bottom()
                >= visible_rect.top()
            ):
                text = block.text()
                leading_whitespace_width = 0
                for char_idx, char_val in enumerate(text):
                    if char_val == " ":
                        leading_whitespace_width += space_width
                    elif char_val == "\t":
                        current_x_pos_in_line = leading_whitespace_width
                        distance_to_next_tab = tab_pixel_width - (
                            current_x_pos_in_line % tab_pixel_width
                        )
                        if distance_to_next_tab == 0:
                            distance_to_next_tab = tab_pixel_width
                        leading_whitespace_width += distance_to_next_tab
                    else:
                        break
                block_rect = self.blockBoundingGeometry(block).translated(
                    self.contentOffset()
                )
                level = 1
                while True:
                    current_line_x = level * tab_pixel_width
                    if current_line_x >= visible_rect.right() + tab_pixel_width:
                        break
                    required_pixel_width_for_level = level * tab_pixel_width
                    if leading_whitespace_width >= required_pixel_width_for_level:
                        if (
                            block_rect.bottom() >= visible_rect.top()
                            and block_rect.top() <= visible_rect.bottom()
                        ):
                            painter.drawLine(
                                int(current_line_x),
                                int(block_rect.top()),
                                int(current_line_x),
                                int(block_rect.bottom()),
                            )
                    else:
                        break
                    level += 1
                    if level > 30:
                        break
            block = block.next()

    def clear_lint_errors(self) -> None:
        self.lint_error_selections = []
        self.highlightCurrentLine()

    def set_lint_errors(self, errors: List[Dict]) -> None:
        self.clear_lint_errors()
        error_highlight_format = QTextCharFormat()
        error_highlight_format.setUnderlineColor(QColor("red"))
        error_highlight_format.setUnderlineStyle(
            QTextCharFormat.UnderlineStyle.WaveUnderline
        )

        doc = self.document()
        for error_info in errors:
            line_number = error_info.get("line")
            column_number = error_info.get("column")

            if (
                line_number is not None
                and line_number > 0
                and line_number <= doc.blockCount()
            ):
                selection = QTextEdit.ExtraSelection()
                selection.format = error_highlight_format

                block = doc.findBlockByNumber(line_number - 1)

                if block.isValid():
                    selection.cursor = QTextCursor(block)
                    line_text = block.text()
                    line_len = len(line_text)

                    highlight_entire_block = True

                    if column_number is not None and column_number > 0:
                        start_char_index = column_number - 1
                        if start_char_index < line_len:
                            selection.cursor.setPosition(
                                block.position() + start_char_index
                            )
                            temp_cursor = QTextCursor(selection.cursor)
                            temp_cursor.select(
                                QTextCursor.SelectionType.WordUnderCursor
                            )
                            if (
                                temp_cursor.hasSelection()
                                and temp_cursor.selectionStart()
                                <= selection.cursor.position()
                                < temp_cursor.selectionEnd()
                            ):
                                selection.cursor = temp_cursor
                                highlight_entire_block = False
                            else:
                                selection.cursor.movePosition(
                                    QTextCursor.MoveOperation.NextCharacter,
                                    QTextCursor.MoveMode.KeepAnchor,
                                    1,
                                )
                                if selection.cursor.hasSelection():
                                    highlight_entire_block = False
                        elif start_char_index >= line_len and line_len > 0:
                            selection.cursor.setPosition(block.position() + line_len)
                            selection.cursor.select(
                                QTextCursor.SelectionType.WordUnderCursor
                            )
                            if (
                                not selection.cursor.hasSelection()
                                or selection.cursor.anchor() < block.position()
                            ):
                                selection.cursor.setPosition(
                                    block.position() + line_len - 1
                                )
                                selection.cursor.movePosition(
                                    QTextCursor.MoveOperation.NextCharacter,
                                    QTextCursor.MoveMode.KeepAnchor,
                                    1,
                                )
                            if (
                                selection.cursor.hasSelection()
                                and selection.cursor.anchor() >= block.position()
                            ):
                                highlight_entire_block = False
                    if highlight_entire_block:
                        selection.cursor.select(
                            QTextCursor.SelectionType.BlockUnderCursor
                        )
                    self.lint_error_selections.append(selection)
        self.highlightCurrentLine()

    # --- Suchen und Ersetzen Methoden ---
    def find_text(
        self,
        text_to_find: str,
        find_flags: QTextDocument.FindFlags,
        cursor: Optional[QTextCursor] = None,
    ) -> bool:
        if not text_to_find:
            return False

        start_cursor = cursor if cursor is not None else self.textCursor()

        if find_flags & QTextDocument.FindFlag.FindBackward:
            if (
                start_cursor.hasSelection()
                and start_cursor.selectedText() == text_to_find
            ):  # Nur wenn die aktuelle Auswahl der Suchtext ist
                search_position = start_cursor.selectionStart()
                temp_cursor = QTextCursor(self.document())
                temp_cursor.setPosition(search_position)
                start_cursor = temp_cursor
            # Ansonsten von der aktuellen Cursorposition (oder Anfang der Auswahl bei Rückwärtssuche)
        else:
            if (
                start_cursor.hasSelection()
                and start_cursor.selectedText() == text_to_find
            ):  # Nur wenn die aktuelle Auswahl der Suchtext ist
                search_position = start_cursor.selectionEnd()
                temp_cursor = QTextCursor(self.document())
                temp_cursor.setPosition(search_position)
                start_cursor = temp_cursor

        found_cursor = self.document().find(text_to_find, start_cursor, find_flags)

        if not found_cursor.isNull():
            self.setTextCursor(found_cursor)
            return True
        else:
            # Wraparound-Suche
            if find_flags & QTextDocument.FindFlag.FindBackward:
                end_cursor = QTextCursor(self.document())
                end_cursor.movePosition(QTextCursor.MoveOperation.End)
                found_cursor = self.document().find(
                    text_to_find, end_cursor, find_flags
                )
            else:
                start_doc_cursor = QTextCursor(self.document())
                start_doc_cursor.movePosition(QTextCursor.MoveOperation.Start)
                found_cursor = self.document().find(
                    text_to_find, start_doc_cursor, find_flags
                )

            if not found_cursor.isNull():
                self.setTextCursor(found_cursor)
                return True
            return False

    def replace_current_selection(self, replace_text: str) -> bool:
        cursor = self.textCursor()
        if cursor.hasSelection():
            cursor.insertText(replace_text)
            return True
        return False

    def replace_all_occurrences(
        self, text_to_find: str, replace_text: str, find_flags: QTextDocument.FindFlags
    ) -> int:
        if not text_to_find:
            return 0

        count = 0
        current_find_flags = find_flags & ~QTextDocument.FindFlag.FindBackward

        current_cursor = QTextCursor(self.document())
        current_cursor.movePosition(QTextCursor.MoveOperation.Start)

        self.textCursor().beginEditBlock()
        try:
            while True:
                # Wichtig: Immer vom aktuellen Cursor suchen, nicht self.textCursor() direkt,
                # da self.textCursor() durch setTextCursor verändert wird.
                found_cursor = self.document().find(
                    text_to_find, current_cursor, current_find_flags
                )
                if found_cursor.isNull():
                    break

                # Den Editor-Cursor auf die Fundstelle setzen, um zu ersetzen
                self.setTextCursor(found_cursor)
                self.textCursor().insertText(replace_text)
                count += 1

                # Den Such-Startcursor für die nächste Iteration setzen.
                # Er muss hinter dem gerade ersetzten Text beginnen.
                current_cursor = self.textCursor()  # Holt den Cursor nach der Einfügung
        finally:
            self.textCursor().endEditBlock()

        return count

    def goto_line(self, line_number: int, select_line: bool = True, column: int = 0):
        """
        Bewegt den Cursor zur angegebenen Zeilennummer (1-basiert) und Spalte (0-basiert)
        und wählt optional die Zeile aus. Stellt sicher, dass der Cursor sichtbar ist.
        """
        if line_number <= 0:
            return

        block = self.document().findBlockByNumber(line_number - 1)  # 0-basiert
        if not block.isValid():
            return

        cursor = QTextCursor(block)
        # Bewege den Cursor zur gewünschten Spalte innerhalb der Zeile
        # Beachte, dass die Spalte die Position innerhalb des Blocks ist.
        cursor.movePosition(
            QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.MoveAnchor, column
        )

        if select_line:
            # Um die Zeile zu markieren, nachdem der Cursor an der Spalte ist:
            # Gehe zum Anfang der Zeile, dann mit KeepAnchor zum Ende.
            temp_cursor_for_selection = QTextCursor(
                block
            )  # Neuer Cursor am Blockanfang
            temp_cursor_for_selection.movePosition(
                QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor
            )
            self.setTextCursor(temp_cursor_for_selection)
        else:
            self.setTextCursor(cursor)  # Nur Cursor setzen

        self.ensureCursorVisible()
