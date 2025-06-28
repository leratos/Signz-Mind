# Klassen/core/code_context_analyzer.py
# -*- coding: utf-8 -*-

"""
Provides services for statically analyzing Python code using the Abstract
Syntax Tree (AST) to extract contextual information like imports and definitions.
"""

import ast
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class DefinitionInfo:
    """A data class to store information about a single code definition."""
    def __init__(
        self,
        name: str,
        def_type: str,
        file_path: Path,
        line_number: int,
        column_number: int,
        end_line_number: Optional[int] = None,
        end_column_number: Optional[int] = None,
        signature: Optional[str] = None,
        docstring: Optional[str] = None,
        parent_name: Optional[str] = None,
    ):
        """
        Initializes the DefinitionInfo object.

        Args:
            name: The name of the identifier (e.g., function name, variable name).
            def_type: The type of definition (e.g., 'function', 'class', 'variable').
            file_path: The path to the file containing the definition.
            line_number: The starting line number of the definition.
            column_number: The starting column number of the definition.
            end_line_number: The ending line number of the definition node.
            end_column_number: The ending column number of the definition node.
            signature: The signature, if applicable (e.g., for functions).
            docstring: The docstring, if available.
            parent_name: The name of the parent scope (e.g., class or function name).
        """
        self.name = name
        self.def_type = def_type
        self.file_path = file_path
        self.line_number = line_number
        self.column_number = column_number
        self.end_line_number = (
            end_line_number if end_line_number is not None else line_number
        )
        self.end_column_number = end_column_number
        self.signature = signature
        self.docstring = docstring
        self.parent_name = parent_name

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the object."""
        return (
            f"DefinitionInfo(name='{self.name}', type='{self.def_type}', "
            f"file='{self.file_path.name}', line={self.line_number}, col={self.column_number})"
        )


class ImportInfo:
    """A data class to store information about a single import statement."""
    def __init__(
        self,
        module_name: Optional[str],
        imported_names: Optional[
            List[Tuple[str, Optional[str]]]
        ] = None,
        line_number: int = 0,
        column_number: int = 0,
        end_line_number: Optional[int] = None,
        end_column_number: Optional[int] = None,
        is_from_import: bool = False,
        level: int = 0,
        file_path: Optional[Path] = None,
        alias_nodes: Optional[List[ast.alias]] = None,
    ):
        """
        Initializes the ImportInfo object.

        Args:
            module_name: The name of the module being imported from (for 'from ... import ...').
            imported_names: A list of tuples, where each is (name, alias).
            line_number: The line number where the import occurs.
            column_number: The column number where the import occurs.
            end_line_number: The ending line number of the import node.
            end_column_number: The ending column number of the import node.
            is_from_import: Flag indicating if it's a 'from' import.
            level: The level for relative imports (e.g., '.' is 1).
            file_path: The path to the file containing the import.
            alias_nodes: The raw ast.alias nodes for more detailed analysis.
        """
        self.module_name = module_name
        self.imported_names: List[Tuple[str, Optional[str]]] = (
            imported_names if imported_names is not None else []
        )
        self.line_number = line_number
        self.column_number = column_number
        self.end_line_number = (
            end_line_number if end_line_number is not None else line_number
        )
        self.end_column_number = end_column_number
        self.is_from_import = is_from_import
        self.level = level
        self.file_path = file_path
        self.alias_nodes = alias_nodes if alias_nodes else []

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the import."""
        if self.is_from_import:
            relative_prefix = "." * self.level
            module_str = self.module_name if self.module_name else ""
            from_part = f"from='{relative_prefix}{module_str}'"
            names_repr_list = [
                f"{name} as {alias}" if alias else name
                for name, alias in self.imported_names
            ]
            names_str = f"import=[{', '.join(repr(n) for n in names_repr_list)}]"
            return f"ImportInfo({from_part}, {names_str}, line={self.line_number})"
        else:
            import_clauses = []
            for name, alias in self.imported_names:
                clause = name
                if alias:
                    clause += f" as {alias}"
                import_clauses.append(clause)
            return f"ImportInfo(import=[{', '.join(repr(c) for c in import_clauses)}], line={self.line_number})"


class CodeContextAnalyzer:
    """
    Analyzes Python code to extract contextual information for AI services.
    This class uses Python's `ast` module to parse code into an Abstract
    Syntax Tree, then walks the tree to find all imports and definitions
    (classes, functions, variables). It uses a cache to avoid re-analyzing
    unchanged files.
    """
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initializes the analyzer.

        Args:
            project_root: The root directory of the current project. Used for
                          resolving project-wide context in the future.
        """
        self.project_root: Optional[Path] = None
        self.analysis_cache: Dict[Path, Dict[str, Any]] = {}

        if project_root:
            self.update_project_root(project_root)

        logger.info(
            f"CodeContextAnalyzer initialisiert. Projektwurzel: {self.project_root}"
        )

    def update_project_root(self, project_root: Optional[Path]):
        """
        Updates the project root directory and clears the analysis cache.

        Args:
            project_root: The new project root path.
        """
        if project_root and project_root.is_dir():
            self.project_root = project_root.resolve()
        else:
            self.project_root = None
        self.analysis_cache.clear()
        logger.info(
            f"CodeContextAnalyzer: Projektwurzel aktualisiert auf {self.project_root}"
        )

    def _parse_code(
        self, code_string: str, file_path_for_logging: str = "<string>"
    ) -> Optional[ast.AST]:
        """
        Safely parses a string of Python code into an AST.

        Args:
            code_string: The code to parse.
            file_path_for_logging: The identifier for logging purposes.

        Returns:
            The parsed `ast.AST` object, or None if a SyntaxError occurs.
        """
        try:
            tree = ast.parse(code_string)
            return tree
        except SyntaxError as e:
            logger.warning(
                f"Syntaxfehler beim Parsen von '{file_path_for_logging}' (Zeile {e.lineno}, Offset {e.offset}): {e.msg}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Allgemeiner Fehler beim Parsen von Code aus '{file_path_for_logging}': {e}",
                exc_info=True,
            )
            return None

    def _get_node_parent_name(self, node: ast.AST, tree: ast.AST) -> Optional[str]:
        return None

    def analyze_imports(
        self,
        code_string: str,
        file_path: Optional[Path] = None,
        tree: Optional[ast.AST] = None,
    ) -> List[ImportInfo]:
        """
        Analyzes the code and returns a list of all found import statements.

        Args:
            code_string: The Python code to analyze.
            file_path: The path of the file being analyzed.
            tree: An optional pre-parsed AST to use.

        Returns:
            A list of `ImportInfo` objects.
        """
        imports_found: List[ImportInfo] = []
        log_path = str(file_path) if file_path else "<string>"
        if tree is None:
            tree = self._parse_code(code_string, log_path)
        if not tree:
            return imports_found
        # Walk the AST to find all Import and ImportFrom nodes
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_module_details: List[Tuple[str, Optional[str]]] = []
                for alias_node in node.names:
                    imported_module_details.append((alias_node.name, alias_node.asname))
                imports_found.append(
                    ImportInfo(
                        module_name=None,
                        imported_names=imported_module_details,
                        line_number=node.lineno,
                        column_number=node.col_offset,
                        end_line_number=getattr(node, "end_lineno", node.lineno),
                        end_column_number=getattr(node, "end_col_offset", None),
                        is_from_import=False,
                        file_path=file_path,
                        alias_nodes=node.names,
                    )
                )
            elif isinstance(node, ast.ImportFrom):
                imported_name_details: List[Tuple[str, Optional[str]]] = []
                for alias_node in node.names:
                    imported_name_details.append((alias_node.name, alias_node.asname))
                imports_found.append(
                    ImportInfo(
                        module_name=node.module,
                        imported_names=imported_name_details,
                        line_number=node.lineno,
                        column_number=node.col_offset,
                        end_line_number=getattr(node, "end_lineno", node.lineno),
                        end_column_number=getattr(node, "end_col_offset", None),
                        is_from_import=True,
                        level=node.level,
                        file_path=file_path,
                        alias_nodes=node.names,
                    )
                )
        return imports_found

    def analyze_definitions(
        self, code_string: str, file_path: Path, tree: Optional[ast.AST] = None
    ) -> List[DefinitionInfo]:
        """
        Analyzes the code and returns a list of all found definitions.

        This includes functions, classes, variables, and parameters.

        Args:
            code_string: The Python code to analyze.
            file_path: The path of the file being analyzed.
            tree: An optional pre-parsed AST to use.

        Returns:
            A list of `DefinitionInfo` objects.
        """
        definitions_found: List[DefinitionInfo] = []
        if tree is None:
            tree = self._parse_code(code_string, str(file_path))
        if not tree:
            return definitions_found
        # Walk the AST to find all definition nodes
        for node in ast.walk(tree):
            parent_name = self._get_node_parent_name(node, tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # This block now uses your more robust signature parsing logic.
                args_list = []
                if node.args:
                    for arg_node in node.args.posonlyargs:
                        args_list.append(
                            arg_node.arg
                            + (
                                f": {ast.unparse(arg_node.annotation)}"
                                if arg_node.annotation
                                else ""
                            )
                        )
                # Add all types of function parameters as definitions
                    for arg_node in node.args.args:
                        args_list.append(
                            arg_node.arg
                            + (
                                f": {ast.unparse(arg_node.annotation)}"
                                if arg_node.annotation
                                else ""
                            )
                        )
                    if node.args.vararg:
                        args_list.append(
                            "*"
                            + node.args.vararg.arg
                            + (
                                f": {ast.unparse(node.args.vararg.annotation)}"
                                if node.args.vararg.annotation
                                else ""
                            )
                        )
                    for arg_node in node.args.kwonlyargs:
                        args_list.append(
                            arg_node.arg
                            + (
                                f": {ast.unparse(arg_node.annotation)}"
                                if arg_node.annotation
                                else ""
                            )
                        )
                    if node.args.kwarg:
                        args_list.append(
                            "**"
                            + node.args.kwarg.arg
                            + (
                                f": {ast.unparse(node.args.kwarg.annotation)}"
                                if node.args.kwarg.annotation
                                else ""
                            )
                        )
                signature = f"{node.name}({', '.join(args_list)})"
                if node.returns:
                    signature += f" -> {ast.unparse(node.returns)}"
                docstring = ast.get_docstring(node, clean=False)
                definitions_found.append(
                    DefinitionInfo(
                        name=node.name,
                        def_type=(
                            "async function"
                            if isinstance(node, ast.AsyncFunctionDef)
                            else "function"
                        ),
                        file_path=file_path,
                        line_number=node.lineno,
                        column_number=node.col_offset,
                        end_line_number=getattr(node, "end_lineno", node.lineno),
                        end_column_number=getattr(node, "end_col_offset", None),
                        signature=signature,
                        docstring=docstring,
                        parent_name=parent_name,
                    )
                )
                for arg_node_type_list in [
                    node.args.args,
                    node.args.posonlyargs,
                    node.args.kwonlyargs,
                ]:
                    for arg_node in arg_node_type_list:
                        definitions_found.append(
                            DefinitionInfo(
                                name=arg_node.arg,
                                def_type="parameter",
                                file_path=file_path,
                                line_number=arg_node.lineno,
                                column_number=arg_node.col_offset,
                                end_line_number=getattr(
                                    arg_node, "end_lineno", arg_node.lineno
                                ),
                                end_column_number=getattr(
                                    arg_node, "end_col_offset", None
                                ),
                                parent_name=node.name,
                            )
                        )
                if node.args.vararg:
                    definitions_found.append(
                        DefinitionInfo(
                            name=node.args.vararg.arg,
                            def_type="parameter",
                            file_path=file_path,
                            line_number=node.args.vararg.lineno,
                            column_number=node.args.vararg.col_offset,
                            parent_name=node.name,
                        )
                    )
                if node.args.kwarg:
                    definitions_found.append(
                        DefinitionInfo(
                            name=node.args.kwarg.arg,
                            def_type="parameter",
                            file_path=file_path,
                            line_number=node.args.kwarg.lineno,
                            column_number=node.args.kwarg.col_offset,
                            parent_name=node.name,
                        )
                    )
            elif isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node, clean=False)
                bases = [ast.unparse(b) for b in node.bases]
                signature = f"class {node.name}"
                if bases:
                    signature += f"({', '.join(bases)})"
                definitions_found.append(
                    DefinitionInfo(
                        name=node.name,
                        def_type="class",
                        file_path=file_path,
                        line_number=node.lineno,
                        column_number=node.col_offset,
                        end_line_number=getattr(node, "end_lineno", node.lineno),
                        end_column_number=getattr(node, "end_col_offset", None),
                        signature=signature,
                        docstring=docstring,
                        parent_name=parent_name,
                    )
                )
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        definitions_found.append(
                            DefinitionInfo(
                                name=target.id,
                                def_type="variable",
                                file_path=file_path,
                                line_number=target.lineno,
                                column_number=target.col_offset,
                                end_line_number=getattr(
                                    target, "end_lineno", target.lineno
                                ),
                                end_column_number=getattr(
                                    target, "end_col_offset", None
                                ),
                                parent_name=parent_name,
                            )
                        )
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    definitions_found.append(
                        DefinitionInfo(
                            name=node.target.id,
                            def_type="variable",
                            file_path=file_path,
                            line_number=node.target.lineno,
                            column_number=node.target.col_offset,
                            end_line_number=getattr(
                                node.target, "end_lineno", node.target.lineno
                            ),
                            end_column_number=getattr(
                                node.target, "end_col_offset", None
                            ),
                            parent_name=parent_name,
                        )
                    )
        return definitions_found

    def get_context_for_ai(self, code_string: str, file_path: Path) -> Dict[str, Any]:
        """
        Analyzes a file and returns a summary of its context for AI prompts.

        Uses a cache to avoid re-parsing unchanged files.

        Args:
            code_string: The full source code of the file.
            file_path: The path to the file.

        Returns:
            A dictionary containing string representations of the file's
            imports and definitions, formatted for an AI prompt.
        """
        current_hash = hash(code_string)
        cached_entry = self.analysis_cache.get(file_path)
        if cached_entry and cached_entry.get("hash") == current_hash:
            return cached_entry["context_for_ai"]
        tree = self._parse_code(code_string, str(file_path))
        # Create string representations for the AI prompt
        imports = self.analyze_imports(code_string, file_path, tree=tree)
        definitions = self.analyze_definitions(code_string, file_path, tree=tree)
        import_strings = [repr(imp) for imp in imports]
        definition_strings = []
        for defn in definitions:
            sig = (
                f"signature='{defn.signature}'" if defn.signature else "signature=None"
            )
            ds_preview = "None"
            if defn.docstring:
                first_line_doc = defn.docstring.strip().split("\n", 1)[0]
                ds_preview = f"'{first_line_doc[:50].replace(chr(10), ' ') + ('...' if len(first_line_doc) > 50 or '\n' in defn.docstring else '')}'"
            definition_strings.append(
                f"DefinitionInfo(name='{defn.name}', type='{defn.def_type}', {sig}, docstring_preview={ds_preview})"
            )
        context_for_ai = {
            "file_path": str(file_path.name),
            "imports_context": (
                "Imports: [" + ", ".join(import_strings) + "]"
                if import_strings
                else "Imports: []"
            ),
            "definitions_context": (
                "Definitions: [" + ", ".join(definition_strings) + "]"
                if definition_strings
                else "Definitions: []"
            ),
        }
        raw_context_data = {
            "imports": imports,
            "definitions": definitions,
            "ast_tree": tree,
        }

        # Store raw data and formatted context in cache
        self.analysis_cache[file_path] = {
            "hash": current_hash,
            "context_for_ai": context_for_ai,
            "raw_data": raw_context_data,
        }
        return context_for_ai

    def _find_node_at_position(
        self, tree: ast.AST, line: int, col: int
    ) -> Optional[ast.AST]:
        # print(f"VERBOSE_DEBUG: _find_node_at_position START L{line}:C{col}"); sys.stdout.flush()
        best_node: Optional[ast.AST] = None
        min_size = float("inf")

        for node in ast.walk(tree):
            if not hasattr(node, "lineno") or not hasattr(node, "col_offset"):
                continue

            node_line_start = node.lineno
            node_col_start = node.col_offset
            node_line_end = getattr(node, "end_lineno", node_line_start)
            node_col_end = getattr(node, "end_col_offset", node_col_start + 1)

            # --- Start: Refined column end calculation for specific identifier nodes ---
            # This helps to define the clickable/identifiable span more accurately.
            current_node_effective_col_start = node_col_start
            current_node_effective_col_end = node_col_end

            if isinstance(node, ast.Name):
                current_node_effective_col_end = node_col_start + len(node.id)
            elif isinstance(node, ast.arg):
                current_node_effective_col_end = node_col_start + len(node.arg)
            elif isinstance(node, ast.alias):
                # For 'import name as asname' or 'from module import name as asname'
                # The node spans from 'name' to 'asname'. col_offset is start of 'name'.
                # We want this node to be a candidate if cursor is on 'name' OR 'asname'.
                # end_col_offset for ast.alias (if present) is usually reliable.
                current_node_effective_col_end = getattr(
                    node,
                    "end_col_offset",
                    node_col_start
                    + len(node.name)
                    + (len(node.asname) + 4 if node.asname else 0),
                )
            elif isinstance(node, ast.Attribute):
                # For 'value.attr', node.col_offset is start of 'value'.
                # node.end_col_offset is end of 'attr'. This span is usually correct.
                # We might later distinguish if cursor is on 'value' or 'attr'.
                pass  # Use getattr default for end_col_offset
            # --- End: Refined column end calculation ---

            cursor_in_node = False
            if node_line_start == line and node_line_end == line:
                if (
                    current_node_effective_col_start
                    <= col
                    < current_node_effective_col_end
                ):
                    cursor_in_node = True
            elif node_line_start == line and line < node_line_end:
                if current_node_effective_col_start <= col:
                    cursor_in_node = True
            elif node_line_start < line and line == node_line_end:
                if col < current_node_effective_col_end:
                    cursor_in_node = True
            elif node_line_start < line < node_line_end:
                cursor_in_node = True

            if cursor_in_node:
                current_size = (node_line_end - node_line_start + 1) * 10000 + (
                    current_node_effective_col_end - current_node_effective_col_start
                )

                if current_size < min_size:
                    min_size = current_size
                    best_node = node
                elif current_size == min_size and best_node is not None:
                    # Tie-breaking: Prefer more specific identifier types or "deeper" nodes
                    is_specific_current = isinstance(
                        node, (ast.Name, ast.Attribute, ast.arg, ast.alias)
                    )
                    is_specific_best = isinstance(
                        best_node, (ast.Name, ast.Attribute, ast.arg, ast.alias)
                    )

                    if is_specific_current and not is_specific_best:
                        best_node = node
                    elif (
                        is_specific_current == is_specific_best
                    ):  # Both specific or both not
                        # Prefer node that is part of an Attribute if cursor is on the attribute's name
                        if isinstance(node, ast.Attribute) and hasattr(
                            node.value, "end_col_offset"
                        ):
                            # Heuristic: if column is after the 'value' part (value + dot)
                            if col > node.value.end_col_offset:  # type: ignore
                                best_node = node  # Prefer the attribute itself
                        elif isinstance(best_node, ast.Attribute) and hasattr(
                            best_node.value, "end_col_offset"
                        ):
                            if col <= best_node.value.end_col_offset:
                                pass  # Keep best_node (which might be the value part)
                        # Fallback to deeper node (more fields)
                        elif len(list(ast.iter_fields(node))) > len(
                            list(ast.iter_fields(best_node))
                        ):
                            best_node = node
        # print(f"VERBOSE_DEBUG: _find_node_at_position END. Best: {type(best_node).__name__ if best_node else 'None'}"); sys.stdout.flush()
        return best_node

    def find_definition_at_position(
        self, code_string: str, file_path: Path, line_number: int, column_number: int
    ) -> Optional[DefinitionInfo]:
        """
        Finds the definition of an identifier at a specific cursor position.

        Args:
            code_string: The source code of the file.
            file_path: The path of the file.
            line_number: The 1-based line number of the cursor.
            column_number: The 0-based column number of the cursor.

        Returns:
            A `DefinitionInfo` object if a definition is found, otherwise None.
        """

        current_hash = hash(code_string)
        cached_entry = self.analysis_cache.get(file_path)
        raw_data: Optional[Dict[str, Any]] = None
        tree: Optional[ast.AST] = None

        # Ensure the file is analyzed and cached if not already
        if cached_entry and cached_entry.get("hash") == current_hash:
            raw_data = cached_entry.get("raw_data")
            tree = raw_data.get("ast_tree") if raw_data else None

        if not tree:
            tree = self._parse_code(code_string, str(file_path))
            if not tree:
                return None
            if not raw_data:
                imports = self.analyze_imports(code_string, file_path, tree=tree)
                definitions = self.analyze_definitions(
                    code_string, file_path, tree=tree
                )
                raw_data = {
                    "imports": imports,
                    "definitions": definitions,
                    "ast_tree": tree,
                }
                self.analysis_cache[file_path] = {
                    "hash": current_hash,
                    "context_for_ai": {},
                    "raw_data": raw_data,
                }

        if not raw_data or "definitions" not in raw_data or "imports" not in raw_data:
            return None

        # Find the smallest AST node that contains the cursor position
        node_at_cursor = self._find_node_at_position(tree, line_number, column_number)
        if not node_at_cursor:
            return None

        # Determine the name of the identifier at the cursor
        identifier_name: Optional[str] = None
        node_is_definition_itself = False

        # --- Start: Identifier Extraction and Definition Check ---
        if isinstance(node_at_cursor, ast.Name):
            identifier_name = node_at_cursor.id
        elif isinstance(node_at_cursor, ast.Attribute):
            identifier_name = node_at_cursor.attr
            # Check if cursor is on the 'attr' part specifically
            if (
                hasattr(node_at_cursor.value, "end_col_offset")
                and node_at_cursor.lineno == line_number
                and column_number > node_at_cursor.value.end_col_offset
            ):
                pass  # identifier_name is already node.attr
            else:  # Cursor is likely on the 'value' part of the attribute
                # Try to re-evaluate node_at_cursor to be the 'value' if it's an ast.Name
                if isinstance(node_at_cursor.value, ast.Name):
                    # Check if the 'value' node itself is a better fit for the cursor position
                    val_node = node_at_cursor.value
                    val_node_col_end = val_node.col_offset + len(val_node.id)
                    if (
                        val_node.lineno == line_number
                        and val_node.col_offset <= column_number < val_node_col_end
                    ):
                        # print(f"VERBOSE_DEBUG: Cursor on value part of Attribute, switching node_at_cursor to Name: {val_node.id}")
                        node_at_cursor = (
                            val_node  # Effectively re-assign node_at_cursor
                        )
                        identifier_name = val_node.id
        elif isinstance(node_at_cursor, ast.arg):
            identifier_name = node_at_cursor.arg
            # Search for the definition of the found identifier name
            # 1. Search in local definitions
            for defn in raw_data.get(
                "definitions", []
            ):  # Check if this is the definition of the param
                if (
                    defn.name == identifier_name
                    and defn.def_type == "parameter"
                    and defn.line_number == node_at_cursor.lineno
                    and defn.column_number == node_at_cursor.col_offset
                ):
                    node_is_definition_itself = True
                    break
        elif isinstance(
            node_at_cursor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            keyword_len = (
                len("def ")
                if isinstance(node_at_cursor, (ast.FunctionDef, ast.AsyncFunctionDef))
                else len("class ")
            )
            name_start_col = node_at_cursor.col_offset + keyword_len
            name_end_col = name_start_col + len(node_at_cursor.name)
            if (
                node_at_cursor.lineno == line_number
                and name_start_col <= column_number < name_end_col
            ):
                identifier_name = node_at_cursor.name
                node_is_definition_itself = True
            else:
                return None
        elif isinstance(node_at_cursor, ast.Call):
            if isinstance(node_at_cursor.func, ast.Name):
                identifier_name = node_at_cursor.func.id
            elif isinstance(node_at_cursor.func, ast.Attribute):
                identifier_name = node_at_cursor.func.attr
        elif isinstance(node_at_cursor, ast.alias):
            identifier_name = (
                node_at_cursor.asname if node_at_cursor.asname else node_at_cursor.name
            )
            # Check if the cursor is on this specific alias node
            # This logic assumes ast.alias node spans the name and asname if present
            name_col_start = node_at_cursor.col_offset
            name_col_end = getattr(
                node_at_cursor,
                "end_col_offset",
                name_col_start
                + len(node_at_cursor.name)
                + (len(node_at_cursor.asname) + 4 if node_at_cursor.asname else 0),
            )
            if (
                node_at_cursor.lineno == line_number
                and name_col_start <= column_number < name_col_end
            ):
                node_is_definition_itself = True
        # --- End: Identifier Extraction and Definition Check ---

        if not identifier_name:
            return None
        # print(f"VERBOSE_DEBUG: find_definition_at_position - Identifier: '{identifier_name}', IsDefItself: {node_is_definition_itself}")

        # 1. Priority: If cursor is on the definition itself
        if node_is_definition_itself:
            for defn in raw_data.get("definitions", []):
                if (
                    defn.name == identifier_name
                    and defn.line_number == node_at_cursor.lineno
                    and defn.column_number == node_at_cursor.col_offset
                ):  # Exact match of definition start
                    return defn
            for imp_info in raw_data.get("imports", []):
                for alias_node in imp_info.alias_nodes:
                    current_name_in_import = (
                        alias_node.asname if alias_node.asname else alias_node.name
                    )
                    if (
                        current_name_in_import == identifier_name
                        and alias_node.lineno == node_at_cursor.lineno
                    ):
                        # Determine the correct column for the DefinitionInfo based on what part of alias was hit
                        def_col = (
                            alias_node.col_offset
                        )  # Default to start of original name
                        if (
                            alias_node.asname == identifier_name
                        ):  # Cursor was on the 'as name' part
                            # Estimate start of asname
                            def_col = (
                                alias_node.col_offset
                                + len(alias_node.name)
                                + len(" as ")
                            )

                        if def_col == node_at_cursor.col_offset or (
                            alias_node.asname == identifier_name
                            and column_number
                            >= (
                                alias_node.col_offset
                                + len(alias_node.name)
                                + len(" as ")
                            )
                        ):  # Cursor on alias
                            return DefinitionInfo(
                                name=identifier_name,
                                def_type="import",
                                file_path=imp_info.file_path or file_path,
                                line_number=alias_node.lineno,
                                column_number=def_col,  # Use the determined column of the name/alias
                                signature=repr(imp_info),
                            )

        # 2. General search for identifier_name (if not on the definition itself)
        for defn in raw_data.get("definitions", []):
            if defn.name == identifier_name:
                return defn

        for imp_info in raw_data.get("imports", []):
            for original_name, alias_name_in_tuple in imp_info.imported_names:
                actual_name_used_in_code = (
                    alias_name_in_tuple if alias_name_in_tuple else original_name
                )
                if actual_name_used_in_code == identifier_name:
                    # Find the specific ast.alias node to get its precise location for the DefinitionInfo
                    col_for_def = (
                        imp_info.column_number
                    )  # Default to start of import statement
                    line_for_def = imp_info.line_number
                    for an_node in imp_info.alias_nodes:
                        name_part = an_node.name
                        asname_part = an_node.asname
                        if (asname_part == identifier_name) or (
                            name_part == identifier_name and not asname_part
                        ):
                            line_for_def = an_node.lineno
                            if (
                                asname_part == identifier_name
                            ):  # Cursor was on the alias part
                                col_for_def = (
                                    an_node.col_offset + len(name_part) + len(" as ")
                                )
                            else:  # Cursor was on the original name part
                                col_for_def = an_node.col_offset
                            break
                    return DefinitionInfo(
                        name=identifier_name,
                        def_type="import",
                        file_path=(
                            imp_info.file_path if imp_info.file_path else file_path
                        ),
                        line_number=line_for_def,
                        column_number=col_for_def,
                        signature=repr(imp_info),
                    )
        return None

    def clear_cache(self):
        self.analysis_cache.clear()
        logger.info("CodeContextAnalyzer: Analyse-Cache geleert.")

    def invalidate_cache_for_file(self, file_path: Path):
        if file_path in self.analysis_cache:
            del self.analysis_cache[file_path]
            logger.info(
                f"CodeContextAnalyzer: Cache für Datei '{file_path.name}' invalidiert."
            )
