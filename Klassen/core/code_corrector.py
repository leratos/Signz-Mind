# Klassen/code_corrector.py
# -*- coding: utf-8 -*-

"""
CodeCorrector: AI-assisted code correction utility for Python source code.

- Attempts to fix syntax errors using an LLM and validates with the AST parser.
- Handles code extraction, whitespace/formatting, and applies autopep8.
- Returns correction results and logs steps for debugging.
"""
import ast
import autopep8
import logging
import re
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING, Tuple, Union
import textwrap

from ..core import config as global_config
import torch

if TYPE_CHECKING:
    from .inference_service import InferenceService
    from transformers import PreTrainedTokenizer  # type: ignore

logger = logging.getLogger(__name__)
ReplacementMode = Literal["standard", "heuristic_except_only", "single_line_focused"]
ErrorDetails = Dict[str, Union[str, int, None]]
CorrectionResult = Dict[str, Any]


class CodeCorrector:
    """AI-assisted code correction utility for Python source code."""
    MAX_CORRECTION_ATTEMPTS_DEFAULT = 5

    def __init__(self, inference_service: "InferenceService", config_dict: dict):
        """Initialize the code corrector."""
        self.inference_service = inference_service
        self.config_dict = config_dict
        self.max_correction_attempts = self.config_dict.get(
            "max_correction_attempts", self.MAX_CORRECTION_ATTEMPTS_DEFAULT
        )
        logger.info(
            f"CodeCorrector initialisiert mit max_attempts={self.max_correction_attempts}"
        )

    def _get_leading_whitespace(self, line: str) -> str:
        """Return the leading whitespace of a line."""
        match = re.match(r"^([ \t]*)", line)
        return match.group(1) if match else ""

    @staticmethod
    def _extract_code_from_ai_response(generated_text_raw: str, tokenizer: Optional["PreTrainedTokenizer"], log_list: List[str]) -> str:  # type: ignore
        """Extracts code block from AI response text, handling multiple markdown/code block formats."""
        cleaned_suggestion = ""
        match_block = re.search(
            r"```python\s*\n(.*?)(?:\n```|\Z)",
            generated_text_raw,
            re.DOTALL | re.IGNORECASE,
        )
        if match_block:
            cleaned_suggestion = match_block.group(1).strip()
            log_list.append(
                f"Code aus ```python Block extrahiert:\n---\n{cleaned_suggestion}\n---"
            )
        else:
            match_any_ticks = re.search(
                r"```\s*\n(.*?)(?:\n```|\Z)",
                generated_text_raw,
                re.DOTALL | re.IGNORECASE,
            )
            if match_any_ticks:
                cleaned_suggestion = match_any_ticks.group(1).strip()
                log_list.append(
                    f"Code aus generischem ``` Block extrahiert:\n---\n{cleaned_suggestion}\n---"
                )
            else:
                eos_token_pattern = (
                    re.escape(tokenizer.eos_token)
                    if tokenizer and tokenizer.eos_token
                    else r"</s>_NEVER_MATCH_THIS_EOS_FALLBACK_"
                )
                match_fallback = re.search(
                    r"^(.*?)(\n```|\Z|{})".format(eos_token_pattern),
                    generated_text_raw,
                    re.DOTALL | re.MULTILINE,
                )
                if match_fallback:
                    cleaned_suggestion = match_fallback.group(1).strip()
                    log_list.append(
                        f"Code via Fallback (bis EOS oder Ende des Strings):\n---\n{cleaned_suggestion}\n---"
                    )
                else:
                    cleaned_suggestion = generated_text_raw.strip()
                    log_list.append(
                        f"Code via Fallback (gesamter Text, keine Begrenzer gefunden):\n---\n{cleaned_suggestion}\n---"
                    )

        # Remove leading template/boilerplate phrases and code fences
        patterns_to_remove = [
            r"^(Okay, here's the corrected code:|Here's the corrected version:|Here is the corrected code snippet:)",
            r"```python",
            r"```",
        ]
        temp_suggestion = cleaned_suggestion
        for pattern in patterns_to_remove:
            temp_suggestion = re.sub(
                pattern, "", temp_suggestion, flags=re.IGNORECASE | re.MULTILINE
            ).strip()

        lines = temp_suggestion.splitlines()
        final_lines = []
        for line in lines:
            if "HF_LIBRARIES_AVAILABLE =" not in line or line.strip().startswith("#"):
                final_lines.append(line)
        cleaned_suggestion = "\n".join(final_lines).strip()
        return cleaned_suggestion

    def _find_original_try_indent(
        self,
        original_lines: List[str],
        context_snippet_start_idx: int,
        log_list: List[str],
    ) -> Optional[str]:
        if context_snippet_start_idx < len(original_lines) and original_lines[
            context_snippet_start_idx
        ].strip().startswith("try:"):
            return self._get_leading_whitespace(
                original_lines[context_snippet_start_idx]
            )
        search_from_index = context_snippet_start_idx - 1
        if search_from_index < 0:
            return None
        for i in range(search_from_index, -1, -1):
            if i < len(original_lines) and original_lines[i].strip().startswith("try:"):
                return self._get_leading_whitespace(original_lines[i])
        return None

    def _reconstruct_except_block(
        self, suggestion_lines: List[str], original_try_indent: str, log_list: List[str]
    ) -> Optional[str]:
        temp_except_clause: Optional[str] = None
        temp_except_body_lines: List[str] = []
        in_except_block_parsing = False
        for line_sugg in suggestion_lines:
            stripped_line = line_sugg.strip()
            if not stripped_line:
                if in_except_block_parsing:
                    temp_except_body_lines.append(line_sugg)
                continue
            if (
                temp_except_clause is None
                and stripped_line.startswith("except ")
                and stripped_line.endswith(":")
            ):
                temp_except_clause = stripped_line
                in_except_block_parsing = True
            elif in_except_block_parsing:
                if line_sugg.startswith(" ") or not stripped_line:
                    temp_except_body_lines.append(line_sugg.lstrip())
                else:
                    break
        if temp_except_clause:
            reconstructed_list = [f"{original_try_indent}{temp_except_clause.strip()}"]
            if not any(line.strip() for line in temp_except_body_lines):
                temp_except_body_lines = ["pass"]
            body_indent = original_try_indent + "    "
            for exc_line in temp_except_body_lines:
                reconstructed_list.append(f"{body_indent}{exc_line}")
            return "\n".join(reconstructed_list).rstrip() + "\n"
        return None

    def _apply_indentation_fix(
        self,
        suggestion_lines: List[str],
        target_base_indent_str: str,
        log_list: List[str],
    ) -> str:
        log_list.append(
            f"_apply_indentation_fix: AI suggestion lines: {suggestion_lines}"
        )
        if not suggestion_lines:
            log_list.append(
                "  Keine Vorschlagszeilen von KI, gebe leeren String zurück."
            )
            return ""
        log_list.append(
            f"  Ziel-Basiseinrückung für diesen Block: '{repr(target_base_indent_str)}'"
        )
        if all(not line.strip() for line in suggestion_lines):
            log_list.append("  KI-Vorschlag besteht nur aus Whitespace/leeren Zeilen.")
            return "\n" * len(suggestion_lines)
        try:
            dedented_block = textwrap.dedent("\n".join(suggestion_lines))
            log_list.append(f"  Block nach textwrap.dedent:\n'''\n{dedented_block}'''")
        except Exception as e:
            log_list.append(
                f"  Fehler bei textwrap.dedent: {e}. Fallback auf manuelle Dedent-Logik."
            )
            if len(suggestion_lines) > 1:
                min_indent = float("inf")
                for line in suggestion_lines:
                    if line.strip():
                        min_indent = min(
                            min_indent, len(self._get_leading_whitespace(line))
                        )
                if min_indent == float("inf"):
                    dedented_block = "\n".join(suggestion_lines)
                else:
                    temp_dedented_lines = []
                    for line in suggestion_lines:
                        if line.strip():
                            temp_dedented_lines.append(line[min_indent:])
                        else:
                            temp_dedented_lines.append(line)
                    dedented_block = "\n".join(temp_dedented_lines)
            elif suggestion_lines:
                dedented_block = suggestion_lines[0].lstrip()
            else:
                dedented_block = ""
            log_list.append(f"  Block nach manuellem Dedent:\n'''\n{dedented_block}'''")

        result_lines = []
        for line in dedented_block.splitlines():
            if line.strip():
                result_lines.append(target_base_indent_str + line)
            else:
                result_lines.append("")
        for i, line_val in enumerate(result_lines):
            log_list.append(f"  Final re-indented AI line {i+1}: '{repr(line_val)}'")
        final_block = "\n".join(result_lines)
        if final_block.strip():
            return final_block.rstrip("\n") + "\n"
        elif suggestion_lines:
            if all(not s.strip() for s in suggestion_lines):
                return "\n" * len(suggestion_lines)
            return "\n"
        return ""

    def _build_ai_prompt(
        self,
        context_code_snippet: str,
        error_details_for_prompt: Dict[str, Any],
        log_list: List[str],
        file_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        error_msg = error_details_for_prompt.get("msg", "Unknown error")
        error_line_in_snippet = error_details_for_prompt.get("error_line_in_snippet", 0)
        original_error_line = error_details_for_prompt.get("original_lineno", 0)
        error_text_from_exception = str(error_details_for_prompt.get("text", ""))
        context_info_str_parts = []
        if file_context:
            imports_ctx = file_context.get("imports_context", "")
            if imports_ctx and imports_ctx != "Imports: []":
                try:
                    raw_imports_list_str = imports_ctx.replace("Imports: ", "").strip()
                    if raw_imports_list_str != "[]":
                        parsed_list = ast.literal_eval(raw_imports_list_str)
                        if parsed_list:
                            context_info_str_parts.append(
                                "'''python\n# Relevant imports from the file:"
                            )
                            for imp_item_repr in parsed_list:
                                actual_import_statement = (
                                    self._format_import_info_for_prompt(
                                        str(imp_item_repr)
                                    )
                                )
                                if actual_import_statement:
                                    context_info_str_parts.append(
                                        f"# {actual_import_statement}"
                                    )
                                else:
                                    context_info_str_parts.append(
                                        f"# {str(imp_item_repr)[:120]}"
                                    )
                            context_info_str_parts.append("'''")
                except Exception as e:
                    logger.debug(
                        f"Could not parse 'imports_context' for prompt: {e}. Using raw string."
                    )
                    context_info_str_parts.append(
                        f"# Raw Imports Context: {imports_ctx}"
                    )
            defs_ctx = file_context.get("definitions_context", "")
            if defs_ctx and defs_ctx != "Definitions: []":
                try:
                    raw_defs_list_str = defs_ctx.replace("Definitions: ", "").strip()
                    if raw_defs_list_str != "[]":
                        parsed_list = ast.literal_eval(raw_defs_list_str)
                        if parsed_list:
                            context_info_str_parts.append(
                                "'''python\n# Relevant definitions from the file (summary):"
                            )
                            for def_item_repr in parsed_list:
                                actual_definition_summary = (
                                    self._format_definition_info_for_prompt(
                                        str(def_item_repr)
                                    )
                                )
                                if actual_definition_summary:
                                    context_info_str_parts.append(
                                        f"# {actual_definition_summary}"
                                    )
                                else:
                                    context_info_str_parts.append(
                                        f"# {str(def_item_repr)[:150]}"
                                    )
                            context_info_str_parts.append("'''")
                except Exception as e:
                    logger.debug(
                        f"Could not parse 'definitions_context' for prompt: {e}. Using raw string."
                    )
                    context_info_str_parts.append(
                        f"# Raw Definitions Context: {defs_ctx}"
                    )
            if context_info_str_parts:
                context_info_str = (
                    "### Relevant Code Context from the Current File:\n"
                    + "\n".join(context_info_str_parts)
                    + "\n### --- End of Context ---\n\n"
                )
            elif file_context:
                context_info_str = "# Note: No specific imports or definitions could be reliably extracted from the broader file context (or file has syntax errors).\n\n"
            else:
                context_info_str = (
                    "# Note: No broader file context was provided for this snippet.\n\n"
                )
        else:
            context_info_str = (
                "# Note: No broader file context was provided for this snippet.\n\n"
            )
        prompt_intro = "You are an expert Python programming assistant. Your task is to fix a syntax error in the Python code snippet below."
        snippet_lines = context_code_snippet.rstrip().split("\n")
        marked_snippet_lines = []
        line_to_mark_idx = -1
        if error_text_from_exception.strip():
            for i, line_content in enumerate(snippet_lines):
                if line_content.strip() == error_text_from_exception.strip():
                    line_to_mark_idx = i
                    break
        if (
            line_to_mark_idx == -1
            and error_line_in_snippet > 0
            and error_line_in_snippet <= len(snippet_lines)
        ):
            line_to_mark_idx = error_line_in_snippet - 1
        elif line_to_mark_idx == -1:
            line_to_mark_idx = 0 if snippet_lines else -1
        for i, line_content in enumerate(snippet_lines):
            if i == line_to_mark_idx:
                marked_snippet_lines.append(f"{line_content}  # <<< ERROR: {error_msg}")
            else:
                marked_snippet_lines.append(line_content)
        marked_context_code_snippet = "\n".join(marked_snippet_lines)
        error_line_content_for_prompt = (
            error_text_from_exception.strip()
            if error_text_from_exception.strip()
            else (
                snippet_lines[line_to_mark_idx].strip()
                if line_to_mark_idx != -1 and line_to_mark_idx < len(snippet_lines)
                else "N/A"
            )
        )
        prompt_error_details = (
            f"The Python code snippet below contains a syntax error.\n"
            f"The error is: '{error_msg}'.\nThe parser indicated an issue at or near line {error_line_in_snippet} of THIS SNIPPET "
            f"(which corresponds to original line {original_error_line} in the file).\n"
            f"The line content most likely causing this is '{error_line_content_for_prompt}', marked with '# <<< ERROR'."
        )

        prompt_instructions_list = [
            "1. Carefully analyze the 'Code Snippet with Error', the error message, and especially the line marked with '# <<< ERROR'.",
            "2. Identify the precise syntax error highlighted by the marker.",
            "3. IMPORTANT: Return the ENTIRE corrected code snippet that was provided to you (between ```python and ``` in the prompt). Your response must include all original lines from the snippet, with the identified error fixed. Do NOT omit any lines from the original snippet, even if they are not directly part of the error, unless the correction logically requires removing them (e.g., a duplicate line). For example, if the snippet was 4 lines long and you corrected line 2, your response MUST contain all 4 lines, with line 2 corrected.",
            "4. If the error is an indentation error, correct the indentation to what it should be relative to the snippet's structure. Consider the overall class or function structure if an unindent error occurs.",
            "5. If a code block is empty and causes an error (e.g., after 'if x:', 'try:', 'def func():'), your correction should be to add a 'pass' statement with the correct indentation for that block.",
            "6. VERY IMPORTANT: Output ONLY the raw corrected Python code line(s). Do NOT include any explanations, introductory phrases, or markdown formatting like ```python or ```.",
        ]
        full_instructions = "\n".join(prompt_instructions_list)
        prompt = (
            f"{context_info_str}{prompt_intro}\n\n{prompt_error_details}\n\n"
            f"Code Snippet with Error:\n```python\n{marked_context_code_snippet}\n```\n\n"
            f"Instructions for your response:\n{full_instructions}\n\n"
            f"Corrected Code Line(s) ONLY (do not use markdown):\n"
        )
        logger.debug(
            f"Prompt an KI (CodeCorrector) gesendet. Kontext enthalten: {bool(file_context and any(part.strip() for part in context_info_str_parts))}"
        )
        log_list.append(f"Vollständiger Prompt an KI (CodeCorrector):\n{prompt}")
        return prompt

    def _format_import_info_for_prompt(self, import_info_repr: str) -> Optional[str]:
        try:
            is_from_import_match = re.search(r"from='([^']*)'", import_info_repr)
            imports_match = re.search(r"import=\[([^\]]*)\]", import_info_repr)
            level_match = re.search(r"level=(\d+)", import_info_repr)
            level = int(level_match.group(1)) if level_match else 0
            if not imports_match:
                return None
            imported_items_str = imports_match.group(1)
            raw_items = []
            balance = 0
            current_item = ""
            for char_val in imported_items_str:
                if char_val == "(":
                    balance += 1
                elif char_val == ")":
                    balance -= 1
                elif char_val == "," and balance == 0:
                    raw_items.append(current_item.strip())
                    current_item = ""
                    continue
                current_item += char_val
            if current_item.strip():
                raw_items.append(current_item.strip())
            parsed_items = []
            for item_str in raw_items:
                item_str = item_str.strip("'\" ")
                if item_str.startswith("(") and item_str.endswith(")"):
                    try:
                        alias_tuple = ast.literal_eval(item_str)
                        if isinstance(alias_tuple, tuple) and len(alias_tuple) == 2:
                            parsed_items.append(f"{alias_tuple[0]} as {alias_tuple[1]}")
                        else:
                            parsed_items.append(item_str)
                    except (ValueError, SyntaxError):
                        parsed_items.append(item_str)
                else:
                    parsed_items.append(item_str)
            import_list_str = ", ".join(parsed_items)
            if is_from_import_match:
                module_name = is_from_import_match.group(1)
                relative_prefix = "." * level
                return f"from {relative_prefix}{module_name} import {import_list_str}"
            else:
                return f"import {import_list_str}"
        except Exception as e:
            logger.debug(
                f"Error formatting ImportInfo repr for prompt: {import_info_repr} -> {e}"
            )
            return None

    def _format_definition_info_for_prompt(self, def_info_repr: str) -> Optional[str]:
        try:
            name_match = re.search(r"name='([^']*)'", def_info_repr)
            type_match = re.search(r"type='([^']*)'", def_info_repr)
            sig_match = re.search(r"signature='([^']*)'", def_info_repr)
            if not (name_match and type_match):
                return None
            name = name_match.group(1)
            def_type = type_match.group(1)
            signature_preview = ""
            if sig_match:
                full_signature = sig_match.group(1)
                if full_signature != "None":
                    if def_type in ["function", "async function"]:
                        params_match = re.search(r"\((.*)\)", full_signature)
                        if params_match:
                            signature_preview = f"({params_match.group(1)})"
                        else:
                            signature_preview = "()"
                    elif def_type == "class":
                        bases_match = re.search(r"\((.*)\)", full_signature)
                        if bases_match:
                            signature_preview = f"({bases_match.group(1)})"
            elif def_type in ["function", "async function", "class"]:
                signature_preview = "(...)"
            return f"{def_type} {name}{signature_preview}"
        except Exception as e:
            logger.debug(
                f"Error formatting DefinitionInfo repr for prompt: {def_info_repr} -> {e}"
            )
            return None

    def _get_ai_correction(
        self,
        context_code_snippet: str,
        error_details_for_prompt: Dict[str, Any],
        log_list: List[str],
        file_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not torch:
            log_list.append("FEHLER: PyTorch nicht verfügbar für _get_ai_correction.")
            return None
        model_tokenizer_tuple = self.inference_service.get_loaded_model_and_tokenizer()
        if (
            model_tokenizer_tuple is None
            or model_tokenizer_tuple[0] is None
            or model_tokenizer_tuple[1] is None
        ):
            log_list.append(
                "FEHLER: KI-Modell/Tokenizer nicht verfügbar für _get_ai_correction."
            )
            return None
        model, tokenizer = model_tokenizer_tuple
        prompt = self._build_ai_prompt(
            context_code_snippet,
            error_details_for_prompt,
            log_list,
            file_context=file_context,
        )
        try:
            model.eval()
            device = next(model.parameters()).device
            model_max_len = self.config_dict.get(
                "hf_max_length", getattr(global_config, "HF_MAX_LENGTH", 1024)
            )
            max_new_tokens = self.config_dict.get("max_new_tokens_correction", 150)
            buffer_tokens = 50
            max_prompt_len_for_tokenizer = (
                model_max_len - max_new_tokens - buffer_tokens
            )
            if max_prompt_len_for_tokenizer <= 0:
                log_list.append(
                    f"FEHLER: Berechnete max_prompt_len_for_tokenizer ist <= 0 ({max_prompt_len_for_tokenizer})."
                )
                if "### Relevant Code Context" in prompt:
                    prompt_without_file_context = re.sub(
                        r"### Relevant Code Context.*?### --- End of Context ---\n\n",
                        "# Note: File context removed due to length limitations.\n\n",
                        prompt,
                        flags=re.DOTALL,
                    )
                    log_list.append(
                        "  Fallback: Entferne Dateikontext aus Prompt, um Länge zu reduzieren."
                    )
                    prompt = prompt_without_file_context
                if len(tokenizer.encode(prompt)) > max_prompt_len_for_tokenizer:
                    log_list.append(
                        "  Fallback fehlgeschlagen: Prompt immer noch zu lang."
                    )
                    return None
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=max_prompt_len_for_tokenizer,
                truncation=True,
            ).to(device)
            if inputs["input_ids"].nelement() == 0:
                log_list.append(
                    "FEHLER: Tokenizer hat leere input_ids für den Prompt zurückgegeben."
                )
                return None
            pad_id = (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            )
            if pad_id is None:
                pad_id = 0
            eos_id = (
                tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
            )
            with torch.no_grad():
                output_sequences = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    do_sample=self.config_dict.get("correction_do_sample", True),
                    temperature=self.config_dict.get("correction_temperature", 0.1),
                    top_k=self.config_dict.get("correction_top_k", 5),
                    top_p=self.config_dict.get("correction_top_p", 0.9),
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    early_stopping=True,
                )
            generated_ids = output_sequences[0][inputs["input_ids"].shape[-1]:]
            raw_output = tokenizer.decode(generated_ids, skip_special_tokens=False)
            return self._extract_code_from_ai_response(raw_output, tokenizer, log_list)
        except Exception as ai_e:
            log_list.append(f"Fehler während des KI-Aufrufs (CodeCorrector): {ai_e}")
            logger.error(
                f"Fehler KI-Aufruf in _get_ai_correction: {ai_e}", exc_info=True
            )
            return None

    def _process_ai_suggestion(
        self,
        cleaned_suggestion: str,
        original_code_lines: List[str],
        context_start_original_idx: int,
        target_base_indent_str: str,
        error_msg: str,
        log_list: List[str],
    ) -> Tuple[str, ReplacementMode, bool]:
        suggestion_lines = cleaned_suggestion.splitlines()
        is_single_line_suggestion = len(suggestion_lines) == 1
        processed_block: str
        replacement_mode: ReplacementMode = "standard"

        if is_single_line_suggestion and not (
            "expected an indented block" in error_msg
        ):
            replacement_mode = "single_line_focused"
            log_list.append(
                f"  KI-Vorschlag ist einzeilig und Fehler ist nicht 'expected indented block'. Modus: {replacement_mode}"
            )

        err_suggests_structure = (
            "expected 'except'" in error_msg
            or "expected 'finally'" in error_msg
            or "invalid syntax" in error_msg
            or "expected an indented block" in error_msg
        )
        ai_suggests_exc = (
            "except importerror" in cleaned_suggestion.lower()
            or "except:" in cleaned_suggestion.lower()
            or "excpt" in cleaned_suggestion.lower()
        )
        try_heuristic = err_suggests_structure or ai_suggests_exc
        heuristic_success = False

        if try_heuristic and replacement_mode != "single_line_focused":
            try_indent = self._find_original_try_indent(
                original_code_lines, context_start_original_idx, log_list
            )
            if try_indent is not None:
                reconstructed_block = self._reconstruct_except_block(
                    suggestion_lines, try_indent, log_list
                )
                if reconstructed_block is not None:
                    processed_block = reconstructed_block
                    heuristic_success = True
                    replacement_mode = "heuristic_except_only"
                    log_list.append(
                        "  Heuristik für except-Block erfolgreich angewendet."
                    )

        if not heuristic_success:
            processed_block = self._apply_indentation_fix(
                suggestion_lines, target_base_indent_str, log_list
            )
            if replacement_mode != "single_line_focused":
                replacement_mode = "standard"

        if not processed_block.strip() and cleaned_suggestion.strip():
            log_list.append(
                "WARNUNG: Verarbeitung ergab leeren Block trotz KI-Vorschlag. Fallback auf einfache Einrückung."
            )
            processed_block = "\n".join(
                [target_base_indent_str + line.lstrip() for line in suggestion_lines]
            )
            processed_block = (
                processed_block.rstrip("\n") + "\n" if processed_block.strip() else ""
            )
            if replacement_mode != "single_line_focused":
                replacement_mode = "standard"

        return processed_block, replacement_mode, is_single_line_suggestion

    def _build_prospective_code(
        self,
        original_lines: List[str],
        block_to_insert: str,
        replacement_mode: ReplacementMode,
        actual_err_idx_in_original: int,
        context_start_idx: int,
        context_end_idx: int,
        log_list: List[str],
    ) -> str:

        if replacement_mode == "single_line_focused":
            log_list.append(
                f"  Single-Line Focused Ersetzung: Ersetze Originalzeile {actual_err_idx_in_original + 1}."
            )
            corrected_line_with_newline = block_to_insert.rstrip("\n") + "\n"
            new_code_lines = list(original_lines)
            if actual_err_idx_in_original < len(new_code_lines):
                new_code_lines[actual_err_idx_in_original] = corrected_line_with_newline
                return "".join(new_code_lines)
            else:
                log_list.append(
                    f"  WARNUNG: actual_err_idx_in_original ({actual_err_idx_in_original}) außerhalb des Bereichs. Fallback auf Standardersetzung."
                )
                return "".join(
                    original_lines[:context_start_idx]
                    + [block_to_insert]
                    + original_lines[context_end_idx:]
                )

        elif replacement_mode == "heuristic_except_only":
            log_list.append(
                f"  Heuristische Ersetzung (z.B. except-Block): Ersetze von {context_start_idx + 1} bis {context_end_idx}."
            )
            return "".join(
                original_lines[:context_start_idx]
                + [block_to_insert]
                + original_lines[context_end_idx:]
            )

        else:
            log_list.append(
                f"  Standard-Ersetzung (mehrzeiliger Vorschlag): Originalzeilen von {context_start_idx + 1} bis {context_end_idx} werden ersetzt."
            )
            return "".join(
                original_lines[:context_start_idx]
                + [block_to_insert]
                + original_lines[context_end_idx:]
            )

    def _validate_correction(
        self, prospective_code: str, log_list: List[str]
    ) -> Tuple[bool, Optional[ErrorDetails]]:
        code_to_validate = (
            prospective_code.strip("\n") + "\n" if prospective_code.strip() else "\n"
        )
        try:
            ast.parse(code_to_validate)
            log_list.append("  AST-Validierung ERFOLGREICH")
            return True, None
        except SyntaxError as validation_err:
            new_err_details = self._extract_syntax_error_details(
                validation_err, prospective_code
            )
            log_list.append(
                f"  AST-Validierung FEHLGESCHLAGEN: {new_err_details.get('msg')} "
                f"in (potenziell neuer) Zeile {new_err_details.get('lineno')} "
                f"(Offset {new_err_details.get('offset')}) "
                f"Text: '{new_err_details.get('text')}'"
            )
            return False, new_err_details
        except Exception as general_parse_err:
            log_list.append(
                f"  AST-Validierung FEHLGESCHLAGEN (allgemeiner Parser-Fehler): {general_parse_err}"
            )
            fallback_details: ErrorDetails = {
                "msg": str(general_parse_err),
                "lineno": None,
                "offset": None,
                "text": None,
                "actual_line_idx": -1,
            }
            return False, fallback_details

    def _get_error_context_details(
        self,
        current_code_lines: List[str],
        actual_error_line_idx: int,
        error_msg: str = "",
    ) -> Tuple[str, int, int, int]:
        context_before = self.config_dict.get(
            "correction_context_before",
            getattr(global_config, "CORRECTION_CONTEXT_BEFORE", 2),
        )
        context_after = self.config_dict.get(
            "correction_context_after",
            getattr(global_config, "CORRECTION_CONTEXT_AFTER", 1),
        )
        start_idx = max(0, actual_error_line_idx - context_before)
        end_idx = min(
            len(current_code_lines), actual_error_line_idx + 1 + context_after
        )
        if (
            error_msg
            and isinstance(error_msg, str)
            and ("indent" in error_msg.lower() or "unindent" in error_msg.lower())
        ):
            current_line_stripped = ""
            if actual_error_line_idx < len(current_code_lines):
                current_line_stripped = current_code_lines[
                    actual_error_line_idx
                ].strip()
            if current_line_stripped.startswith(
                "def "
            ) or current_line_stripped.startswith("async def "):
                class_def_idx = -1
                search_limit = max(0, actual_error_line_idx - 20)
                for i in range(actual_error_line_idx - 1, search_limit - 1, -1):
                    if i < len(current_code_lines) and current_code_lines[
                        i
                    ].strip().startswith("class "):
                        class_def_idx = i
                        break
                if class_def_idx != -1:
                    new_start_idx = min(start_idx, class_def_idx)
                    if new_start_idx < start_idx:
                        logger.debug(
                            f"IndentationError: Snippet erweitert von Zeile {start_idx + 1} zu {new_start_idx + 1} um Klassendefinition einzubeziehen."
                        )
                        start_idx = new_start_idx
        context_lines = current_code_lines[start_idx:end_idx]
        context_str = "".join(context_lines)
        error_line_in_snippet = actual_error_line_idx - start_idx + 1
        return context_str, start_idx, end_idx, error_line_in_snippet

    def _extract_syntax_error_details(
        self, syn_e: SyntaxError, current_code: str
    ) -> ErrorDetails:
        err_line_num = syn_e.lineno if syn_e.lineno is not None else 0
        err_msg = syn_e.msg if syn_e.msg is not None else "Unbekannter Syntaxfehler"
        err_offset = syn_e.offset if syn_e.offset is not None else 0
        err_text_from_exception = syn_e.text if syn_e.text is not None else ""
        lines = current_code.splitlines(True)
        actual_error_line_idx = -1
        if err_line_num > 0 and err_line_num <= len(lines):
            actual_error_line_idx = err_line_num - 1
        elif not lines:
            actual_error_line_idx = 0
        elif lines:
            actual_error_line_idx = 0
        if actual_error_line_idx >= len(lines) and lines:
            actual_error_line_idx = len(lines) - 1
        if actual_error_line_idx < 0 and lines:
            actual_error_line_idx = 0
        text_for_error_line = err_text_from_exception
        if (
            not text_for_error_line.strip()
            and actual_error_line_idx != -1
            and actual_error_line_idx < len(lines)
        ):
            text_for_error_line = lines[actual_error_line_idx]
        return {
            "msg": err_msg,
            "lineno": err_line_num,
            "offset": err_offset,
            "text": text_for_error_line.strip(),
            "actual_line_idx": actual_error_line_idx,
        }

    def _single_correction_attempt(
        self,
        current_code: str,
        correction_log: List[str],
        attempt_num: int,
        max_attempts: int,
        file_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[ErrorDetails]]:
        code_to_parse = (
            current_code.strip("\n") + "\n" if current_code.strip() else "\n"
        )
        try:
            ast.parse(code_to_parse)
            correction_log.append(f"Syntax OK nach {attempt_num - 1} Versuchen.")
            return current_code, None
        except SyntaxError as syn_e:
            err_details_current = self._extract_syntax_error_details(
                syn_e, current_code
            )
            actual_err_idx = err_details_current.get("actual_line_idx", 0)
            if not isinstance(actual_err_idx, int) or actual_err_idx < 0:
                correction_log.append(
                    f"  Ungültiger Fehlerindex ({actual_err_idx}). Abbruch des Versuchs."
                )
                return current_code, err_details_current

            correction_log.append(
                f"--- Versuch {attempt_num}/{max_attempts}: SyntaxError: '{err_details_current.get('msg')}' "
                f"nahe Zeile {err_details_current.get('lineno')} (Index {actual_err_idx}) im aktuellen Code ---"
            )
            original_code_lines = current_code.splitlines(True)
            current_error_msg_for_snippet_logic = str(
                err_details_current.get("msg", "")
            )
            (
                snippet_text_for_ai,
                snippet_start_idx,
                snippet_end_idx,
                err_line_in_snippet_calc,
            ) = self._get_error_context_details(
                original_code_lines, actual_err_idx, current_error_msg_for_snippet_logic
            )

            if not snippet_text_for_ai.strip():
                correction_log.append(
                    f"  Kontext-Snippet für KI ist leer (Fehler in Zeile {actual_err_idx + 1}). Überspringe KI-Versuch."
                )
                return current_code, err_details_current

            prompt_error_info = {
                "msg": err_details_current.get("msg", "Unbekannter Fehler"),
                "error_line_in_snippet": err_line_in_snippet_calc,
                "original_lineno": err_details_current.get("lineno", 0),
                "text": err_details_current.get("text", ""),
            }
            ai_suggestion_raw = self._get_ai_correction(
                snippet_text_for_ai,
                prompt_error_info,
                correction_log,
                file_context=file_context,
            )

            if (
                ai_suggestion_raw is None
                or not ai_suggestion_raw.strip()
                and ai_suggestion_raw != "\n"
            ):
                correction_log.append(
                    "  KI-Korrektur fehlgeschlagen oder kein/leerer Vorschlag erhalten (außer Newline)."
                )
                return current_code, err_details_current

            num_ai_suggestion_lines = len(ai_suggestion_raw.splitlines())
            num_snippet_lines = len(snippet_text_for_ai.splitlines())
            is_complex_error = (
                "indent" in current_error_msg_for_snippet_logic.lower()
                or "expected an indented block"
                in current_error_msg_for_snippet_logic.lower()
            )

            if num_ai_suggestion_lines < num_snippet_lines and is_complex_error:
                correction_log.append(
                    f"  WARNUNG: KI-Vorschlag ({num_ai_suggestion_lines} Zeilen) ist kürzer als der Snippet ({num_snippet_lines} Zeilen) "
                    f"für einen komplexen Fehler ('{current_error_msg_for_snippet_logic}'). "
                    "Vorschlag wird nicht angewendet, um Codeverlust zu vermeiden."
                )
                return current_code, err_details_current

            target_indent_for_suggestion_block = ""
            current_error_msg_str = str(err_details_current.get("msg", ""))

            if num_ai_suggestion_lines == num_snippet_lines and num_snippet_lines > 0:
                first_line_of_original_snippet = snippet_text_for_ai.splitlines(True)[0]
                indent_of_first_snippet_line = self._get_leading_whitespace(
                    first_line_of_original_snippet
                )
                correction_log.append(
                    f"  KI hat den gesamten Snippet zurückgegeben. Setze Ziel-Basiseinrückung auf: "
                    f"'{repr(indent_of_first_snippet_line)}' (basierend auf erster Zeile des Original-Snippets)."
                )
                target_indent_for_suggestion_block = indent_of_first_snippet_line
            elif (
                "expected an indented block" in current_error_msg_str
                and actual_err_idx > 0
                and actual_err_idx <= len(original_code_lines)
            ):
                prev_line_indent = self._get_leading_whitespace(
                    original_code_lines[actual_err_idx - 1]
                )
                target_indent_for_suggestion_block = prev_line_indent + "    "
                correction_log.append(
                    f"  Error type 'expected an indented block'. Target indent: '{repr(target_indent_for_suggestion_block)}' (based on prev line + 4 spaces)."
                )
            elif actual_err_idx < len(original_code_lines):
                target_indent_for_suggestion_block = self._get_leading_whitespace(
                    original_code_lines[actual_err_idx]
                )
                correction_log.append(
                    f"  Using current indent of error line {actual_err_idx + 1} as target base indent: '{repr(target_indent_for_suggestion_block)}'."
                )
            else:
                target_indent_for_suggestion_block = ""
                correction_log.append(
                    f"  Fallback: Target base indent set to empty string for error line {actual_err_idx + 1}."
                )

            processed_block, replacement_mode, _ = self._process_ai_suggestion(
                ai_suggestion_raw,
                original_code_lines,
                snippet_start_idx,
                target_indent_for_suggestion_block,
                err_details_current.get("msg", "Unbekannter Fehler"),
                correction_log,
            )

            is_likely_single_line_fix = ai_suggestion_raw.count("\n") == 0
            if is_likely_single_line_fix and not (
                "expected an indented block" in current_error_msg_str
            ):
                replacement_mode_for_build = "single_line_focused"
                correction_log.append(
                    f"  KI-Vorschlag ist einzeilig und Fehler ist nicht 'expected indented block'. Build-Modus: {replacement_mode_for_build}"
                )
            else:
                if (
                    num_ai_suggestion_lines == num_snippet_lines
                    and num_snippet_lines > 0
                ):
                    replacement_mode_for_build = "standard"
                    correction_log.append(
                        f"  KI hat ganzen Snippet zurückgegeben. Build-Modus: {replacement_mode_for_build}"
                    )
                else:
                    replacement_mode_for_build = replacement_mode

            prospective_code = self._build_prospective_code(
                original_code_lines,
                processed_block,
                replacement_mode_for_build,
                actual_err_idx,
                snippet_start_idx,
                snippet_end_idx,
                correction_log,
            )
            is_valid, validation_error_details = self._validate_correction(
                prospective_code, correction_log
            )

            if is_valid:
                correction_log.append(
                    f"  KI-Fix für Fehler nahe Zeile {actual_err_idx + 1} erfolgreich angewendet."
                )
                return prospective_code, None
            else:
                correction_log.append(
                    "  KI-Vorschlag führte zu neuem Validierungsfehler. Der Code mit diesem lokalen Fix wird für den nächsten Versuch verwendet."
                )
                return prospective_code, validation_error_details

        except Exception as other_e:
            correction_log.append(
                f"Unerwarteter Fehler im Korrekturzyklus für Versuch {attempt_num}: {other_e}"
            )
            logger.error(
                f"Unerwarteter Fehler in _single_correction_attempt: {other_e}",
                exc_info=True,
            )
            unexpected_err_details: ErrorDetails = {
                "msg": f"Interner Fehler: {str(other_e)}",
                "lineno": None,
                "offset": None,
                "text": None,
                "actual_line_idx": -1,
            }
            return current_code, unexpected_err_details

    def analyze_and_correct(
        self,
        code: str,
        max_attempts: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> CorrectionResult:
        """
        Main entry point for code correction.
        - Tries up to max_attempts corrections using LLM and validates output.
        - Applies autopep8 formatting to each attempt.
        - Returns CorrectionResult with logs.
        """
        max_attempts_to_use = (
            max_attempts if max_attempts is not None else self.max_correction_attempts
        )
        current_code_iter = code
        original_code_for_comparison = code
        correction_log: List[str] = [
            "--- Starte iterative Code-Analyse & KI-Korrektur ---"
        ]

        if context:
            imports_str = context.get("imports_context", "Imports: []")
            defs_str = context.get("definitions_context", "Definitions: []")
            file_path_str = context.get("file_path", "aktuelle Datei")
            log_context_msg_parts = [f"Kontext für {file_path_str}:"]
            if imports_str and imports_str != "Imports: []":
                log_context_msg_parts.append(f"  {imports_str}")
            if defs_str and defs_str != "Definitions: []":
                log_context_msg_parts.append(f"  {defs_str}")
            if len(log_context_msg_parts) == 1:
                log_context_msg_parts.append(
                    "  (Keine spezifischen Importe oder Definitionen im Kontext gefunden)"
                )
            correction_log.append("\n".join(log_context_msg_parts))
            logger.info("\n".join(log_context_msg_parts))

        if not current_code_iter.strip():
            correction_log.append("Eingabecode ist leer.")
            return {
                "status": "empty_code",
                "code": current_code_iter,
                "log": correction_log,
                "error_details": None,
            }

        model_tokenizer_tuple = self.inference_service.get_loaded_model_and_tokenizer()
        if (
            model_tokenizer_tuple is None
            or model_tokenizer_tuple[0] is None
            or model_tokenizer_tuple[1] is None
        ):
            correction_log.append(
                "KI-Modell/Tokenizer nicht geladen. Fallback auf Formatierung."
            )
            return self._fallback_to_formatting(current_code_iter, correction_log)

        last_error_details: Optional[ErrorDetails] = {
            "msg": "Initial error check pending"
        }
        loop_attempt_count = 0

        for loop_attempt_count_val in range(1, max_attempts_to_use + 1):
            loop_attempt_count = loop_attempt_count_val
            code_after_attempt, error_details_after_attempt = (
                self._single_correction_attempt(
                    current_code_iter,
                    correction_log,
                    loop_attempt_count,
                    max_attempts_to_use,
                    file_context=context,
                )
            )
            current_code_iter = code_after_attempt
            last_error_details = error_details_after_attempt
            if last_error_details is None:
                break
            if loop_attempt_count == max_attempts_to_use:
                correction_log.append(
                    f"Maximal {max_attempts_to_use} Versuche erreicht. Fehler ggf. nicht behoben."
                )

        if last_error_details is None:
            formatted_code = autopep8.fix_code(
                current_code_iter, options={"aggressive": 1}
            )
            status_key: str
            if current_code_iter.strip() != original_code_for_comparison.strip():
                status_key = (
                    "corrected_and_formatted"
                    if formatted_code.strip() != current_code_iter.strip()
                    else "corrected"
                )
            elif formatted_code.strip() != original_code_for_comparison.strip():
                status_key = "formatted"
            else:
                status_key = "no_changes"
            logger.info(f"Korrekturprozess abgeschlossen. Status: {status_key}")
            return {
                "status": status_key,
                "code": formatted_code,
                "log": correction_log,
                "error_details": None,
            }
        else:
            logger.warning(
                f"Korrekturprozess beendet. Fehler nicht vollständig behoben: {last_error_details.get('msg')}"
            )
            status_if_error = "max_attempts_reached"
            return {
                "status": status_if_error,
                "code": current_code_iter,
                "log": correction_log,
                "error_details": last_error_details,
            }

    def _fallback_to_formatting(
        self, code: str, correction_log: List[str]
    ) -> CorrectionResult:
        try:
            code_to_parse = code.strip("\n") + "\n" if code.strip() else "\n"
            ast.parse(code_to_parse)
            formatted_code = autopep8.fix_code(code, options={"aggressive": 1})
            status = (
                "formatted" if formatted_code.strip() != code.strip() else "no_changes"
            )
            return {
                "status": status,
                "code": formatted_code,
                "log": correction_log,
                "error_details": None,
            }
        except SyntaxError as syn_e:
            err_details: ErrorDetails = self._extract_syntax_error_details(syn_e, code)
            return {
                "status": "syntax_error_unresolved",
                "code": code,
                "log": correction_log,
                "error_details": err_details,
            }
        except Exception as e:
            err_details_fallback: ErrorDetails = {
                "msg": str(e),
                "lineno": None,
                "offset": None,
                "text": None,
                "actual_line_idx": -1,
            }
            return {
                "status": "error",
                "code": code,
                "log": correction_log,
                "error_details": err_details_fallback,
            }
