"""Compact, low-ambiguity coding tool surface tailored for Inkling 2 Mini.

Every tool schema is canonical and immutable so Hikari's static system+tool prefix
stays strictly byte-identical across turns and threads. Tool outputs are capped with
head/tail preservation before entering durable prompt history, and malformed arguments
or failed edits return structured self-correction diagnostics instead of aborting.
"""

from __future__ import annotations

import asyncio
import difflib
import fnmatch
import json
import os
import re
import shutil
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..runner.agent_inference import unique_fields

MAX_TOOL_RESULT_CHARS = 12_000
MAX_READ_LINES = 300
MAX_FIND_RESULTS = 100
MAX_SEARCH_MATCHES = 50
DEFAULT_COMMAND_TIMEOUT_SECONDS = 60
MAX_COMMAND_TIMEOUT_SECONDS = 300

IGNORED_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".next",
        ".turbo",
    }
)


class _StrictToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ReadFileArgs(_StrictToolArgs):
    path: str = Field(
        min_length=1,
        max_length=1024,
        description="Relative path to the file inside the workspace.",
    )
    start_line: int = Field(
        default=1,
        ge=1,
        description="1-based first line number to read (default 1).",
    )
    end_line: int | None = Field(
        default=None,
        ge=1,
        description="1-based last line number (inclusive). Defaults to start_line + 199.",
    )


class FindFilesArgs(_StrictToolArgs):
    pattern: str = Field(
        default="**/*",
        min_length=1,
        max_length=256,
        description="Glob pattern relative to path (for example '**/*.py' or '*.tsx').",
    )
    path: str = Field(
        default=".",
        min_length=1,
        max_length=1024,
        description="Subdirectory to search within (default '.').",
    )
    max_results: int = Field(
        default=80,
        ge=1,
        le=MAX_FIND_RESULTS,
        description="Maximum number of file paths to return.",
    )


class SearchTextArgs(_StrictToolArgs):
    query: str = Field(
        min_length=1,
        max_length=512,
        description="Text or regex pattern to search for.",
    )
    path: str = Field(
        default=".",
        min_length=1,
        max_length=1024,
        description="Subdirectory or file to search within (default '.').",
    )
    glob: str | None = Field(
        default=None,
        max_length=128,
        description="Optional filename glob filter (for example '*.py').",
    )
    is_regex: bool = Field(
        default=False,
        description="Treat query as a regular expression when true.",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Match case strictly when true.",
    )
    max_matches: int = Field(
        default=40,
        ge=1,
        le=MAX_SEARCH_MATCHES,
        description="Maximum number of matching lines to return.",
    )


class EditFileArgs(_StrictToolArgs):
    path: str = Field(
        min_length=1,
        max_length=1024,
        description="Relative path to an existing file to edit.",
    )
    old_text: str = Field(
        min_length=1,
        max_length=100_000,
        description="Exact existing text to replace (must match uniquely unless replace_all is true).",
    )
    new_text: str = Field(
        max_length=100_000,
        description="Replacement text (empty string deletes old_text).",
    )
    replace_all: bool = Field(
        default=False,
        description="Replace every occurrence of old_text when true.",
    )


class WriteFileArgs(_StrictToolArgs):
    path: str = Field(
        min_length=1,
        max_length=1024,
        description="Relative path to create or overwrite inside the workspace.",
    )
    content: str = Field(
        max_length=200_000,
        description="Complete file content to write.",
    )


class RunCommandArgs(_StrictToolArgs):
    command: str = Field(
        min_length=1,
        max_length=4000,
        description="Shell command to run inside the workspace.",
    )
    cwd: str | None = Field(
        default=None,
        max_length=1024,
        description="Optional subdirectory relative to workspace root.",
    )
    timeout_seconds: int = Field(
        default=DEFAULT_COMMAND_TIMEOUT_SECONDS,
        ge=1,
        le=MAX_COMMAND_TIMEOUT_SECONDS,
        description="Maximum execution time in seconds (default 60, max 300).",
    )


class GitInspectArgs(_StrictToolArgs):
    mode: Literal["status", "diff", "log"] = Field(
        default="status",
        description="Git view to inspect: 'status', 'diff', or 'log'.",
    )
    path: str | None = Field(
        default=None,
        max_length=1024,
        description="Optional relative file or directory path filter.",
    )
    staged: bool = Field(
        default=False,
        description="When mode is 'diff', inspect staged changes (--cached).",
    )
    max_commits: int = Field(
        default=10,
        ge=1,
        le=30,
        description="When mode is 'log', maximum number of commits to show.",
    )


class AskUserArgs(_StrictToolArgs):
    question: str = Field(
        min_length=1,
        max_length=4000,
        description="Clear, plain-language question for the user.",
    )
    title: str = Field(
        default="Question",
        min_length=1,
        max_length=120,
        description="Short heading shown on the question card.",
    )
    options: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Optional short choices the user can click.",
    )


TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {
    "read_file": ReadFileArgs,
    "find_files": FindFilesArgs,
    "search_text": SearchTextArgs,
    "edit_file": EditFileArgs,
    "write_file": WriteFileArgs,
    "run_command": RunCommandArgs,
    "git_inspect": GitInspectArgs,
    "ask_user": AskUserArgs,
}

_TOOL_DESCRIPTIONS: dict[str, str] = {
    "read_file": (
        "Read lines from a workspace file with 1-based line numbers. "
        "Use explicit line ranges for large files."
    ),
    "find_files": (
        "List files in the workspace matching a glob pattern, skipping .git and build folders."
    ),
    "search_text": (
        "Search text or regex patterns across workspace files, returning path:line matches."
    ),
    "edit_file": (
        "Replace an exact snippet in an existing file. Returns nearby lines on mismatch so you can self-correct."
    ),
    "write_file": (
        "Create a new file or overwrite an existing file with complete contents."
    ),
    "run_command": (
        "Execute a shell command scoped to the workspace with bounded output capture and timeout."
    ),
    "git_inspect": (
        "Inspect repository status, diff, or recent commit log without running arbitrary shell commands."
    ),
    "ask_user": (
        "Pause and ask the user a clarifying question when blocked or when a product decision is needed."
    ),
}


def _compact_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Build a minimal, deterministic JSON schema without redundant titles."""
    raw = model.model_json_schema()
    properties: dict[str, Any] = {}
    for prop_name in sorted(raw.get("properties", {})):
        prop = dict(raw["properties"][prop_name])
        prop.pop("title", None)
        properties[prop_name] = prop
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    required = raw.get("required")
    if required:
        schema["required"] = sorted(required)
    return schema


def _build_static_tool_schemas() -> tuple[dict[str, Any], ...]:
    schemas: list[dict[str, Any]] = []
    for name in (
        "read_file",
        "find_files",
        "search_text",
        "edit_file",
        "write_file",
        "run_command",
        "git_inspect",
        "ask_user",
    ):
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": _TOOL_DESCRIPTIONS[name],
                    "parameters": _compact_schema(TOOL_ARG_MODELS[name]),
                },
            }
        )
    return tuple(schemas)


INKLING_TOOL_SCHEMAS: tuple[dict[str, Any], ...] = _build_static_tool_schemas()
INKLING_TOOL_NAMES: tuple[str, ...] = tuple(
    item["function"]["name"] for item in INKLING_TOOL_SCHEMAS
)


def canonical_tool_schemas_json() -> str:
    """Stable canonical JSON of the static tool schemas for prefix digest verification."""
    return json.dumps(
        list(INKLING_TOOL_SCHEMAS),
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    )


def cap_tool_output(
    text: str, *, max_chars: int = MAX_TOOL_RESULT_CHARS
) -> tuple[str, bool]:
    """Bound tool output using head+tail preservation before entering prompt history."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    total = len(text)
    notice_budget = 140
    usable = max(64, max_chars - notice_budget)
    head_len = int(usable * 0.6)
    tail_len = max(16, usable - head_len)
    omitted = max(0, total - head_len - tail_len)
    notice = (
        f"\n... [output truncated: omitted {omitted} of {total} chars; "
        "narrow your command, path, or line range] ...\n"
    )
    bounded = text[:head_len].rstrip("\n") + notice + text[-tail_len:].lstrip("\n")
    return bounded, True


def validate_tool_arguments(
    tool_name: str, raw_arguments: Any
) -> tuple[BaseModel | None, dict[str, Any] | None]:
    """Validate model tool arguments and return guided recovery diagnostics on error."""
    model_cls = TOOL_ARG_MODELS.get(tool_name)
    if model_cls is None:
        return None, {
            "ok": False,
            "error_type": "unknown_tool",
            "tool": tool_name,
            "valid_tools": list(INKLING_TOOL_NAMES),
            "hint": "Call one of the declared tools in valid_tools.",
        }
    parsed: Any = raw_arguments
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed, object_pairs_hook=unique_fields)
        except ValueError as exc:
            return None, {
                "ok": False,
                "error_type": "invalid_json",
                "tool": tool_name,
                "problems": [
                    {"field": "arguments", "code": "json_invalid", "message": str(exc)}
                ],
                "hint": "Arguments must be a valid JSON object matching the tool schema.",
            }
    if not isinstance(parsed, dict):
        return None, {
            "ok": False,
            "error_type": "invalid_arguments",
            "tool": tool_name,
            "problems": [
                {
                    "field": "arguments",
                    "code": "dict_type",
                    "message": "Tool arguments must be a JSON object.",
                }
            ],
            "hint": "Pass a JSON object with the required fields for this tool.",
        }
    try:
        return model_cls.model_validate(parsed), None
    except ValidationError as exc:
        problems = [
            {
                "field": ".".join(str(p) for p in err["loc"][:4]) or "arguments",
                "code": err["type"],
                "message": err["msg"],
            }
            for err in exc.errors(
                include_url=False, include_input=False, include_context=False
            )[:6]
        ]
        return None, {
            "ok": False,
            "error_type": "invalid_arguments",
            "tool": tool_name,
            "problems": problems,
            "hint": (
                "No tool was executed. Correct the arguments using the bound tool schema; "
                "do not repeat the unchanged call."
            ),
        }


def resolve_workspace_path(working_dir: Path, path_str: str) -> Path:
    """Resolve `path_str` inside `working_dir`, preventing symlink or '..' escape."""
    root = working_dir.resolve()
    raw = Path(path_str)
    candidate = (raw if raw.is_absolute() else root / raw).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Path '{path_str}' is outside the project workspace.")
    return candidate


def _relative_display(root: Path, target: Path) -> str:
    try:
        return str(target.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(target)


def find_nearby_lines(
    content: str, needle: str, *, max_candidates: int = 2, context_lines: int = 2
) -> list[dict[str, Any]]:
    """Find the closest matching line windows in `content` when `edit_file` misses."""
    file_lines = content.splitlines()
    if not file_lines:
        return []
    needle_lines = needle.splitlines() or [needle]
    window = max(1, min(len(needle_lines), len(file_lines)))
    needle_norm = "\n".join(line.strip() for line in needle_lines)
    scored: list[tuple[float, int, int]] = []
    # Scan bounded windows (cap iterations on huge files).
    step = 1 if len(file_lines) <= 2000 else max(1, len(file_lines) // 2000)
    for idx in range(0, len(file_lines) - window + 1, step):
        chunk = "\n".join(line.strip() for line in file_lines[idx : idx + window])
        ratio = difflib.SequenceMatcher(None, needle_norm, chunk).ratio()
        if ratio >= 0.35:
            scored.append((ratio, idx, idx + window))
    scored.sort(key=lambda item: item[0], reverse=True)

    results: list[dict[str, Any]] = []
    used_ranges: list[tuple[int, int]] = []
    for ratio, start_idx, end_idx in scored:
        if any(not (end_idx <= s or start_idx >= e) for s, e in used_ranges):
            continue
        used_ranges.append((start_idx, end_idx))
        ctx_start = max(0, start_idx - context_lines)
        ctx_end = min(len(file_lines), end_idx + context_lines)
        numbered = "\n".join(
            f"{ln}: {file_lines[ln - 1]}" for ln in range(ctx_start + 1, ctx_end + 1)
        )
        results.append(
            {
                "start_line": ctx_start + 1,
                "end_line": ctx_end,
                "similarity": round(ratio, 3),
                "snippet": numbered,
            }
        )
        if len(results) >= max_candidates:
            break
    return results


@dataclass
class ToolExecutionResult:
    ok: bool
    output: str
    is_error: bool = False
    truncated: bool = False
    file_change: tuple[str, Literal["created", "modified"]] | None = None
    exit_code: int | None = None
    data: dict[str, Any] = field(default_factory=dict)


class InklingToolset:
    """Executes Inkling coding tools inside a single bound workspace directory."""

    def __init__(
        self,
        working_dir: str | Path,
        *,
        max_output_chars: int = MAX_TOOL_RESULT_CHARS,
    ):
        self.working_dir = Path(working_dir).resolve()
        self.max_output_chars = max_output_chars
        self._active_procs: set[asyncio.subprocess.Process] = set()

    async def cancel_running(self) -> None:
        """Terminate any active subprocess groups started by this toolset."""
        for proc in list(self._active_procs):
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except TimeoutError:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        proc.kill()

    async def execute(self, tool_name: str, args: BaseModel) -> ToolExecutionResult:
        if tool_name == "read_file":
            assert isinstance(args, ReadFileArgs)
            return await asyncio.to_thread(self._read_file, args)
        if tool_name == "find_files":
            assert isinstance(args, FindFilesArgs)
            return await asyncio.to_thread(self._find_files, args)
        if tool_name == "search_text":
            assert isinstance(args, SearchTextArgs)
            return await self._search_text(args)
        if tool_name == "edit_file":
            assert isinstance(args, EditFileArgs)
            return await asyncio.to_thread(self._edit_file, args)
        if tool_name == "write_file":
            assert isinstance(args, WriteFileArgs)
            return await asyncio.to_thread(self._write_file, args)
        if tool_name == "run_command":
            assert isinstance(args, RunCommandArgs)
            return await self._run_command(args)
        if tool_name == "git_inspect":
            assert isinstance(args, GitInspectArgs)
            return await self._git_inspect(args)
        return ToolExecutionResult(
            ok=False,
            is_error=True,
            output=json.dumps(
                {"ok": False, "error": f"Unsupported tool '{tool_name}'."}
            ),
        )

    def _read_file(self, args: ReadFileArgs) -> ToolExecutionResult:
        try:
            target = resolve_workspace_path(self.working_dir, args.path)
        except ValueError as exc:
            return ToolExecutionResult(ok=False, is_error=True, output=str(exc))
        if not target.exists():
            return ToolExecutionResult(
                ok=False,
                is_error=True,
                output=f"File not found: {args.path}",
            )
        if target.is_dir():
            return ToolExecutionResult(
                ok=False,
                is_error=True,
                output=f"'{args.path}' is a directory. Use find_files to list its contents.",
            )
        try:
            raw = target.read_bytes()
        except OSError as exc:
            return ToolExecutionResult(
                ok=False, is_error=True, output=f"Could not read '{args.path}': {exc}"
            )
        if b"\x00" in raw[:4096]:
            return ToolExecutionResult(
                ok=False,
                is_error=True,
                output=f"'{args.path}' appears to be a binary file ({len(raw)} bytes).",
            )
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        total_lines = len(lines)
        start = args.start_line
        requested_end = args.end_line if args.end_line is not None else start + 199
        if requested_end < start:
            return ToolExecutionResult(
                ok=False,
                is_error=True,
                output=f"end_line ({requested_end}) must be >= start_line ({start}).",
            )
        clamped_end = min(total_lines, start + MAX_READ_LINES - 1, requested_end)
        rel = _relative_display(self.working_dir, target)
        if total_lines == 0:
            return ToolExecutionResult(ok=True, output=f"{rel} (empty file)")
        if start > total_lines:
            return ToolExecutionResult(
                ok=False,
                is_error=True,
                output=f"{rel} only has {total_lines} lines (requested start_line={start}).",
            )
        body_lines = [f"{ln}: {lines[ln - 1]}" for ln in range(start, clamped_end + 1)]
        header = f"{rel} (lines {start}-{clamped_end} of {total_lines})"
        footer = ""
        if clamped_end < total_lines:
            footer = f"\n... [{total_lines - clamped_end} more lines; call read_file with start_line={clamped_end + 1}]"
        output, truncated = cap_tool_output(
            header + "\n" + "\n".join(body_lines) + footer,
            max_chars=self.max_output_chars,
        )
        return ToolExecutionResult(
            ok=True,
            output=output,
            truncated=truncated or clamped_end < total_lines,
        )

    def _find_files(self, args: FindFilesArgs) -> ToolExecutionResult:
        try:
            base = resolve_workspace_path(self.working_dir, args.path)
        except ValueError as exc:
            return ToolExecutionResult(ok=False, is_error=True, output=str(exc))
        if not base.exists() or not base.is_dir():
            return ToolExecutionResult(
                ok=False, is_error=True, output=f"Directory not found: {args.path}"
            )
        matches: list[str] = []
        truncated = False
        pattern = args.pattern.strip()
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
            current = Path(dirpath)
            for fname in sorted(filenames):
                full = current / fname
                rel_to_base = full.relative_to(base).as_posix()
                if fnmatch.fnmatch(rel_to_base, pattern) or (
                    pattern.startswith("**/") and fnmatch.fnmatch(fname, pattern[3:])
                ):
                    matches.append(_relative_display(self.working_dir, full))
                    if len(matches) >= args.max_results:
                        truncated = True
                        break
            if truncated:
                break
        if not matches:
            return ToolExecutionResult(
                ok=True,
                output=f"No files matched '{args.pattern}' under '{args.path}'.",
            )
        lines = matches
        if truncated:
            lines.append(
                f"... [capped at {args.max_results} files; refine pattern or path]"
            )
        output, capped = cap_tool_output(
            "\n".join(lines), max_chars=self.max_output_chars
        )
        return ToolExecutionResult(
            ok=True, output=output, truncated=truncated or capped
        )

    async def _search_text(self, args: SearchTextArgs) -> ToolExecutionResult:
        try:
            base = resolve_workspace_path(self.working_dir, args.path)
        except ValueError as exc:
            return ToolExecutionResult(ok=False, is_error=True, output=str(exc))
        if not base.exists():
            return ToolExecutionResult(
                ok=False, is_error=True, output=f"Path not found: {args.path}"
            )
        if args.is_regex:
            try:
                re.compile(args.query)
            except re.error as exc:
                return ToolExecutionResult(
                    ok=False,
                    is_error=True,
                    output=f"Invalid regular expression '{args.query}': {exc}",
                )
        rg_bin = shutil.which("rg")
        if rg_bin:
            return await self._search_text_rg(rg_bin, base, args)
        return await asyncio.to_thread(self._search_text_python, base, args)

    async def _search_text_rg(
        self, rg_bin: str, base: Path, args: SearchTextArgs
    ) -> ToolExecutionResult:
        cmd = [
            rg_bin,
            "--line-number",
            "--no-heading",
            "--color=never",
            "--max-count",
            str(args.max_matches),
        ]
        if not args.is_regex:
            cmd.append("--fixed-strings")
        if not args.case_sensitive:
            cmd.append("--ignore-case")
        if args.glob:
            cmd.extend(["--glob", args.glob])
        for ignored in sorted(IGNORED_DIRS):
            cmd.extend(["--glob", f"!{ignored}/**"])
        cmd.extend(["--", args.query, str(base)])
        stdout, stderr, code = await self._run_subprocess(
            cmd, cwd=self.working_dir, timeout_seconds=20, shell=False
        )
        if code == 1 and not stdout.strip():
            return ToolExecutionResult(
                ok=True, output=f"No matches for '{args.query}'."
            )
        if code not in (0, 1):
            return ToolExecutionResult(
                ok=False,
                is_error=True,
                output=stderr.strip() or f"Search failed with exit code {code}.",
            )
        raw_lines = stdout.splitlines()
        rel_root = str(self.working_dir) + os.sep
        cleaned = [
            line.removeprefix(rel_root) for line in raw_lines[: args.max_matches]
        ]
        truncated = len(raw_lines) > args.max_matches
        if truncated:
            cleaned.append(
                f"... [capped at {args.max_matches} matches; narrow path or query]"
            )
        output, capped = cap_tool_output(
            "\n".join(cleaned), max_chars=self.max_output_chars
        )
        return ToolExecutionResult(
            ok=True, output=output, truncated=truncated or capped
        )

    def _search_text_python(
        self, base: Path, args: SearchTextArgs
    ) -> ToolExecutionResult:
        flags = 0 if args.case_sensitive else re.IGNORECASE
        pattern = re.compile(
            args.query if args.is_regex else re.escape(args.query), flags
        )
        files: list[Path] = []
        if base.is_file():
            files = [base]
        else:
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
                for fname in sorted(filenames):
                    if args.glob and not fnmatch.fnmatch(fname, args.glob):
                        continue
                    files.append(Path(dirpath) / fname)
        hits: list[str] = []
        truncated = False
        for path in files:
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:2048]:
                continue
            rel = _relative_display(self.working_dir, path)
            for idx, line in enumerate(
                raw.decode("utf-8", errors="replace").splitlines(), start=1
            ):
                if pattern.search(line):
                    hits.append(f"{rel}:{idx}:{line[:240]}")
                    if len(hits) >= args.max_matches:
                        truncated = True
                        break
            if truncated:
                break
        if not hits:
            return ToolExecutionResult(
                ok=True, output=f"No matches for '{args.query}'."
            )
        if truncated:
            hits.append(
                f"... [capped at {args.max_matches} matches; narrow path or query]"
            )
        output, capped = cap_tool_output(
            "\n".join(hits), max_chars=self.max_output_chars
        )
        return ToolExecutionResult(
            ok=True, output=output, truncated=truncated or capped
        )

    def _edit_file(self, args: EditFileArgs) -> ToolExecutionResult:
        try:
            target = resolve_workspace_path(self.working_dir, args.path)
        except ValueError as exc:
            return ToolExecutionResult(ok=False, is_error=True, output=str(exc))
        if not target.is_file():
            return ToolExecutionResult(
                ok=False,
                is_error=True,
                output=f"File not found: {args.path}. Use write_file to create a new file.",
            )
        content = target.read_text(encoding="utf-8", errors="replace")
        rel = _relative_display(self.working_dir, target)
        count = content.count(args.old_text)
        if count == 0:
            nearby = find_nearby_lines(content, args.old_text)
            diagnostic = {
                "ok": False,
                "error_type": "edit_mismatch",
                "path": rel,
                "message": "old_text was not found verbatim in the file.",
                "nearby_lines": nearby,
                "hint": (
                    "Inspect nearby_lines (or call read_file) and retry edit_file "
                    "with an exact, verbatim old_text match including indentation."
                ),
            }
            output, capped = cap_tool_output(
                json.dumps(diagnostic, indent=2, ensure_ascii=False),
                max_chars=self.max_output_chars,
            )
            return ToolExecutionResult(
                ok=False, is_error=True, output=output, truncated=capped
            )
        if count > 1 and not args.replace_all:
            # Find 1-based line numbers of all matches for disambiguation.
            match_lines: list[int] = []
            offset = 0
            while True:
                pos = content.find(args.old_text, offset)
                if pos == -1:
                    break
                match_lines.append(content.count("\n", 0, pos) + 1)
                offset = pos + len(args.old_text)
            diagnostic = {
                "ok": False,
                "error_type": "ambiguous_edit",
                "path": rel,
                "match_count": count,
                "match_lines": match_lines[:20],
                "hint": (
                    "Include more surrounding context in old_text so it matches once, "
                    "or set replace_all=true to update every occurrence."
                ),
            }
            return ToolExecutionResult(
                ok=False,
                is_error=True,
                output=json.dumps(diagnostic, indent=2, ensure_ascii=False),
            )

        updated = (
            content.replace(args.old_text, args.new_text)
            if args.replace_all
            else content.replace(args.old_text, args.new_text, 1)
        )
        target.write_text(updated, encoding="utf-8")
        replaced = count if args.replace_all else 1
        return ToolExecutionResult(
            ok=True,
            output=f"Edited {rel} ({replaced} replacement{'s' if replaced != 1 else ''}).",
            file_change=(rel, "modified"),
        )

    def _write_file(self, args: WriteFileArgs) -> ToolExecutionResult:
        try:
            target = resolve_workspace_path(self.working_dir, args.path)
        except ValueError as exc:
            return ToolExecutionResult(ok=False, is_error=True, output=str(exc))
        existed = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args.content, encoding="utf-8")
        rel = _relative_display(self.working_dir, target)
        change_type: Literal["created", "modified"] = (
            "modified" if existed else "created"
        )
        line_count = len(args.content.splitlines())
        return ToolExecutionResult(
            ok=True,
            output=f"{'Updated' if existed else 'Created'} {rel} ({line_count} lines, {len(args.content)} chars).",
            file_change=(rel, change_type),
        )

    async def _run_command(self, args: RunCommandArgs) -> ToolExecutionResult:
        try:
            cwd = (
                resolve_workspace_path(self.working_dir, args.cwd)
                if args.cwd
                else self.working_dir
            )
        except ValueError as exc:
            return ToolExecutionResult(ok=False, is_error=True, output=str(exc))
        if not cwd.is_dir():
            return ToolExecutionResult(
                ok=False,
                is_error=True,
                output=f"Working directory does not exist: {args.cwd}",
            )
        try:
            stdout, stderr, code = await self._run_subprocess(
                args.command,
                cwd=cwd,
                timeout_seconds=args.timeout_seconds,
                shell=True,
            )
        except TimeoutError:
            return ToolExecutionResult(
                ok=False,
                is_error=True,
                exit_code=-1,
                output=(
                    f"Command timed out after {args.timeout_seconds}s: {args.command}\n"
                    "Try a narrower command or a longer timeout_seconds."
                ),
            )
        parts: list[str] = [f"exit_code: {code}"]
        if stdout.strip():
            parts.append("stdout:\n" + stdout.rstrip("\n"))
        if stderr.strip():
            parts.append("stderr:\n" + stderr.rstrip("\n"))
        if len(parts) == 1:
            parts.append("(no output)")
        output, truncated = cap_tool_output(
            "\n".join(parts), max_chars=self.max_output_chars
        )
        return ToolExecutionResult(
            ok=code == 0,
            is_error=code != 0,
            output=output,
            truncated=truncated,
            exit_code=code,
            data={"command": args.command, "exit_code": code},
        )

    async def _git_inspect(self, args: GitInspectArgs) -> ToolExecutionResult:
        rel_path: str | None = None
        if args.path:
            try:
                resolved = resolve_workspace_path(self.working_dir, args.path)
            except ValueError as exc:
                return ToolExecutionResult(ok=False, is_error=True, output=str(exc))
            rel_path = _relative_display(self.working_dir, resolved)
        if args.mode == "status":
            cmd = ["git", "status", "--short", "--branch"]
            if rel_path:
                cmd.extend(["--", rel_path])
        elif args.mode == "diff":
            cmd = ["git", "diff"]
            if args.staged:
                cmd.append("--cached")
            if rel_path:
                cmd.extend(["--", rel_path])
        else:
            cmd = [
                "git",
                "log",
                f"-n{args.max_commits}",
                "--oneline",
                "--decorate",
            ]
            if rel_path:
                cmd.extend(["--", rel_path])
        stdout, stderr, code = await self._run_subprocess(
            cmd, cwd=self.working_dir, timeout_seconds=15, shell=False
        )
        if code != 0:
            return ToolExecutionResult(
                ok=False,
                is_error=True,
                exit_code=code,
                output=stderr.strip() or f"git {args.mode} failed (exit {code}).",
            )
        text = stdout.strip() or f"(clean: no output for git {args.mode})"
        output, truncated = cap_tool_output(text, max_chars=self.max_output_chars)
        return ToolExecutionResult(
            ok=True, output=output, truncated=truncated, exit_code=0
        )

    async def _run_subprocess(
        self,
        cmd: str | list[str],
        *,
        cwd: Path,
        timeout_seconds: int,
        shell: bool,
    ) -> tuple[str, str, int]:
        if shell:
            assert isinstance(cmd, str)
            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        else:
            assert isinstance(cmd, list)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        self._active_procs.add(proc)
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds
            )
            return (
                stdout_b.decode("utf-8", errors="replace"),
                stderr_b.decode("utf-8", errors="replace"),
                proc.returncode if proc.returncode is not None else -1,
            )
        except (TimeoutError, asyncio.CancelledError):
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except TimeoutError:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        proc.kill()
            raise
        finally:
            self._active_procs.discard(proc)
