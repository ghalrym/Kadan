"""File tools: read, write, search and watch files on the computer."""

from typing import Literal

from pydantic import Field

from ...api.sandbox_client import SandboxClient, segment
from .base import Arguments, NoArguments, body, computer_tool


class ReadArguments(Arguments):
    file: str = Field(description="Absolute file path.")
    start_line: int | None = Field(default=None, description="First line, 0-based.")
    end_line: int | None = Field(default=None, description="End line, exclusive.")
    sudo: bool | None = Field(default=False, description="Use sudo privileges.")


@computer_tool("POST")
async def file_read_file(args: ReadArguments, computer: SandboxClient) -> dict:
    """Read a file's content."""
    return await computer.run("POST", "/v1/file/read", json_body=body(args))


class WriteArguments(Arguments):
    file: str = Field(description="Absolute file path.")
    content: str = Field(description="Text, or base64 for binary content.")
    encoding: Literal["utf-8", "base64", "raw"] | None = Field(
        default="utf-8", description="utf-8 for text, base64 for binary."
    )
    append: bool | None = Field(default=False, description="Append to the file.")
    leading_newline: bool | None = Field(
        default=False, description="Add a leading newline (text only)."
    )
    trailing_newline: bool | None = Field(
        default=False, description="Add a trailing newline (text only)."
    )
    sudo: bool | None = Field(default=False, description="Use sudo privileges.")


@computer_tool("POST")
async def file_write_file(args: WriteArguments, computer: SandboxClient) -> dict:
    """Write a text or binary file."""
    return await computer.run("POST", "/v1/file/write", json_body=body(args))


class ReplaceArguments(Arguments):
    file: str = Field(description="Absolute file path.")
    old_str: str = Field(description="Text to replace.")
    new_str: str = Field(description="Replacement text.")
    sudo: bool | None = Field(default=False, description="Use sudo privileges.")


@computer_tool("POST")
async def file_replace_in_file(args: ReplaceArguments, computer: SandboxClient) -> dict:
    """Replace a string in a file."""
    return await computer.run("POST", "/v1/file/replace", json_body=body(args))


class SearchArguments(Arguments):
    file: str = Field(description="Absolute file path.")
    regex: str = Field(description="Regular expression to search for.")
    sudo: bool | None = Field(default=False, description="Use sudo privileges.")


@computer_tool("POST")
async def file_search_in_file(args: SearchArguments, computer: SandboxClient) -> dict:
    """Search a file's content with a regular expression."""
    return await computer.run("POST", "/v1/file/search", json_body=body(args))


class EditorArguments(Arguments):
    command: Literal["view", "create", "str_replace", "insert", "undo_edit"] = Field(
        description="Editor command."
    )
    path: str = Field(description="Absolute path to a file or directory.")
    file_text: str | None = Field(
        default=None, description="create: content of the new file."
    )
    old_str: str | None = Field(
        default=None, description="str_replace: text in the file to replace."
    )
    new_str: str | None = Field(
        default=None,
        description="str_replace: replacement text. insert: text to insert.",
    )
    insert_line: int | None = Field(
        default=None, description="insert: new_str goes after this line."
    )
    view_range: list[int] | None = Field(
        default=None, description="view: [start, end] lines of a file."
    )
    replace_mode: Literal["ALL", "FIRST", "LAST"] | None = Field(
        default=None, description="str_replace: which occurrences to replace."
    )
    page_range: list[int] | None = Field(
        default=None, description="view: [start, end] pages of a PDF, 1-indexed."
    )
    sheet_name: str | None = Field(
        default=None, description="view: sheet of an Excel file."
    )
    row_range: list[int] | None = Field(
        default=None, description="view: [start, end] rows of an Excel sheet."
    )
    slide_range: list[int] | None = Field(
        default=None, description="view: [start, end] slides of a PPTX file."
    )
    enable_metadata: bool | None = Field(
        default=False, description="view: include file metadata."
    )


@computer_tool("POST")
async def file_str_replace_editor(
    args: EditorArguments, computer: SandboxClient
) -> dict:
    """View, create and edit files: view, create, str_replace, insert, undo_edit."""
    return await computer.run(
        "POST", "/v1/file/str_replace_editor", json_body=body(args)
    )


class ListArguments(Arguments):
    path: str = Field(description="Directory to list.")
    recursive: bool | None = Field(default=False, description="List recursively.")
    show_hidden: bool | None = Field(default=True, description="Show hidden files.")
    file_types: list[str] | None = Field(
        default=None, description='Extensions to include, e.g. [".py", ".txt"].'
    )
    max_depth: int | None = Field(default=None, description="Maximum depth.")
    include_size: bool | None = Field(default=True, description="Include sizes.")
    include_permissions: bool | None = Field(
        default=False, description="Include permissions."
    )
    sort_by: str | None = Field(
        default="name", description="name, size, modified or type."
    )
    sort_desc: bool | None = Field(default=False, description="Sort descending.")


@computer_tool("POST")
async def file_list_path(args: ListArguments, computer: SandboxClient) -> dict:
    """List a directory's contents."""
    return await computer.run("POST", "/v1/file/list", json_body=body(args))


class FindArguments(Arguments):
    path: str = Field(description="Directory to search.")
    glob: str = Field(description="Filename glob.")


@computer_tool("POST")
async def file_find_files(args: FindArguments, computer: SandboxClient) -> dict:
    """Find files by name pattern."""
    return await computer.run("POST", "/v1/file/find", json_body=body(args))


class GlobArguments(Arguments):
    path: str = Field(description="Base directory.")
    pattern: str = Field(description="Glob pattern (**, *, ?, [...]).")
    exclude: list[str] | None = Field(default=None, description="Globs to exclude.")
    include_hidden: bool | None = Field(
        default=False, description="Include hidden files."
    )
    files_only: bool | None = Field(
        default=True, description="Only files, not directories."
    )
    include_metadata: bool | None = Field(
        default=True, description="Include size and modified time."
    )
    max_results: int | None = Field(default=5000, description="Maximum results.")
    sort_by: str | None = Field(
        default="path", description="path, name, size or modified."
    )
    sort_desc: bool | None = Field(default=False, description="Sort descending.")


@computer_tool("POST")
async def file_glob_files(args: GlobArguments, computer: SandboxClient) -> dict:
    """Match files with a glob, optionally with metadata."""
    return await computer.run("POST", "/v1/file/glob", json_body=body(args))


class GrepArguments(Arguments):
    path: str = Field(description="File or directory to search.")
    pattern: str = Field(description="Regex or fixed string.")
    include: list[str] | None = Field(
        default=None, description='File globs to include, e.g. ["*.py"].'
    )
    exclude: list[str] | None = Field(default=None, description="Globs to exclude.")
    case_insensitive: bool | None = Field(default=False, description="Ignore case.")
    fixed_strings: bool | None = Field(
        default=False, description="Treat the pattern as literal text."
    )
    context_before: int | None = Field(default=0, description="Lines before (-B).")
    context_after: int | None = Field(default=0, description="Lines after (-A).")
    max_results: int | None = Field(default=500, description="Maximum matches.")
    max_file_size: str | None = Field(
        default="1M", description="Skip larger files, e.g. 1M or 500K."
    )
    multiline: bool | None = Field(
        default=False, description="Let patterns span lines."
    )
    offset: int | None = Field(default=0, description="Matches to skip.")
    type: str | None = Field(
        default=None, description='ripgrep file type, e.g. "py" or "js".'
    )
    recursive: bool | None = Field(default=True, description="Search recursively.")


@computer_tool("POST")
async def file_grep_files(args: GrepArguments, computer: SandboxClient) -> dict:
    """Search file contents across files with ripgrep."""
    return await computer.run("POST", "/v1/file/grep", json_body=body(args))


class UploadArguments(Arguments):
    file: str = Field(description="Text content of the file.")
    path: str | None = Field(default=None, description="Absolute destination path.")


@computer_tool("POST")
async def file_upload_file(args: UploadArguments, computer: SandboxClient) -> dict:
    """Upload text content as a file."""
    return await computer.run(
        "POST",
        "/v1/file/upload",
        data={"path": args.path} if args.path else None,
        files={"file": ("file", args.file.encode())},
    )


class DownloadArguments(Arguments):
    path: str = Field(description="Absolute file path.")
    change_policy: Literal["ignore", "abort"] | None = Field(
        default=None, description="abort if the file changes while downloading."
    )


@computer_tool("GET")
async def file_download_file(args: DownloadArguments, computer: SandboxClient) -> dict:
    """Download a file. Text files return their content; binary files only a size."""
    return await computer.run("GET", "/v1/file/download", params=body(args))


# Watching for changes


class CreateWatchArguments(Arguments):
    path: str = Field(description="Directory or file to watch.")
    recursive: bool = Field(default=True, description="Include subdirectories.")
    exclude: list[str] = Field(
        default=[
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "*.pyc",
            "*.pyo",
            ".DS_Store",
            "*.swp",
            "*.swo",
        ],
        description="Directories or globs to ignore.",
    )
    debounce: int = Field(default=300, description="Debounce window in ms.")
    include_patterns: list[str] = Field(
        default=[], description="Globs to include; empty includes everything."
    )


@computer_tool("POST")
async def file_create_watch(
    args: CreateWatchArguments, computer: SandboxClient
) -> dict:
    """Start watching a path for file changes."""
    return await computer.run("POST", "/v1/file/watch", json_body=body(args))


@computer_tool("GET")
async def file_list_watches(args: NoArguments, computer: SandboxClient) -> dict:
    """List active file watches."""
    return await computer.run("GET", "/v1/file/watch")


class WatcherArguments(Arguments):
    watcher_id: str = Field(description="Watcher ID from file_create_watch.")


class PollArguments(WatcherArguments):
    cursor: int = Field(default=0, description="Return only events after this cursor.")
    limit: int = Field(default=100, description="Maximum events.")
    timeout: int = Field(
        default=0, description="Seconds to long-poll; 0 returns immediately."
    )


@computer_tool("POST")
async def file_poll_events(args: PollArguments, computer: SandboxClient) -> dict:
    """Read new events from a file watch."""
    return await computer.run(
        "POST",
        f"/v1/file/watch/{segment(args.watcher_id)}/poll",
        json_body=args.model_dump(mode="json", exclude={"watcher_id"}),
    )


@computer_tool("GET")
async def file_watch_events(args: WatcherArguments, computer: SandboxClient) -> dict:
    """Stream a file watch's events for a few seconds."""
    return await computer.run(
        "GET", f"/v1/file/watch/{segment(args.watcher_id)}/events"
    )


@computer_tool("DELETE")
async def file_stop_watch(args: WatcherArguments, computer: SandboxClient) -> dict:
    """Stop a file watch."""
    return await computer.run("DELETE", f"/v1/file/watch/{segment(args.watcher_id)}")


class WaitForFileArguments(Arguments):
    path: str = Field(description="Exact file path to wait for.")
    timeout: int = Field(default=30, description="Maximum seconds to wait.")
    event_types: list[Literal["create", "write", "remove", "rename", "chmod"]] = Field(
        default=["create", "write", "remove", "rename", "chmod"],
        description="Events to wait for.",
    )


@computer_tool("POST")
async def file_wait_for_file(
    args: WaitForFileArguments, computer: SandboxClient
) -> dict:
    """Wait until a file is created, written, removed, renamed or changed."""
    return await computer.run("POST", "/v1/file/watch/wait", json_body=body(args))


TOOLS = [
    file_read_file,
    file_write_file,
    file_replace_in_file,
    # file_search_in_file,
    # file_str_replace_editor,
    # file_list_path,
    # file_find_files,
    # file_glob_files,
    file_grep_files,
    # file_upload_file,
    # file_download_file,
    # file_create_watch,
    # file_list_watches,
    # file_poll_events,
    # file_watch_events,
    # file_stop_watch,
    # file_wait_for_file,
]
