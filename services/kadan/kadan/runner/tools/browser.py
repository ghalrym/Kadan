"""Browser tools: the Chromium window on the computer's desktop.

Page tools drive the current tab through Playwright. `browser_execute_action`
drives the whole display with the mouse and keyboard.
"""

import json
import re
from typing import Any, Literal

from pydantic import Field, model_validator

from ...api.config import ServiceError
from ...api.sandbox_client import (
    MAX_RESULT_CHARACTERS,
    SandboxClient,
    segment,
    truncated,
)
from .base import Arguments, NoArguments, body, computer_tool

MouseButton = Literal["left", "right", "middle"]
NAVIGATE_EXCERPT_CHARACTERS = 1000
NAVIGATE_LINKS_LIMIT = 10
NAVIGATE_LINKS_MAX_CHARACTERS = 800
TRUNCATED_LINKS_LIMIT = 15
PAGE_RESULT_MAX_CHARACTERS = 6000
LINKS_BLOCK_MAX_CHARACTERS = 1000
MAX_LINK_URL_CHARACTERS = 220
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
NOISE_URL_MARKERS = (
    "googleadservices.com/",
    "doubleclick.net/",
    "googlesyndication.com/",
    "accounts.google.com/ServiceLogin",
    "adservice.google.",
)


def _is_noise_url(url: str) -> bool:
    lowered = url.lower()
    if any(marker.lower() in lowered for marker in NOISE_URL_MARKERS):
        return True
    # Drop giant ad/tracking URLs whose query strings explode token counts.
    return len(url) > MAX_LINK_URL_CHARACTERS and ("?" in url or "&" in url)


def _format_bounded_links(
    links: list[tuple[str, str]],
    limit: int,
    max_characters: int,
    prefix: str = "",
) -> list[str]:
    formatted: list[str] = []
    used = 0
    for label, url in links:
        if len(formatted) >= limit:
            break
        if _is_noise_url(url):
            continue
        clean_url = (
            url
            if len(url) <= MAX_LINK_URL_CHARACTERS
            else url[:MAX_LINK_URL_CHARACTERS]
        )
        entry = f"{prefix}[{label}]({clean_url})" if label else f"{prefix}{clean_url}"
        cost = len(entry) + (1 if formatted else 0)
        if formatted and used + cost > max_characters:
            break
        if not formatted and len(entry) > max_characters:
            continue
        formatted.append(entry)
        used += cost
    return formatted


def _extract_markdown(result: dict[str, Any]) -> tuple[str | None, dict[str, str]]:
    """Extract page markdown text and optional metadata from a sandbox response."""
    raw = result.get("body")
    if isinstance(raw, str):
        return raw, {}
    if not isinstance(raw, dict):
        return None, {}
    meta: dict[str, str] = {}
    sources = [raw]
    data = raw.get("data")
    if isinstance(data, dict):
        sources.append(data)
    for source in sources:
        for key in ("title", "url"):
            value = source.get(key)
            if isinstance(value, str) and value.strip() and key not in meta:
                meta[key] = value.strip()
    if isinstance(data, str):
        return data, meta
    for source in sources:
        for key in ("markdown", "content", "text"):
            value = source.get(key)
            if isinstance(value, str):
                return value, meta
    return None, meta


def _extract_links(markdown: str) -> list[tuple[str, str]]:
    """Return deduplicated (label, url) pairs from Markdown link syntax."""
    links: list[tuple[str, str]] = []
    seen: dict[str, int] = {}
    for raw_label, raw_url in MARKDOWN_LINK_RE.findall(markdown):
        url = raw_url.strip()
        if not url or url.startswith(("#", "javascript:", "data:")):
            continue
        label = " ".join(raw_label.split())
        if url in seen:
            existing_label, _ = links[seen[url]]
            if not existing_label and label:
                links[seen[url]] = (label, url)
            continue
        seen[url] = len(links)
        links.append((label, url))
    return links


def _select_top_links(
    links: list[tuple[str, str]],
    limit: int,
    max_characters: int = LINKS_BLOCK_MAX_CHARACTERS,
) -> list[str]:
    """Prioritize descriptive content links over navigation chrome and ad trackers."""
    descriptive: list[tuple[str, str]] = []
    short: list[tuple[str, str]] = []
    for label, url in links:
        if _is_noise_url(url):
            continue
        if len(label) >= 12 or len(label.split()) >= 2:
            descriptive.append((label, url))
        else:
            short.append((label, url))
    return _format_bounded_links([*descriptive, *short], limit, max_characters)


def _matches_query(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return all(term in lowered for term in terms)


def _filter_lines_by_query(text: str, query: str, context_lines: int = 3) -> str:
    """Return matching lines with surrounding context lines."""
    terms = [term for term in query.lower().split() if term]
    if not terms:
        return text
    lines = text.splitlines()
    keep: set[int] = set()
    for index, line in enumerate(lines):
        if _matches_query(line, terms):
            keep.update(
                range(
                    max(0, index - context_lines),
                    min(len(lines), index + context_lines + 1),
                )
            )
    sections: list[str] = []
    last = -2
    for index in sorted(keep):
        if sections and index > last + 1:
            sections.append("...")
        sections.append(lines[index])
        last = index
    return "\n".join(sections)


def _filter_markdown_by_query(markdown: str, query: str) -> str:
    """Return matching lines with surrounding context plus matching clean links."""
    terms = [term for term in query.lower().split() if term]
    if not terms:
        return markdown
    sections = _filter_lines_by_query(markdown, query, context_lines=3)
    matched_pairs = [
        (label, url)
        for label, url in _extract_links(markdown)
        if not _is_noise_url(url) and _matches_query(f"{label} {url}", terms)
    ]
    matching_links = _format_bounded_links(
        matched_pairs,
        TRUNCATED_LINKS_LIMIT,
        LINKS_BLOCK_MAX_CHARACTERS,
        prefix="- ",
    )
    if matching_links:
        link_block = "Matching links:\n" + "\n".join(matching_links)
        return f"{sections}\n\n{link_block}" if sections else link_block
    return sections


def _bounded_result(
    result: dict[str, Any], max_characters: int = PAGE_RESULT_MAX_CHARACTERS
) -> dict[str, Any]:
    """Ensure a raw sandbox result never exceeds max_characters."""
    raw = result.get("body")
    if isinstance(raw, str):
        if len(raw) > max_characters:
            return {
                **{k: v for k, v in result.items() if k != "body"},
                **truncated(raw, max_characters),
            }
        return result
    if isinstance(raw, (dict, list)):
        serialized = json.dumps(raw, separators=(",", ":"))
        if len(serialized) > max_characters:
            return {
                **{k: v for k, v in result.items() if k != "body"},
                **truncated(serialized, max_characters),
            }
    return result


def _compact_page_observation(result: dict[str, Any]) -> dict[str, Any] | None:
    """Build a compact post-navigation page observation from markdown output."""
    if not (200 <= result.get("status", 500) < 300):
        return None
    markdown, meta = _extract_markdown(result)
    if markdown is None or not markdown.strip():
        return None
    stripped = markdown.strip()
    observation: dict[str, Any] = {**meta}
    if len(stripped) <= NAVIGATE_EXCERPT_CHARACTERS:
        observation["excerpt"] = stripped
    else:
        observation["excerpt"] = stripped[:NAVIGATE_EXCERPT_CHARACTERS]
        observation["truncated"] = True
        observation["total_characters"] = len(stripped)
    top_links = _select_top_links(
        _extract_links(stripped), NAVIGATE_LINKS_LIMIT, NAVIGATE_LINKS_MAX_CHARACTERS
    )
    if top_links:
        observation["links"] = top_links
    return observation


# Page navigation


class NavigateArguments(Arguments):
    url: str = Field(description="Address to open.")
    wait_until: Literal["load", "domcontentloaded", "networkidle", "commit"] = Field(
        default="load", description="Page event that ends the navigation."
    )
    timeout: float = Field(default=30.0, description="Seconds to wait.")


@computer_tool("POST")
async def browser_navigate(args: NavigateArguments, computer: SandboxClient) -> dict:
    """Navigate the current page to a URL and include a compact page summary."""
    result = await computer.run(
        "POST", "/v1/browser/page/navigate", json_body=body(args)
    )
    if 200 <= result.get("status", 500) < 300:
        try:
            page_result = await computer.run(
                "GET", "/v1/browser/page/markdown", max_characters=None
            )
        except ServiceError:
            page_result = None
        if isinstance(page_result, dict):
            page = _compact_page_observation(page_result)
            if page is not None:
                result = {**result, "page": page}
    return result


@computer_tool("POST")
async def browser_go_back(args: NoArguments, computer: SandboxClient) -> dict:
    """Go back in browser history."""
    return await computer.run("POST", "/v1/browser/page/back")


@computer_tool("POST")
async def browser_go_forward(args: NoArguments, computer: SandboxClient) -> dict:
    """Go forward in browser history."""
    return await computer.run("POST", "/v1/browser/page/forward")


@computer_tool("POST")
async def browser_reload(args: NoArguments, computer: SandboxClient) -> dict:
    """Reload the current page."""
    return await computer.run("POST", "/v1/browser/page/reload")


# Reading the page


@computer_tool("GET")
async def browser_get_interactive_elements(
    args: NoArguments, computer: SandboxClient
) -> dict:
    """List the page's clickable inputs, buttons, and links with indexes."""
    result = await computer.run(
        "GET", "/v1/browser/page/elements", max_characters=PAGE_RESULT_MAX_CHARACTERS
    )
    return _bounded_result(result, PAGE_RESULT_MAX_CHARACTERS)


class GetMarkdownArguments(Arguments):
    query: str | None = Field(
        default=None,
        description="Optional search terms to return matching page sections and links.",
    )
    links_only: bool = Field(
        default=False,
        description="Return only deduplicated Markdown links from the page.",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Character offset into the selected content for reading further.",
    )


@computer_tool("GET")
async def browser_get_markdown(
    args: GetMarkdownArguments, computer: SandboxClient
) -> dict:
    """Get page content as Markdown, with optional search, link extraction, or offset."""
    result = await computer.run(
        "GET", "/v1/browser/page/markdown", max_characters=None
    )
    if not (200 <= result.get("status", 500) < 300) or "body" not in result:
        return _bounded_result(result)
    markdown, meta = _extract_markdown(result)
    if markdown is None:
        return _bounded_result(result)
    base = {k: v for k, v in result.items() if k != "body"}
    query = args.query.strip() if args.query else None
    links = [(label, url) for label, url in _extract_links(markdown) if not _is_noise_url(url)]
    if args.links_only:
        terms = [term for term in query.lower().split() if term] if query else []
        selected = [
            f"- [{label}]({url})" if label else f"- {url}"
            for label, url in links
            if not terms or _matches_query(f"{label} {url}", terms)
        ]
        content = "\n".join(selected)
        sliced = content[args.offset :]
        return {
            **base,
            **meta,
            "links_count": len(selected),
            "total_characters": len(content),
            "offset": args.offset,
            **truncated(sliced, PAGE_RESULT_MAX_CHARACTERS),
        }
    if query:
        content = _filter_markdown_by_query(markdown, query)
        sliced = content[args.offset :]
        return {
            **base,
            **meta,
            "query": query,
            "total_characters": len(markdown),
            "matched_characters": len(content),
            "offset": args.offset,
            **truncated(sliced, PAGE_RESULT_MAX_CHARACTERS),
        }
    sliced = markdown[args.offset :]
    if len(sliced) <= PAGE_RESULT_MAX_CHARACTERS:
        return {
            **base,
            **meta,
            **({"offset": args.offset} if args.offset else {}),
            "body": sliced,
        }
    top_links = (
        _select_top_links(links, TRUNCATED_LINKS_LIMIT, LINKS_BLOCK_MAX_CHARACTERS)
        if args.offset == 0
        else []
    )
    links_chars = sum(len(item) + 2 for item in top_links)
    body_limit = max(1000, PAGE_RESULT_MAX_CHARACTERS - links_chars)
    response: dict[str, Any] = {
        **base,
        **meta,
        **({"offset": args.offset} if args.offset else {}),
        "body": sliced[:body_limit],
        "truncated": True,
        "characters": len(markdown),
    }
    if top_links:
        response["links"] = top_links
    return response


class GetTextArguments(Arguments):
    query: str | None = Field(
        default=None,
        description="Optional search terms to return matching text lines with surrounding context.",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Character offset into the text for reading further.",
    )


@computer_tool("GET")
async def browser_get_text(args: GetTextArguments, computer: SandboxClient) -> dict:
    """Get clean visible text from the page, optionally filtered by query or offset."""
    result = await computer.run("GET", "/v1/browser/page/text", max_characters=None)
    if not (200 <= result.get("status", 500) < 300) or "body" not in result:
        return _bounded_result(result)
    text, meta = _extract_markdown(result)
    if text is None:
        return _bounded_result(result)
    base = {k: v for k, v in result.items() if k != "body"}
    query = args.query.strip() if args.query else None
    if query:
        content = _filter_lines_by_query(text, query, context_lines=3)
        sliced = content[args.offset :]
        return {
            **base,
            **meta,
            "query": query,
            "total_characters": len(text),
            "matched_characters": len(content),
            "offset": args.offset,
            **truncated(sliced, PAGE_RESULT_MAX_CHARACTERS),
        }
    sliced = text[args.offset :]
    if len(sliced) <= PAGE_RESULT_MAX_CHARACTERS:
        return {
            **base,
            **meta,
            **({"offset": args.offset} if args.offset else {}),
            "body": sliced,
        }
    return {
        **base,
        **meta,
        **({"offset": args.offset} if args.offset else {}),
        **truncated(sliced, PAGE_RESULT_MAX_CHARACTERS),
        "total_characters": len(text),
    }


class GetHtmlArguments(Arguments):
    outer: bool = Field(
        default=False, description="Include the outer HTML of the document element."
    )


@computer_tool("GET")
async def browser_get_html(args: GetHtmlArguments, computer: SandboxClient) -> dict:
    """Get the page HTML content."""
    return await computer.run("GET", "/v1/browser/page/html", params=body(args))


class FindTextArguments(Arguments):
    keyword: str = Field(description="Text to find.")


@computer_tool("POST")
async def browser_find_text(args: FindTextArguments, computer: SandboxClient) -> dict:
    """Find text occurrences on the page."""
    return await computer.run(
        "POST", "/v1/browser/page/find_text", json_body=body(args)
    )


class EvaluateArguments(Arguments):
    expression: str = Field(description="JavaScript expression to run in the page.")


@computer_tool("POST")
async def browser_evaluate(args: EvaluateArguments, computer: SandboxClient) -> dict:
    """Execute JavaScript in the page and return a compact result."""
    result = await computer.run(
        "POST",
        "/v1/browser/page/evaluate",
        json_body=body(args),
        max_characters=PAGE_RESULT_MAX_CHARACTERS,
    )
    return _bounded_result(result, PAGE_RESULT_MAX_CHARACTERS)


class ConsoleArguments(Arguments):
    clear: bool = Field(default=False, description="Clear the log after reading it.")


@computer_tool("GET")
async def browser_get_console_logs(
    args: ConsoleArguments, computer: SandboxClient
) -> dict:
    """Get captured browser console log messages."""
    return await computer.run("GET", "/v1/browser/page/console", params=body(args))


class ExportConsoleArguments(Arguments):
    save_path: str = Field(description="Absolute path of the JSON file to write.")
    clear: bool = Field(default=False, description="Clear the log after exporting.")


@computer_tool("POST")
async def browser_export_console_logs(
    args: ExportConsoleArguments, computer: SandboxClient
) -> dict:
    """Export console logs to a JSON file."""
    return await computer.run(
        "POST", "/v1/browser/page/console/export", json_body=body(args)
    )


# Interacting with elements


class ClickArguments(Arguments):
    selector: str | None = Field(default=None, description="CSS selector to click.")
    index: int | None = Field(
        default=None, description="Index from browser_get_interactive_elements."
    )
    x: float | None = Field(default=None, description="Page X coordinate.")
    y: float | None = Field(default=None, description="Page Y coordinate.")
    button: str = Field(default="left", description="Mouse button.")
    click_count: int = Field(default=1, description="Number of clicks.")


@computer_tool("POST")
async def browser_click(args: ClickArguments, computer: SandboxClient) -> dict:
    """Click an element by selector, index, or coordinates."""
    return await computer.run("POST", "/v1/browser/page/click", json_body=body(args))


class FillArguments(Arguments):
    selector: str | None = Field(default=None, description="CSS selector of the input.")
    index: int | None = Field(
        default=None, description="Index from browser_get_interactive_elements."
    )
    text: str = Field(description="Text to put in the input.")


@computer_tool("POST")
async def browser_fill(args: FillArguments, computer: SandboxClient) -> dict:
    """Fill an input field, replacing its value."""
    return await computer.run("POST", "/v1/browser/page/fill", json_body=body(args))


class FillFormArguments(Arguments):
    items: list[dict[str, Any]] = Field(description="Fields to fill, in order.")


@computer_tool("POST")
async def browser_fill_form(args: FillFormArguments, computer: SandboxClient) -> dict:
    """Batch fill multiple form fields."""
    return await computer.run(
        "POST", "/v1/browser/page/fill_form", json_body=body(args)
    )


class TypeTextArguments(Arguments):
    text: str = Field(description="Text to type into the focused element.")
    delay: float = Field(default=0, description="Milliseconds between keystrokes.")


@computer_tool("POST")
async def browser_type_text(args: TypeTextArguments, computer: SandboxClient) -> dict:
    """Type text into the focused element, with optional delay between keystrokes."""
    return await computer.run("POST", "/v1/browser/page/type", json_body=body(args))


class PressKeyArguments(Arguments):
    key: str = Field(description='Key to press, e.g. "Enter", "Tab", "Escape".')


@computer_tool("POST")
async def browser_press_key(args: PressKeyArguments, computer: SandboxClient) -> dict:
    """Press a single key."""
    return await computer.run(
        "POST", "/v1/browser/page/press_key", json_body=body(args)
    )


class HotKeyArguments(Arguments):
    keys: list[str] = Field(description='Keys pressed together, e.g. ["Control", "a"].')


@computer_tool("POST")
async def browser_hot_key(args: HotKeyArguments, computer: SandboxClient) -> dict:
    """Press a key combination."""
    return await computer.run("POST", "/v1/browser/page/hot_key", json_body=body(args))


class HoverArguments(Arguments):
    selector: str | None = Field(default=None, description="CSS selector to hover.")
    x: float | None = Field(default=None, description="Page X coordinate.")
    y: float | None = Field(default=None, description="Page Y coordinate.")


@computer_tool("POST")
async def browser_hover(args: HoverArguments, computer: SandboxClient) -> dict:
    """Hover over an element by selector or coordinates."""
    return await computer.run("POST", "/v1/browser/page/hover", json_body=body(args))


class SelectOptionArguments(Arguments):
    selector: str = Field(description="CSS selector of the dropdown.")
    value: str | None = Field(default=None, description="Option value to select.")
    label: str | None = Field(default=None, description="Option label to select.")
    index: int | None = Field(default=None, description="Option index to select.")


@computer_tool("POST")
async def browser_select_option(
    args: SelectOptionArguments, computer: SandboxClient
) -> dict:
    """Select an option in a dropdown."""
    return await computer.run(
        "POST", "/v1/browser/page/select_option", json_body=body(args)
    )


class SelectorArguments(Arguments):
    selector: str = Field(description="CSS selector of the element.")


@computer_tool("POST")
async def browser_check(args: SelectorArguments, computer: SandboxClient) -> dict:
    """Check a checkbox."""
    return await computer.run("POST", "/v1/browser/page/check", json_body=body(args))


@computer_tool("POST")
async def browser_uncheck(args: SelectorArguments, computer: SandboxClient) -> dict:
    """Uncheck a checkbox."""
    return await computer.run("POST", "/v1/browser/page/uncheck", json_body=body(args))


class UploadArguments(Arguments):
    selector: str = Field(description="CSS selector of the file input.")
    files: list[str] = Field(description="Absolute paths of files on the computer.")


@computer_tool("POST")
async def browser_upload_file(args: UploadArguments, computer: SandboxClient) -> dict:
    """Upload files from the computer to a file input element."""
    return await computer.run(
        "POST", "/v1/browser/page/upload_file", json_body=body(args)
    )


# Scrolling


class ScrollArguments(Arguments):
    direction: str = Field(default="down", description="up, down, left or right.")
    amount: int = Field(default=300, description="Pixels to scroll.")


@computer_tool("POST")
async def browser_scroll(args: ScrollArguments, computer: SandboxClient) -> dict:
    """Scroll the page in a direction."""
    return await computer.run("POST", "/v1/browser/page/scroll", json_body=body(args))


class ScrollToArguments(Arguments):
    x: int = Field(default=0, description="Horizontal position in pixels.")
    y: int = Field(default=0, description="Vertical position in pixels.")


@computer_tool("POST")
async def browser_scroll_to(args: ScrollToArguments, computer: SandboxClient) -> dict:
    """Scroll to an absolute position."""
    return await computer.run(
        "POST", "/v1/browser/page/scroll_to", json_body=body(args)
    )


@computer_tool("POST")
async def browser_scroll_to_element(
    args: SelectorArguments, computer: SandboxClient
) -> dict:
    """Scroll an element into view."""
    return await computer.run(
        "POST", "/v1/browser/page/scroll_to_element", json_body=body(args)
    )


# Waiting


class WaitArguments(Arguments):
    type: Literal[
        "selector",
        "load",
        "url",
        "network_idle",
        "download",
        "function",
        "response",
        "request",
        "timeout",
    ] = Field(description="What to wait for.")
    selector: str | None = Field(default=None, description="For selector waits.")
    state: str | None = Field(
        default=None, description="Element or load state to wait for."
    )
    url: str | None = Field(default=None, description="For url waits.")
    save_path: str | None = Field(default=None, description="For download waits.")
    timeout: float = Field(default=30.0, description="Seconds to wait.")
    expression: str | None = Field(
        default=None, description="JavaScript for function waits."
    )
    polling: float | None = Field(
        default=None, description="Polling interval for function waits."
    )
    url_pattern: str | None = Field(
        default=None, description="For response and request waits."
    )


@computer_tool("POST")
async def browser_wait(args: WaitArguments, computer: SandboxClient) -> dict:
    """Wait for a selector, load, URL, network idle, download, function, response,
    request, or a fixed timeout."""
    return await computer.run("POST", "/v1/browser/page/wait", json_body=body(args))


# Screens and recordings


class PageScreenshotArguments(Arguments):
    full_page: bool = Field(default=False, description="Capture the whole page.")
    format: str = Field(default="png", description="png or jpeg.")
    quality: int | None = Field(default=None, description="JPEG quality 0-100.")


@computer_tool("GET")
async def browser_page_screenshot(
    args: PageScreenshotArguments, computer: SandboxClient
) -> dict:
    """Capture a screenshot of the current page. You cannot view the image."""
    return await computer.run("GET", "/v1/browser/page/screenshot", params=body(args))


class ScreenshotArguments(Arguments):
    format: Literal["png", "jpg", "jpeg"] = Field(default="png", description="Format.")
    quality: int | None = Field(default=None, description="JPEG quality 0-100.")


@computer_tool("GET")
async def browser_take_screenshot(
    args: ScreenshotArguments, computer: SandboxClient
) -> dict:
    """Take a screenshot of the whole display. You cannot view the image."""
    return await computer.run("GET", "/v1/browser/screenshot", params=body(args))


class PageRecordArguments(Arguments):
    action: Literal["once", "start", "pause", "resume", "stop", "status"] = Field(
        default="once", description="Recording action."
    )
    save_path: str | None = Field(default=None, description="Output file path.")
    duration: float = Field(default=10.0, description="Seconds for once.")
    fps: int = Field(default=15, description="Frames per second.")
    quality: int = Field(default=80, description="Image quality 0-100.")


@computer_tool("POST")
async def browser_page_record(
    args: PageRecordArguments, computer: SandboxClient
) -> dict:
    """Record a screencast of the page (once, start, pause, resume, stop, status)."""
    return await computer.run("POST", "/v1/browser/page/record", json_body=body(args))


# Tabs


@computer_tool("GET")
async def browser_list_tabs(args: NoArguments, computer: SandboxClient) -> dict:
    """List all open browser tabs."""
    return await computer.run("GET", "/v1/browser/tabs")


class CreateTabArguments(Arguments):
    url: str | None = Field(default=None, description="Address to open in the tab.")


@computer_tool("POST")
async def browser_create_tab(args: CreateTabArguments, computer: SandboxClient) -> dict:
    """Create a new browser tab."""
    return await computer.run("POST", "/v1/browser/tabs", json_body=body(args))


class TabArguments(Arguments):
    index: int = Field(description="Tab index from browser_list_tabs.")


@computer_tool("DELETE")
async def browser_close_tab(args: TabArguments, computer: SandboxClient) -> dict:
    """Close a browser tab by index."""
    return await computer.run("DELETE", f"/v1/browser/tabs/{segment(args.index)}")


@computer_tool("PUT")
async def browser_activate_tab(args: TabArguments, computer: SandboxClient) -> dict:
    """Bring a browser tab to the front by index."""
    return await computer.run("PUT", f"/v1/browser/tabs/{segment(args.index)}/activate")


# Cookies and stored state


class GetCookiesArguments(Arguments):
    urls: str | None = Field(
        default=None, description="Comma-separated URLs to filter by."
    )


@computer_tool("GET")
async def browser_get_cookies(
    args: GetCookiesArguments, computer: SandboxClient
) -> dict:
    """Get browser cookies, optionally filtered by URLs."""
    return await computer.run("GET", "/v1/browser/cookies", params=body(args))


class SetCookiesArguments(Arguments):
    cookies: list[dict[str, Any]] = Field(
        description="Cookies with name, value and domain or url."
    )


@computer_tool("POST")
async def browser_set_cookies(
    args: SetCookiesArguments, computer: SandboxClient
) -> dict:
    """Set browser cookies."""
    return await computer.run("POST", "/v1/browser/cookies", json_body=body(args))


@computer_tool("DELETE")
async def browser_clear_cookies(args: NoArguments, computer: SandboxClient) -> dict:
    """Clear all browser cookies."""
    return await computer.run("DELETE", "/v1/browser/cookies")


class StatePathArguments(Arguments):
    path: str = Field(description="Absolute path of the state file.")


@computer_tool("POST")
async def browser_save_state(args: StatePathArguments, computer: SandboxClient) -> dict:
    """Save browser state (cookies, localStorage, etc.) to a file."""
    return await computer.run("POST", "/v1/browser/state/save", json_body=body(args))


@computer_tool("POST")
async def browser_load_state(args: StatePathArguments, computer: SandboxClient) -> dict:
    """Load browser state from a previously saved file."""
    return await computer.run("POST", "/v1/browser/state/load", json_body=body(args))


# Network


class HeadersArguments(Arguments):
    headers: dict[str, str] = Field(description="Header names and values.")


@computer_tool("POST")
async def browser_set_extra_headers(
    args: HeadersArguments, computer: SandboxClient
) -> dict:
    """Set extra HTTP headers for all subsequent requests."""
    return await computer.run(
        "POST", "/v1/browser/network/headers", json_body=body(args)
    )


class ScopedHeadersArguments(Arguments):
    origin: str = Field(description="Origin the headers apply to.")
    headers: dict[str, str] = Field(description="Header names and values.")


@computer_tool("POST")
async def browser_set_scoped_headers(
    args: ScopedHeadersArguments, computer: SandboxClient
) -> dict:
    """Set HTTP headers scoped to a specific origin."""
    return await computer.run(
        "POST", "/v1/browser/network/scoped_headers", json_body=body(args)
    )


class MockResponse(Arguments):
    status: int = Field(default=200, description="HTTP status.")
    headers: dict[str, str] = Field(default_factory=dict, description="Headers.")
    body: str = Field(default="", description="Response body.")
    content_type: str = Field(default="text/plain", description="Content type.")


class AddRouteArguments(Arguments):
    url_pattern: str = Field(description="URL glob of requests to intercept.")
    response: MockResponse | None = Field(
        default=None, description="Response to return instead of the network."
    )
    abort: bool = Field(default=False, description="Abort matching requests.")


@computer_tool("POST")
async def browser_add_route(args: AddRouteArguments, computer: SandboxClient) -> dict:
    """Add a network route to mock or abort matching requests."""
    return await computer.run("POST", "/v1/browser/network/route", json_body=body(args))


class RemoveRouteArguments(Arguments):
    url_pattern: str = Field(description="URL glob of the route to remove.")


@computer_tool("DELETE")
async def browser_remove_route(
    args: RemoveRouteArguments, computer: SandboxClient
) -> dict:
    """Remove a previously added network route."""
    return await computer.run(
        "DELETE", "/v1/browser/network/route", json_body=body(args)
    )


class RequestsArguments(Arguments):
    filter: str | None = Field(default=None, description="Substring of request URLs.")
    limit: int = Field(default=100, description="Maximum requests to return.")


@computer_tool("GET")
async def browser_get_requests(
    args: RequestsArguments, computer: SandboxClient
) -> dict:
    """Get tracked network requests."""
    return await computer.run("GET", "/v1/browser/network/requests", params=body(args))


class ExportHarArguments(Arguments):
    save_path: str = Field(description="Absolute path of the HAR file to write.")


@computer_tool("POST")
async def browser_export_har(args: ExportHarArguments, computer: SandboxClient) -> dict:
    """Export tracked network requests as a HAR file."""
    return await computer.run(
        "POST", "/v1/browser/network/export_har", json_body=body(args)
    )


@computer_tool("GET")
async def browser_get_proxy_pac(args: NoArguments, computer: SandboxClient) -> dict:
    """Return the current PAC file for proxy split-routing, if configured."""
    return await computer.run("GET", "/v1/browser/proxy.pac")


# CAPTCHAs


@computer_tool("GET")
async def browser_detect_captcha(args: NoArguments, computer: SandboxClient) -> dict:
    """Detect a CAPTCHA on the current page."""
    return await computer.run("GET", "/v1/browser/captcha/detect")


class WaitCaptchaArguments(Arguments):
    timeout: float = Field(default=60.0, description="Seconds to wait.")
    poll_interval: float = Field(default=2.0, description="Seconds between checks.")


@computer_tool("POST")
async def browser_wait_for_captcha(
    args: WaitCaptchaArguments, computer: SandboxClient
) -> dict:
    """Wait for a CAPTCHA to be resolved, for example by the user."""
    return await computer.run("POST", "/v1/browser/captcha/wait", json_body=body(args))


# Display mouse and keyboard

DisplayActionType = Literal[
    "CLICK",
    "MOVE_TO",
    "MOVE_REL",
    "MOUSE_DOWN",
    "MOUSE_UP",
    "RIGHT_CLICK",
    "DOUBLE_CLICK",
    "DRAG_TO",
    "DRAG_REL",
    "SCROLL",
    "TYPING",
    "PRESS",
    "KEY_DOWN",
    "KEY_UP",
    "HOTKEY",
    "WAIT",
]

_ACTION_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "MOVE_TO": ("x", "y"),
    "DRAG_TO": ("x", "y"),
    "MOVE_REL": ("x_offset", "y_offset"),
    "DRAG_REL": ("x_offset", "y_offset"),
    "TYPING": ("text",),
    "PRESS": ("key",),
    "KEY_DOWN": ("key",),
    "KEY_UP": ("key",),
    "HOTKEY": ("keys",),
    "WAIT": ("duration",),
}

_ACTION_DEFAULTS: dict[str, dict[str, Any]] = {
    "CLICK": {"button": "left", "num_clicks": 1},
    "MOUSE_DOWN": {"button": "left"},
    "MOUSE_UP": {"button": "left"},
    "SCROLL": {"dx": 0, "dy": 0},
    "TYPING": {"use_clipboard": True},
}


class ExecuteActionArguments(Arguments):
    action_type: DisplayActionType = Field(description="Mouse or keyboard action type.")
    x: float | None = Field(default=None, description="Display X coordinate.")
    y: float | None = Field(default=None, description="Display Y coordinate.")
    x_offset: float | None = Field(default=None, description="Relative X offset.")
    y_offset: float | None = Field(default=None, description="Relative Y offset.")
    button: MouseButton | None = Field(
        default=None, description="Mouse button (left, right, middle)."
    )
    num_clicks: Literal[1, 2, 3] | None = Field(
        default=None, description="Click count for CLICK (1, 2, or 3)."
    )
    dx: int | None = Field(default=None, description="Horizontal scroll delta.")
    dy: int | None = Field(default=None, description="Vertical scroll delta.")
    text: str | None = Field(default=None, description="Text for TYPING.")
    use_clipboard: bool | None = Field(
        default=None, description="Use clipboard for TYPING."
    )
    key: str | None = Field(
        default=None, description="Key name for PRESS, KEY_DOWN, or KEY_UP."
    )
    keys: list[str] | None = Field(default=None, description="Key list for HOTKEY.")
    duration: float | None = Field(default=None, description="Seconds for WAIT.")

    @model_validator(mode="after")
    def validate_required_fields(self) -> "ExecuteActionArguments":
        missing = [
            field
            for field in _ACTION_REQUIRED_FIELDS.get(self.action_type, ())
            if getattr(self, field) is None
        ]
        if missing:
            raise ValueError(
                f"{self.action_type} requires {', '.join(missing)}"
            )
        return self

    def to_payload(self) -> dict[str, Any]:
        payload = {**_ACTION_DEFAULTS.get(self.action_type, {}), **body(self)}
        return payload


@computer_tool("POST")
async def browser_execute_action(
    args: ExecuteActionArguments, computer: SandboxClient
) -> dict:
    """Perform one mouse or keyboard action on the whole display: click, move,
    drag, scroll, type, press keys, hotkey, or wait."""
    return await computer.run(
        "POST", "/v1/browser/actions", json_body=args.to_payload()
    )


# Browser process


@computer_tool("GET")
async def browser_get_browser_info(args: NoArguments, computer: SandboxClient) -> dict:
    """Get browser information: CDP URL, viewport size and so on."""
    return await computer.run("GET", "/v1/browser/info")


class Resolution(Arguments):
    width: int
    height: int


class SetConfigArguments(Arguments):
    resolution: Resolution | None = Field(
        default=None,
        description="Screen resolution: 1920x1080, 640x480, 1360x768, 1280x720, "
        "800x600, 1024x768 or 1280x800.",
    )


@computer_tool("POST")
async def browser_set_config(args: SetConfigArguments, computer: SandboxClient) -> dict:
    """Change the display configuration, such as its resolution."""
    return await computer.run("POST", "/v1/browser/config", json_body=body(args))


class RestartArguments(Arguments):
    mode: Literal["soft", "hard"] = Field(
        default="soft",
        description="soft reconnects automation; hard restarts the browser process.",
    )
    url_blocklist: list[str] | None = Field(
        default=None, description="Hard restart only: URLs to block."
    )
    url_allowlist: list[str] | None = Field(
        default=None, description="Hard restart only: the only URLs to allow."
    )
    allow_file_selection_dialogs: bool | None = Field(
        default=None, description="Hard restart only: allow file pickers."
    )
    locale: str | None = Field(default=None, description="Hard restart only: locale.")


@computer_tool("POST")
async def browser_restart(args: RestartArguments, computer: SandboxClient) -> dict:
    """Restart the browser session."""
    return await computer.run("POST", "/v1/browser/restart", json_body=body(args))


TOOLS = [
    browser_navigate,
    # browser_go_back,
    # browser_go_forward,
    # browser_reload,
    browser_get_interactive_elements,
    # browser_get_markdown,
    browser_get_text,
    # browser_get_html,
    # browser_find_text,
    browser_evaluate,
    # browser_get_console_logs,
    # browser_export_console_logs,
    # browser_click,
    # browser_fill,
    # browser_fill_form,
    # browser_type_text,
    # browser_press_key,
    # browser_hot_key,
    # browser_hover,
    # browser_select_option,
    # browser_check,
    # browser_uncheck,
    # browser_upload_file,
    # browser_scroll,
    # browser_scroll_to,
    # browser_scroll_to_element,
    # browser_wait,
    # browser_page_screenshot,
    # browser_take_screenshot,
    # browser_page_record,
    # browser_list_tabs,
    # browser_create_tab,
    # browser_close_tab,
    # browser_activate_tab,
    # browser_get_cookies,
    # browser_set_cookies,
    # browser_clear_cookies,
    # browser_save_state,
    # browser_load_state,
    # browser_set_extra_headers,
    # browser_set_scoped_headers,
    # browser_add_route,
    # browser_remove_route,
    # browser_get_requests,
    # browser_export_har,
    # browser_get_proxy_pac,
    # browser_detect_captcha,
    # browser_wait_for_captcha,
    browser_execute_action,
    # browser_get_browser_info,
    # browser_set_config,
    # browser_restart,
]
