"""Browser tools: the Chromium window on the computer's desktop.

Page tools drive the current tab through Playwright. `browser_execute_action`
drives the whole display with the mouse and keyboard.
"""

from typing import Annotated, Any, Literal

from pydantic import Field

from ...api.sandbox_client import SandboxClient, segment
from .base import Arguments, NoArguments, body, computer_tool

MouseButton = Literal["left", "right", "middle"]


# Page navigation


class NavigateArguments(Arguments):
    url: str = Field(description="Address to open.")
    wait_until: Literal["load", "domcontentloaded", "networkidle", "commit"] = Field(
        default="load", description="Page event that ends the navigation."
    )
    timeout: float = Field(default=30.0, description="Seconds to wait.")


@computer_tool("POST")
async def browser_navigate(args: NavigateArguments, computer: SandboxClient) -> dict:
    """Navigate the current page to a URL."""
    return await computer.run("POST", "/v1/browser/page/navigate", json_body=body(args))


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
    """List the page's inputs, buttons and links, each with an index.

    Indexes change as a page loads, so list elements again right before using one.
    """
    return await computer.run("GET", "/v1/browser/page/elements")


@computer_tool("GET")
async def browser_get_markdown(args: NoArguments, computer: SandboxClient) -> dict:
    """Get the page content as Markdown (Readability and Turndown)."""
    return await computer.run("GET", "/v1/browser/page/markdown")


@computer_tool("GET")
async def browser_get_text(args: NoArguments, computer: SandboxClient) -> dict:
    """Get all visible text from the page."""
    return await computer.run("GET", "/v1/browser/page/text")


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
    """Execute JavaScript in the page and return the result."""
    return await computer.run("POST", "/v1/browser/page/evaluate", json_body=body(args))


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


class MoveTo(Arguments):
    action_type: Literal["MOVE_TO"] = "MOVE_TO"
    x: float
    y: float


class MoveRel(Arguments):
    action_type: Literal["MOVE_REL"] = "MOVE_REL"
    x_offset: float
    y_offset: float


class Click(Arguments):
    action_type: Literal["CLICK"] = "CLICK"
    x: float | None = None
    y: float | None = None
    button: MouseButton = "left"
    num_clicks: Literal[1, 2, 3] = 1


class MouseDown(Arguments):
    action_type: Literal["MOUSE_DOWN"] = "MOUSE_DOWN"
    button: MouseButton = "left"


class MouseUp(Arguments):
    action_type: Literal["MOUSE_UP"] = "MOUSE_UP"
    button: MouseButton = "left"


class RightClick(Arguments):
    action_type: Literal["RIGHT_CLICK"] = "RIGHT_CLICK"
    x: float | None = None
    y: float | None = None


class DoubleClick(Arguments):
    action_type: Literal["DOUBLE_CLICK"] = "DOUBLE_CLICK"
    x: float | None = None
    y: float | None = None


class DragTo(Arguments):
    action_type: Literal["DRAG_TO"] = "DRAG_TO"
    x: float
    y: float


class DragRel(Arguments):
    action_type: Literal["DRAG_REL"] = "DRAG_REL"
    x_offset: float
    y_offset: float


class Scroll(Arguments):
    action_type: Literal["SCROLL"] = "SCROLL"
    dx: int = 0
    dy: int = 0


class Typing(Arguments):
    action_type: Literal["TYPING"] = "TYPING"
    text: str
    use_clipboard: bool | None = True


class Press(Arguments):
    action_type: Literal["PRESS"] = "PRESS"
    key: str


class KeyDown(Arguments):
    action_type: Literal["KEY_DOWN"] = "KEY_DOWN"
    key: str


class KeyUp(Arguments):
    action_type: Literal["KEY_UP"] = "KEY_UP"
    key: str


class Hotkey(Arguments):
    action_type: Literal["HOTKEY"] = "HOTKEY"
    keys: list[str]


class Wait(Arguments):
    action_type: Literal["WAIT"] = "WAIT"
    duration: float


DisplayAction = Annotated[
    MoveTo
    | MoveRel
    | Click
    | MouseDown
    | MouseUp
    | RightClick
    | DoubleClick
    | DragTo
    | DragRel
    | Scroll
    | Typing
    | Press
    | KeyDown
    | KeyUp
    | Hotkey
    | Wait,
    Field(discriminator="action_type"),
]


class ExecuteActionArguments(Arguments):
    action: DisplayAction = Field(description="One mouse or keyboard action.")


@computer_tool("POST")
async def browser_execute_action(
    args: ExecuteActionArguments, computer: SandboxClient
) -> dict:
    """Perform one mouse or keyboard action on the whole display, outside page
    automation: move, click, drag, scroll, type, press keys or wait."""
    return await computer.run(
        "POST", "/v1/browser/actions", json_body=body(args)["action"]
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
    # browser_get_interactive_elements,
    browser_get_markdown,
    # browser_get_text,
    # browser_get_html,
    # browser_find_text,
    # browser_evaluate,
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
    browser_take_screenshot,
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
