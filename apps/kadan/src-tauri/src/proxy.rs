use crate::config::{normalize_host, save_config, DesktopConfig, RunMode};
use crate::hardware::collect_hardware;
use crate::host_config::{save_config as save_host_config, ServerConfig};
use crate::host_updater::AppUpdater;
use crate::supervisor::{ServerSupervisor, ServiceId};
use crate::updater;
use crate::work_sidecar::WorkSidecar;
use axum::body::Body;
use axum::extract::ws::{Message as AxumMsg, WebSocket, WebSocketUpgrade};
use axum::extract::{FromRequestParts, Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{Json, Router};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message as TungMsg;

#[derive(Clone)]
pub struct ProxyState {
    pub config: Arc<RwLock<DesktopConfig>>,
    pub web_root: PathBuf,
    pub client: reqwest::Client,
    pub app_handle: Option<tauri::AppHandle>,
    pub work_sidecar: WorkSidecar,
    /// Runs Kadan's services when this app hosts them.
    pub supervisor: Arc<ServerSupervisor>,
    pub host_updater: Arc<AppUpdater>,
}

impl ProxyState {
    fn mode(&self) -> RunMode {
        self.config.read().unwrap().run_mode()
    }

    /// Where requests go: this computer's services, or the chosen server.
    fn upstream(&self) -> String {
        let port = self.supervisor.config().server_port;
        self.config.read().unwrap().upstream(port)
    }

    fn unreachable_detail(&self, upstream: &str) -> String {
        match self.mode() {
            RunMode::Host => "Kadan is still starting on this computer, or it is stopped. Try again in a moment, or start it from Kadan Server → Settings.".to_string(),
            RunMode::Connect => format!("Can’t reach Kadan at {upstream}. Check the address in Settings."),
        }
    }
}

#[derive(Deserialize)]
struct TestRequest {
    host: String,
}

#[derive(Serialize)]
struct TestResponse {
    ok: bool,
    message: String,
    version: Option<String>,
    app: Option<String>,
    latency_ms: Option<u64>,
}

pub async fn start_loopback_proxy(
    config: DesktopConfig,
    web_root: PathBuf,
    app_handle: Option<tauri::AppHandle>,
    work_sidecar: WorkSidecar,
    supervisor: Arc<ServerSupervisor>,
    host_updater: Arc<AppUpdater>,
) -> Result<(SocketAddr, Arc<RwLock<DesktopConfig>>), String> {
    let shared_config = Arc::new(RwLock::new(config));
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))?;

    let state = ProxyState {
        config: shared_config.clone(),
        web_root,
        client,
        app_handle,
        work_sidecar,
        supervisor,
        host_updater,
    };

    let app = Router::new()
        .route("/__kadan_desktop__/config", get(get_config).put(put_config))
        .route("/__kadan_desktop__/test", post(test_connection))
        .route("/__kadan_desktop__/pick-folder", post(pick_folder))
        .route("/__kadan_desktop__/save-download", post(save_download))
        .route("/__kadan_desktop__/updates", get(check_desktop_updates))
        .route("/__kadan_desktop__/updates/install", post(install_desktop_update))
        .route("/__kadan_desktop__/host", get(host_status))
        .route("/__kadan_desktop__/host/settings", axum::routing::put(put_host_settings))
        .route("/__kadan_desktop__/host/start", post(host_start))
        .route("/__kadan_desktop__/host/stop", post(host_stop))
        .route("/__kadan_desktop__/host/restart", post(host_restart))
        .route("/__kadan_desktop__/host/logs", get(host_logs))
        .route("/kadan/api/work/{*path}", any(proxy_local_work))
        .route("/kadan/api/work", any(proxy_local_work))
        .route("/kadan/api/{*path}", any(proxy_api))
        .route("/kadan/api", any(proxy_api))
        .route("/tsuchi/api/{*path}", any(proxy_api))
        .route("/tsuchi/api", any(proxy_api))
        .route("/hikari/v1/{*path}", any(proxy_api))
        .route("/hikari/model-lifecycle/{*path}", any(proxy_api))
        .route("/hikari/health", any(proxy_api))
        .route("/hikari/openapi.json", any(proxy_api))
        .route("/hikari/docs", any(proxy_api))
        .route("/api/{*path}", any(proxy_api))
        .route("/api", any(proxy_api))
        .fallback(get(serve_frontend))
        .with_state(state);

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| format!("Could not bind local desktop proxy: {e}"))?;
    let addr = listener
        .local_addr()
        .map_err(|e| format!("Could not read proxy address: {e}"))?;

    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    Ok((addr, shared_config))
}

async fn get_config(State(state): State<ProxyState>) -> Json<DesktopConfig> {
    let cfg = state.config.read().unwrap().clone();
    Json(cfg)
}

async fn put_config(
    State(state): State<ProxyState>,
    Json(incoming): Json<DesktopConfig>,
) -> Response {
    let before = state.mode();
    match save_config(incoming) {
        Ok(saved) => {
            *state.config.write().unwrap() = saved.clone();
            // Switching where Kadan runs starts or stops this computer's services.
            match (before, saved.run_mode()) {
                (RunMode::Connect, RunMode::Host) => {
                    let supervisor = state.supervisor.clone();
                    tokio::spawn(async move {
                        let _ = supervisor.start_all().await;
                    });
                }
                (RunMode::Host, RunMode::Connect) => state.supervisor.stop_all().await,
                _ => {}
            }
            state
                .work_sidecar
                .update_upstream(&state.client, &state.upstream())
                .await;
            Json(saved).into_response()
        }
        Err(detail) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({ "detail": detail })),
        )
            .into_response(),
    }
}

async fn test_connection(
    State(state): State<ProxyState>,
    Json(payload): Json<TestRequest>,
) -> Response {
    let host = match normalize_host(&payload.host) {
        Ok(h) => h,
        Err(detail) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({ "detail": detail })),
            )
                .into_response();
        }
    };
    let start = Instant::now();
    let health_urls = [
        format!("{host}/kadan/api/health"),
        format!("{host}/api/health"),
    ];
    let mut last_status: Option<u16> = None;
    for health_url in &health_urls {
        match state
            .client
            .get(health_url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                let latency_ms = start.elapsed().as_millis() as u64;
                let body: serde_json::Value = resp.json().await.unwrap_or_default();
                let version = body
                    .get("version")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let app = body
                    .get("app")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                return Json(TestResponse {
                    ok: true,
                    message: "Connected to Kadan Server.".to_string(),
                    version,
                    app,
                    latency_ms: Some(latency_ms),
                })
                .into_response();
            }
            Ok(resp) => {
                last_status = Some(resp.status().as_u16());
            }
            Err(_) => {
                break;
            }
        }
    }
    if let Some(code) = last_status {
        return Json(TestResponse {
            ok: false,
            message: format!(
                "Server responded with HTTP {code}. Check that it is running Kadan Server."
            ),
            version: None,
            app: None,
            latency_ms: None,
        })
        .into_response();
    }
    Json(TestResponse {
        ok: false,
        message: format!(
            "Could not reach Kadan Server at {host}. Make sure Kadan Server is running and reachable."
        ),
        version: None,
        app: None,
        latency_ms: None,
    })
    .into_response()
}

fn is_websocket_upgrade(headers: &HeaderMap) -> bool {
    headers
        .get(axum::http::header::UPGRADE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.eq_ignore_ascii_case("websocket"))
        .unwrap_or(false)
}

fn build_upstream_url(base: &str, path_and_query: &str, ws: bool) -> String {
    let trimmed = base.trim_end_matches('/');
    let base_url = if ws {
        if let Some(rest) = trimmed.strip_prefix("https://") {
            format!("wss://{rest}")
        } else if let Some(rest) = trimmed.strip_prefix("http://") {
            format!("ws://{rest}")
        } else {
            format!("ws://{trimmed}")
        }
    } else {
        trimmed.to_string()
    };
    format!("{base_url}{path_and_query}")
}

#[derive(Serialize)]
struct PickFolderResponse {
    path: Option<String>,
    cancelled: bool,
    supported: bool,
}

fn run_native_folder_dialog() -> Option<Result<Option<String>, String>> {
    #[cfg(target_os = "linux")]
    {
        for (bin, args) in [
            (
                "zenity",
                &["--file-selection", "--directory", "--title=Choose project folder"][..],
            ),
            (
                "kdialog",
                &["--getexistingdirectory", ".", "--title", "Choose project folder"][..],
            ),
        ] {
            match std::process::Command::new(bin).args(args).output() {
                Ok(out) if out.status.success() => {
                    let chosen = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    return Some(Ok(if chosen.is_empty() { None } else { Some(chosen) }));
                }
                Ok(_) => return Some(Ok(None)),
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
                Err(err) => return Some(Err(err.to_string())),
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        match std::process::Command::new("osascript")
            .args([
                "-e",
                "POSIX path of (choose folder with prompt \"Choose project folder\")",
            ])
            .output()
        {
            Ok(out) if out.status.success() => {
                let chosen = String::from_utf8_lossy(&out.stdout)
                    .trim()
                    .trim_end_matches('/')
                    .to_string();
                Some(Ok(if chosen.is_empty() { None } else { Some(chosen) }))
            }
            Ok(_) => Some(Ok(None)),
            Err(err) => Some(Err(err.to_string())),
        }
    }
    #[cfg(target_os = "windows")]
    {
        let script = "Add-Type -AssemblyName System.Windows.Forms; \
            $d = New-Object System.Windows.Forms.FolderBrowserDialog; \
            $d.Description = 'Choose project folder'; \
            if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $d.SelectedPath }";
        match std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command", script])
            .output()
        {
            Ok(out) if out.status.success() => {
                let chosen = String::from_utf8_lossy(&out.stdout).trim().to_string();
                Some(Ok(if chosen.is_empty() { None } else { Some(chosen) }))
            }
            Ok(_) => Some(Ok(None)),
            Err(err) => Some(Err(err.to_string())),
        }
    }
}

async fn pick_folder() -> Response {
    let result = tokio::task::spawn_blocking(run_native_folder_dialog).await;
    match result {
        Ok(Some(Ok(Some(path)))) => Json(PickFolderResponse {
            path: Some(path),
            cancelled: false,
            supported: true,
        })
        .into_response(),
        Ok(Some(Ok(None))) => Json(PickFolderResponse {
            path: None,
            cancelled: true,
            supported: true,
        })
        .into_response(),
        Ok(None) => Json(PickFolderResponse {
            path: None,
            cancelled: false,
            supported: false,
        })
        .into_response(),
        Ok(Some(Err(detail))) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "detail": detail })),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "detail": err.to_string() })),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
struct SaveDownloadRequest {
    url: String,
    filename: Option<String>,
}

#[derive(Serialize)]
struct SaveDownloadResponse {
    saved: bool,
    cancelled: bool,
    path: Option<String>,
    filename: Option<String>,
    supported: bool,
}

fn is_valid_relative_download_url(url: &str) -> bool {
    url.starts_with('/')
        && !url.starts_with("//")
        && !url.contains('\r')
        && !url.contains('\n')
        && !url.contains('\0')
}

fn sanitize_download_filename(raw: Option<&str>, url: &str) -> String {
    let from_url = url
        .split(['?', '#'])
        .next()
        .unwrap_or("")
        .rsplit('/')
        .find(|segment| !segment.is_empty());
    let candidate = raw
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .or(from_url)
        .unwrap_or("download");
    let leaf = candidate
        .rsplit(['/', '\\'])
        .find(|segment| !segment.is_empty())
        .unwrap_or("download");
    let cleaned: String = leaf
        .chars()
        .filter(|ch| !ch.is_control() && !matches!(ch, '/' | '\\' | '\0'))
        .collect();
    let trimmed = cleaned.trim().trim_matches('.');
    if trimmed.is_empty() {
        "download".to_string()
    } else {
        trimmed.to_string()
    }
}

fn extract_content_disposition_filename(headers: &reqwest::header::HeaderMap) -> Option<String> {
    let value = headers
        .get(reqwest::header::CONTENT_DISPOSITION)?
        .to_str()
        .ok()?;
    for part in value.split(';').skip(1) {
        let trimmed = part.trim();
        let lower = trimmed.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("filename=") {
            let offset = trimmed.len() - rest.len();
            let raw = trimmed[offset..].trim().trim_matches('"');
            if !raw.is_empty() {
                return Some(raw.to_string());
            }
        }
    }
    None
}

fn default_download_dir() -> PathBuf {
    if let Some(dirs) = directories::UserDirs::new() {
        if let Some(downloads) = dirs.download_dir() {
            return downloads.to_path_buf();
        }
        return dirs.home_dir().to_path_buf();
    }
    std::env::temp_dir()
}

fn unique_destination_path(dir: &Path, filename: &str) -> PathBuf {
    let mut candidate = dir.join(filename);
    if !candidate.exists() {
        return candidate;
    }
    let path = Path::new(filename);
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(filename);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{e}"))
        .unwrap_or_default();
    let mut counter = 1u32;
    while candidate.exists() {
        candidate = dir.join(format!("{stem} ({counter}){ext}"));
        counter += 1;
    }
    candidate
}

fn run_native_save_dialog(default_path: PathBuf) -> Option<Result<Option<PathBuf>, String>> {
    #[cfg(target_os = "linux")]
    {
        let filename_arg = format!("--filename={}", default_path.to_string_lossy());
        let path_str = default_path.to_string_lossy().to_string();
        let candidates: [(&str, Vec<&str>); 2] = [
            (
                "zenity",
                vec![
                    "--file-selection",
                    "--save",
                    "--title=Save file",
                    &filename_arg,
                ],
            ),
            (
                "kdialog",
                vec!["--getsavefilename", &path_str, "--title", "Save file"],
            ),
        ];
        for (bin, args) in candidates {
            match std::process::Command::new(bin).args(args).output() {
                Ok(out) if out.status.success() => {
                    let chosen = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    return Some(Ok(if chosen.is_empty() {
                        None
                    } else {
                        Some(PathBuf::from(chosen))
                    }));
                }
                Ok(_) => return Some(Ok(None)),
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
                Err(err) => return Some(Err(err.to_string())),
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        let default_dir = default_path
            .parent()
            .unwrap_or_else(|| Path::new("/"))
            .to_string_lossy()
            .to_string();
        let default_name = default_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("download")
            .to_string();
        match std::process::Command::new("osascript")
            .args([
                "-e",
                "on run argv\nreturn POSIX path of (choose file name with prompt \"Save file\" default location (POSIX file (item 1 of argv)) default name (item 2 of argv))\nend run",
                "--",
                &default_dir,
                &default_name,
            ])
            .output()
        {
            Ok(out) if out.status.success() => {
                let chosen = String::from_utf8_lossy(&out.stdout).trim().to_string();
                Some(Ok(if chosen.is_empty() {
                    None
                } else {
                    Some(PathBuf::from(chosen))
                }))
            }
            Ok(_) => Some(Ok(None)),
            Err(err) => Some(Err(err.to_string())),
        }
    }
    #[cfg(target_os = "windows")]
    {
        let default_dir = default_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_string_lossy()
            .to_string();
        let default_name = default_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("download")
            .to_string();
        let script = "Add-Type -AssemblyName System.Windows.Forms; \
            $d = New-Object System.Windows.Forms.SaveFileDialog; \
            $d.Title = 'Save file'; \
            $d.InitialDirectory = $env:KADAN_SAVE_DIR; \
            $d.FileName = $env:KADAN_SAVE_NAME; \
            $d.OverwritePrompt = $true; \
            if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $d.FileName }";
        match std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command", script])
            .env("KADAN_SAVE_DIR", default_dir)
            .env("KADAN_SAVE_NAME", default_name)
            .output()
        {
            Ok(out) if out.status.success() => {
                let chosen = String::from_utf8_lossy(&out.stdout).trim().to_string();
                Some(Ok(if chosen.is_empty() {
                    None
                } else {
                    Some(PathBuf::from(chosen))
                }))
            }
            Ok(_) => Some(Ok(None)),
            Err(err) => Some(Err(err.to_string())),
        }
    }
}

async fn save_download(
    State(state): State<ProxyState>,
    Json(payload): Json<SaveDownloadRequest>,
) -> Response {
    if !is_valid_relative_download_url(&payload.url) {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({ "detail": "Invalid download address." })),
        )
            .into_response();
    }

    let active_host = state.upstream();
    let upstream = build_upstream_url(&active_host, &payload.url, false);
    let resp = match state.client.get(&upstream).send().await {
        Ok(resp) => resp,
        Err(_) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "detail": state.unreachable_detail(&active_host),
                    "code": "server_unreachable"
                })),
            )
                .into_response();
        }
    };

    if !resp.status().is_success() {
        let code = resp.status().as_u16();
        return (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({
                "detail": format!("Could not download file from Kadan Server (HTTP {code}).")
            })),
        )
            .into_response();
    }

    let header_name = extract_content_disposition_filename(resp.headers());
    let filename = sanitize_download_filename(
        payload.filename.as_deref().or(header_name.as_deref()),
        &payload.url,
    );
    let bytes = match resp.bytes().await {
        Ok(bytes) => bytes,
        Err(_) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "detail": "Download was interrupted. Try again."
                })),
            )
                .into_response();
        }
    };

    let default_dir = default_download_dir();
    let default_path = default_dir.join(&filename);
    let dialog_result =
        tokio::task::spawn_blocking(move || run_native_save_dialog(default_path)).await;

    let (destination, supported) = match dialog_result {
        Ok(Some(Ok(Some(chosen)))) => (chosen, true),
        Ok(Some(Ok(None))) => {
            return Json(SaveDownloadResponse {
                saved: false,
                cancelled: true,
                path: None,
                filename: None,
                supported: true,
            })
            .into_response();
        }
        Ok(None) => (unique_destination_path(&default_dir, &filename), false),
        Ok(Some(Err(detail))) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "detail": format!("Could not open save dialog: {detail}")
                })),
            )
                .into_response();
        }
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "detail": format!("Could not open save dialog: {err}")
                })),
            )
                .into_response();
        }
    };

    if let Some(parent) = destination.parent() {
        if let Err(err) = tokio::fs::create_dir_all(parent).await {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "detail": format!("Could not save file: {err}")
                })),
            )
                .into_response();
        }
    }

    if let Err(err) = tokio::fs::write(&destination, &bytes).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "detail": format!("Could not save file: {err}")
            })),
        )
            .into_response();
    }

    let saved_name = destination
        .file_name()
        .and_then(|n| n.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or(filename);
    Json(SaveDownloadResponse {
        saved: true,
        cancelled: false,
        path: Some(destination.to_string_lossy().to_string()),
        filename: Some(saved_name),
        supported,
    })
    .into_response()
}

async fn check_desktop_updates() -> Response {
    match updater::check_for_update().await {
        Ok(res) => Json(res).into_response(),
        Err(detail) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({ "detail": detail })),
        )
            .into_response(),
    }
}

async fn install_desktop_update(State(state): State<ProxyState>) -> Response {
    match updater::download_and_install(state.app_handle.clone()).await {
        Ok(res) => Json(res).into_response(),
        Err(detail) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({ "detail": detail })),
        )
            .into_response(),
    }
}

async fn proxy_local_work(State(state): State<ProxyState>, req: Request) -> Response {
    let active_host = state.upstream();
    let work_base = match state
        .work_sidecar
        .ensure_running(&state.client, &active_host)
        .await
    {
        Ok(base) => base,
        Err(detail) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "detail": detail,
                    "code": "local_work_unavailable"
                })),
            )
                .into_response();
        }
    };

    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or(req.uri().path());
    let local_path = path_and_query
        .strip_prefix("/kadan")
        .unwrap_or(path_and_query);
    let upstream = build_upstream_url(&work_base, local_path, false);

    let method = req.method().clone();
    let headers = req.headers().clone();
    let mut builder = state.client.request(method, &upstream);
    for (name, value) in &headers {
        if is_hop_by_hop(name) || name == axum::http::header::HOST {
            continue;
        }
        builder = builder.header(name, value);
    }
    let body_stream = req.into_body().into_data_stream();
    builder = builder.body(reqwest::Body::wrap_stream(body_stream));

    match builder.send().await {
        Ok(resp) => {
            let status = resp.status();
            let resp_headers = resp.headers().clone();
            let stream = resp.bytes_stream();
            let mut out = Response::builder().status(status);
            for (name, value) in &resp_headers {
                if is_hop_by_hop(name) {
                    continue;
                }
                if let (Ok(n), Ok(v)) = (
                    HeaderName::from_bytes(name.as_str().as_bytes()),
                    HeaderValue::from_bytes(value.as_bytes()),
                ) {
                    out = out.header(n, v);
                }
            }
            out.body(Body::from_stream(stream))
                .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
        }
        Err(_) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({
                "detail": "Local Work runtime stopped responding. Try refreshing.",
                "code": "local_work_unreachable"
            })),
        )
            .into_response(),
    }
}

async fn proxy_api(State(state): State<ProxyState>, req: Request) -> Response {
    let active_host = state.upstream();
    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or(req.uri().path())
        .to_string();

    if is_websocket_upgrade(req.headers()) {
        let (mut parts, _body) = req.into_parts();
        match WebSocketUpgrade::from_request_parts(&mut parts, &state).await {
            Ok(upgrade) => {
                let upstream = build_upstream_url(&active_host, &path_and_query, true);
                return upgrade.on_upgrade(move |socket| relay_websocket(socket, upstream));
            }
            Err(resp) => return resp.into_response(),
        }
    }

    let upstream = build_upstream_url(&active_host, &path_and_query, false);
    let method = req.method().clone();
    let headers = req.headers().clone();

    let mut builder = state.client.request(method, &upstream);
    for (name, value) in &headers {
        if is_hop_by_hop(name) || name == axum::http::header::HOST {
            continue;
        }
        builder = builder.header(name, value);
    }
    // Only a client of another computer is a separate app for that server to update.
    if state.mode() == RunMode::Connect {
        builder = builder
            .header("X-Kadan-Desktop-Version", updater::current_version())
            .header(
                "X-Kadan-Desktop-Can-Update",
                if updater::can_self_update_locally() { "true" } else { "false" },
            );
    }

    let body_stream = req.into_body().into_data_stream();
    builder = builder.body(reqwest::Body::wrap_stream(body_stream));

    match builder.send().await {
        Ok(resp) => {
            let status = resp.status();
            let resp_headers = resp.headers().clone();
            let stream = resp.bytes_stream();
            let mut out = Response::builder().status(status);
            for (name, value) in &resp_headers {
                if is_hop_by_hop(name) {
                    continue;
                }
                if let (Ok(n), Ok(v)) = (
                    HeaderName::from_bytes(name.as_str().as_bytes()),
                    HeaderValue::from_bytes(value.as_bytes()),
                ) {
                    out = out.header(n, v);
                }
            }
            out.body(Body::from_stream(stream))
                .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
        }
        Err(_) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({
                "detail": state.unreachable_detail(&active_host),
                "code": "server_unreachable"
            })),
        )
            .into_response(),
    }
}

#[derive(Serialize)]
struct HostStatusResponse {
    mode: RunMode,
    status: crate::supervisor::ServerStatusSnapshot,
    settings: ServerConfig,
    hardware: crate::hardware::HardwareSummary,
    updates: crate::host_updater::UpdateStatus,
}

fn host_error(status: StatusCode, detail: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({ "detail": detail.into() }))).into_response()
}

/// What this computer runs, for Kadan Server's settings when Kadan hosts here.
async fn host_status(State(state): State<ProxyState>) -> Json<HostStatusResponse> {
    Json(HostStatusResponse {
        mode: state.mode(),
        status: state.supervisor.snapshot(),
        settings: state.supervisor.config(),
        hardware: tokio::task::spawn_blocking(collect_hardware).await.unwrap_or_default(),
        updates: state.host_updater.status(),
    })
}

#[derive(Deserialize)]
struct HostSettingsRequest {
    settings: ServerConfig,
    #[serde(default)]
    restart: bool,
}

async fn put_host_settings(
    State(state): State<ProxyState>,
    Json(payload): Json<HostSettingsRequest>,
) -> Response {
    if let Err(detail) = save_host_config(&payload.settings) {
        return host_error(StatusCode::UNPROCESSABLE_ENTITY, detail);
    }
    if let Some(app) = &state.app_handle {
        use tauri_plugin_autostart::ManagerExt;
        let _ = if payload.settings.launch_at_login {
            app.autolaunch().enable()
        } else {
            app.autolaunch().disable()
        };
    }
    state
        .host_updater
        .set_data_dir(PathBuf::from(&payload.settings.data_dir));
    state.supervisor.update_config(payload.settings.clone());
    if payload.restart && state.mode() == RunMode::Host {
        if let Err(detail) = state.supervisor.start_all().await {
            return host_error(StatusCode::BAD_GATEWAY, detail);
        }
    }
    Json(payload.settings).into_response()
}

async fn host_start(State(state): State<ProxyState>) -> Response {
    match state.supervisor.start_all().await {
        Ok(()) => Json(state.supervisor.snapshot()).into_response(),
        Err(detail) => host_error(StatusCode::BAD_GATEWAY, detail),
    }
}

async fn host_stop(State(state): State<ProxyState>) -> Response {
    state.supervisor.stop_all().await;
    Json(state.supervisor.snapshot()).into_response()
}

#[derive(Deserialize)]
struct HostRestartRequest {
    service: String,
}

async fn host_restart(
    State(state): State<ProxyState>,
    Json(payload): Json<HostRestartRequest>,
) -> Response {
    let Some(id) = ServiceId::from_key(&payload.service) else {
        return host_error(StatusCode::UNPROCESSABLE_ENTITY, "Unknown service.");
    };
    match state.supervisor.restart_service(id).await {
        Ok(()) => Json(state.supervisor.snapshot()).into_response(),
        Err(detail) => host_error(StatusCode::BAD_GATEWAY, detail),
    }
}

#[derive(Deserialize)]
struct HostLogsQuery {
    service: Option<String>,
}

async fn host_logs(
    State(state): State<ProxyState>,
    axum::extract::Query(query): axum::extract::Query<HostLogsQuery>,
) -> Json<Vec<crate::supervisor::LogEntry>> {
    Json(state.supervisor.logs(query.service.as_deref()))
}

async fn relay_websocket(client_ws: WebSocket, upstream_url: String) {
    let Ok((upstream_ws, _)) = tokio_tungstenite::connect_async(&upstream_url).await else {
        return;
    };
    let (mut client_tx, mut client_rx) = client_ws.split();
    let (mut up_tx, mut up_rx) = upstream_ws.split();

    let c2u = async {
        while let Some(Ok(msg)) = client_rx.next().await {
            let mapped = match msg {
                AxumMsg::Text(t) => Some(TungMsg::Text(t.to_string().into())),
                AxumMsg::Binary(b) => Some(TungMsg::Binary(b)),
                AxumMsg::Ping(p) => Some(TungMsg::Ping(p)),
                AxumMsg::Pong(p) => Some(TungMsg::Pong(p)),
                AxumMsg::Close(_) => {
                    let _ = up_tx.send(TungMsg::Close(None)).await;
                    break;
                }
            };
            if let Some(m) = mapped {
                if up_tx.send(m).await.is_err() {
                    break;
                }
            }
        }
    };

    let u2c = async {
        while let Some(Ok(msg)) = up_rx.next().await {
            let mapped = match msg {
                TungMsg::Text(t) => Some(AxumMsg::Text(t.to_string().into())),
                TungMsg::Binary(b) => Some(AxumMsg::Binary(b)),
                TungMsg::Ping(p) => Some(AxumMsg::Ping(p)),
                TungMsg::Pong(p) => Some(AxumMsg::Pong(p)),
                TungMsg::Close(_) => {
                    let _ = client_tx.send(AxumMsg::Close(None)).await;
                    break;
                }
                TungMsg::Frame(_) => None,
            };
            if let Some(m) = mapped {
                if client_tx.send(m).await.is_err() {
                    break;
                }
            }
        }
    };

    tokio::select! {
        _ = c2u => {}
        _ = u2c => {}
    }
}

fn is_hop_by_hop(name: &HeaderName) -> bool {
    matches!(
        name.as_str().to_ascii_lowercase().as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailers"
            | "transfer-encoding"
            | "upgrade"
    )
}

fn safe_join(root: &Path, req_path: &str) -> Option<PathBuf> {
    let rel = req_path.trim_start_matches('/');
    if rel.is_empty() {
        return None;
    }
    let candidate = root.join(rel);
    let canon_root = root.canonicalize().ok()?;
    let canon_cand = candidate.canonicalize().ok()?;
    if canon_cand.starts_with(&canon_root) && canon_cand.is_file() {
        Some(canon_cand)
    } else {
        None
    }
}

fn inject_desktop_flag(html: String) -> String {
    let snippet = "<script>window.__KADAN_DESKTOP__ = true;</script>";
    if let Some(idx) = html.find("<head>") {
        let mut out = String::with_capacity(html.len() + snippet.len());
        out.push_str(&html[..idx + 6]);
        out.push_str(snippet);
        out.push_str(&html[idx + 6..]);
        out
    } else {
        format!("{snippet}{html}")
    }
}

async fn serve_frontend(State(state): State<ProxyState>, req: Request) -> Response {
    let path = req.uri().path();
    if let Some(file_path) = safe_join(&state.web_root, path) {
        if let Ok(bytes) = tokio::fs::read(&file_path).await {
            let mime = mime_guess::from_path(&file_path).first_or_octet_stream();
            return (
                [(axum::http::header::CONTENT_TYPE, mime.as_ref().to_string())],
                bytes,
            )
                .into_response();
        }
    }

    if path.starts_with("/assets/") {
        return StatusCode::NOT_FOUND.into_response();
    }

    let index_path = state.web_root.join("index.html");
    match tokio::fs::read_to_string(&index_path).await {
        Ok(html) => Html(inject_desktop_flag(html)).into_response(),
        Err(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Html("<h1>Kadan frontend assets not found</h1>".to_string()),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_http_and_https_to_ws_and_wss() {
        assert_eq!(
            build_upstream_url("http://192.168.1.50:8630", "/api/agents/a/calls", true),
            "ws://192.168.1.50:8630/api/agents/a/calls"
        );
        assert_eq!(
            build_upstream_url("https://kadan.example/", "/api/agents/a/calls", true),
            "wss://kadan.example/api/agents/a/calls"
        );
    }

    #[test]
    fn injects_desktop_flag_into_head() {
        let html = "<!doctype html><html><head><title>Kadan</title></head><body></body></html>";
        let out = inject_desktop_flag(html.to_string());
        assert!(out.contains("<head><script>window.__KADAN_DESKTOP__ = true;</script><title>"));
    }

    #[test]
    fn validates_relative_download_urls_and_sanitizes_filenames() {
        assert!(is_valid_relative_download_url(
            "/kadan/api/group-chat/chatgpt-plugin.zip"
        ));
        assert!(!is_valid_relative_download_url("https://example.com/a.zip"));
        assert!(!is_valid_relative_download_url("//example.com/a.zip"));
        assert!(!is_valid_relative_download_url("/kadan\r\n/evil"));

        assert_eq!(
            sanitize_download_filename(
                Some("../evil/kadan-group-chat-plugin.zip"),
                "/kadan/api/group-chat/chatgpt-plugin.zip"
            ),
            "kadan-group-chat-plugin.zip"
        );
        assert_eq!(
            sanitize_download_filename(None, "/kadan/api/group-chat/chatgpt-plugin.zip?v=1"),
            "chatgpt-plugin.zip"
        );
        assert_eq!(sanitize_download_filename(Some("..."), "/"), "download");
    }

    #[test]
    fn unique_destination_path_avoids_overwriting_existing_files() {
        let tmp = std::env::temp_dir().join(format!(
            "kadan-desktop-unique-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let first = unique_destination_path(&tmp, "kadan-group-chat-plugin.zip");
        assert_eq!(first, tmp.join("kadan-group-chat-plugin.zip"));
        std::fs::write(&first, b"one").unwrap();

        let second = unique_destination_path(&tmp, "kadan-group-chat-plugin.zip");
        assert_eq!(second, tmp.join("kadan-group-chat-plugin (1).zip"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn proxy_serves_config_and_relays_upstream() {
        let upstream_app = Router::new().route(
            "/api/health",
            get(|| async { Json(serde_json::json!({"status": "ok", "version": "0.1.0", "app": "kadan-server"})) }),
        );
        let upstream_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream_addr = upstream_listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(upstream_listener, upstream_app).await;
        });

        let tmp = std::env::temp_dir().join(format!("kadan-desktop-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        std::fs::write(
            tmp.join("index.html"),
            "<html><head><title>Test</title></head><body>ok</body></html>",
        )
        .unwrap();

        let work_app = Router::new().route(
            "/api/work/harnesses",
            get(|| async { Json(serde_json::json!([{"id": "local-codex"}])) }),
        );
        let work_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let work_addr = work_listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(work_listener, work_app).await;
        });

        let cfg = DesktopConfig {
            mode: Some(RunMode::Connect),
            active_host: format!("http://{upstream_addr}"),
            saved_servers: vec![],
        };
        let (proxy_addr, _) = start_loopback_proxy(
            cfg,
            tmp.clone(),
            None,
            WorkSidecar::fixed_url(format!("http://{work_addr}")),
            Arc::new(ServerSupervisor::new(ServerConfig::default(), None, None)),
            Arc::new(AppUpdater::new(tmp.clone())),
        )
        .await
        .unwrap();
        let client = reqwest::Client::new();

        let harnesses: serde_json::Value = client
            .get(format!("http://{proxy_addr}/kadan/api/work/harnesses"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(harnesses[0]["id"], "local-codex");

        let html = client
            .get(format!("http://{proxy_addr}/"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(html.contains("window.__KADAN_DESKTOP__ = true"));

        let health: serde_json::Value = client
            .get(format!("http://{proxy_addr}/api/health"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(health["status"], "ok");

        let tested: serde_json::Value = client
            .post(format!("http://{proxy_addr}/__kadan_desktop__/test"))
            .json(&serde_json::json!({ "host": format!("http://{upstream_addr}") }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(tested["ok"], true);
        assert_eq!(tested["version"], "0.1.0");

        let host: serde_json::Value = client
            .get(format!("http://{proxy_addr}/__kadan_desktop__/host"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(host["mode"], "connect");
        assert_eq!(host["status"]["running"], false);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn hosting_sends_requests_to_this_computers_services() {
        let upstream_app = Router::new().route(
            "/api/health",
            get(|headers: HeaderMap| async move {
                Json(serde_json::json!({
                    "status": "ok",
                    "desktop_header": headers.contains_key("x-kadan-desktop-version"),
                }))
            }),
        );
        let upstream_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream_addr = upstream_listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(upstream_listener, upstream_app).await;
        });
        let tmp = std::env::temp_dir().join(format!("kadan-host-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);

        let mut host = ServerConfig::default();
        host.server_port = upstream_addr.port();
        let cfg = DesktopConfig {
            mode: Some(RunMode::Host),
            active_host: "http://192.0.2.1:9".to_string(),
            saved_servers: vec![],
        };
        let (proxy_addr, _) = start_loopback_proxy(
            cfg,
            tmp.clone(),
            None,
            WorkSidecar::fixed_url("http://127.0.0.1:9".to_string()),
            Arc::new(ServerSupervisor::new(host, None, None)),
            Arc::new(AppUpdater::new(tmp.clone())),
        )
        .await
        .unwrap();

        let health: serde_json::Value = reqwest::Client::new()
            .get(format!("http://{proxy_addr}/api/health"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(health["status"], "ok");
        assert_eq!(health["desktop_header"], false);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
