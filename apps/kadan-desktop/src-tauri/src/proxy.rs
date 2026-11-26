use crate::config::{normalize_host, save_config, DesktopConfig};
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
    };

    let app = Router::new()
        .route("/__kadan_desktop__/config", get(get_config).put(put_config))
        .route("/__kadan_desktop__/test", post(test_connection))
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
    match save_config(incoming) {
        Ok(saved) => {
            *state.config.write().unwrap() = saved.clone();
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

async fn proxy_api(State(state): State<ProxyState>, req: Request) -> Response {
    let active_host = state.config.read().unwrap().active_host.clone();
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
                "detail": format!("Cannot reach Kadan Server at {active_host}. Open Server settings to check the host address."),
                "code": "server_unreachable"
            })),
        )
            .into_response(),
    }
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

        let cfg = DesktopConfig {
            active_host: format!("http://{upstream_addr}"),
            saved_servers: vec![],
        };
        let (proxy_addr, _) = start_loopback_proxy(cfg, tmp.clone()).await.unwrap();
        let client = reqwest::Client::new();

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

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
