use crate::supervisor::ServerSupervisor;
use crate::host_updater::AppUpdater;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use std::sync::Arc;
use tokio::net::TcpListener;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct ControlEndpoint {
    pub url: String,
    pub token: String,
}

#[derive(Clone)]
struct ControlState {
    token: String,
    updater: Arc<AppUpdater>,
    supervisor: Arc<ServerSupervisor>,
    app_handle: Option<tauri::AppHandle>,
}

#[derive(Deserialize)]
struct DesktopHeartbeatPayload {
    version: String,
    #[serde(default)]
    can_self_update: bool,
}

#[derive(Deserialize)]
struct DesktopAckPayload {
    request_id: String,
}

#[derive(Deserialize, Default)]
struct StagePayload {
    #[serde(default)]
    target_version: Option<String>,
}

#[derive(Deserialize, Default)]
struct ApplyPayload {
    #[serde(default)]
    target_version: Option<String>,
    #[serde(default = "default_true")]
    include_desktop: bool,
}

fn default_true() -> bool {
    true
}

fn authorize(headers: &HeaderMap, expected_token: &str) -> Result<(), Response> {
    let provided = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    if provided == Some(expected_token) {
        Ok(())
    } else {
        Err((
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "detail": "Unauthorized control request." })),
        )
            .into_response())
    }
}

pub async fn start_control_server(
    updater: Arc<AppUpdater>,
    supervisor: Arc<ServerSupervisor>,
    app_handle: Option<tauri::AppHandle>,
) -> Result<ControlEndpoint, String> {
    let token = Uuid::new_v4().to_string();
    let state = ControlState {
        token: token.clone(),
        updater,
        supervisor,
        app_handle,
    };

    let app = Router::new()
        .route("/updates", get(check_updates))
        .route("/updates/status", get(update_status))
        .route("/updates/desktop/heartbeat", post(desktop_heartbeat))
        .route("/updates/desktop/ack", post(desktop_ack))
        .route("/updates/stage", post(stage_update))
        .route("/updates/apply", post(apply_update))
        .with_state(state);

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| format!("Could not bind local control server: {e}"))?;
    let addr = listener
        .local_addr()
        .map_err(|e| format!("Could not read local control address: {e}"))?;

    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    Ok(ControlEndpoint {
        url: format!("http://{addr}"),
        token,
    })
}

async fn check_updates(State(state): State<ControlState>, headers: HeaderMap) -> Response {
    if let Err(resp) = authorize(&headers, &state.token) {
        return resp;
    }
    match state.updater.check_for_update().await {
        Ok(result) => Json(result).into_response(),
        Err(detail) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({ "detail": detail })),
        )
            .into_response(),
    }
}

async fn update_status(State(state): State<ControlState>, headers: HeaderMap) -> Response {
    if let Err(resp) = authorize(&headers, &state.token) {
        return resp;
    }
    Json(state.updater.status()).into_response()
}

async fn desktop_heartbeat(
    State(state): State<ControlState>,
    headers: HeaderMap,
    Json(payload): Json<DesktopHeartbeatPayload>,
) -> Response {
    if let Err(resp) = authorize(&headers, &state.token) {
        return resp;
    }
    state
        .updater
        .record_desktop_heartbeat(payload.version, payload.can_self_update);
    Json(state.updater.status()).into_response()
}

async fn desktop_ack(
    State(state): State<ControlState>,
    headers: HeaderMap,
    Json(payload): Json<DesktopAckPayload>,
) -> Response {
    if let Err(resp) = authorize(&headers, &state.token) {
        return resp;
    }
    let cleared = state.updater.ack_desktop_update(&payload.request_id);
    Json(serde_json::json!({ "ok": true, "acknowledged": cleared })).into_response()
}

async fn stage_update(
    State(state): State<ControlState>,
    headers: HeaderMap,
    payload: Option<Json<StagePayload>>,
) -> Response {
    if let Err(resp) = authorize(&headers, &state.token) {
        return resp;
    }
    let target = payload.and_then(|p| p.0.target_version);
    match state.updater.stage_update(target).await {
        Ok(res) => Json(res).into_response(),
        Err(detail) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({ "detail": detail })),
        )
            .into_response(),
    }
}

async fn apply_update(
    State(state): State<ControlState>,
    headers: HeaderMap,
    payload: Option<Json<ApplyPayload>>,
) -> Response {
    if let Err(resp) = authorize(&headers, &state.token) {
        return resp;
    }
    let body = payload.map(|p| p.0).unwrap_or_default();
    match state.updater.apply_update(
        state.supervisor.clone(),
        state.app_handle.clone(),
        body.target_version,
        body.include_desktop,
    ) {
        Ok(res) => Json(res).into_response(),
        Err(detail) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({ "detail": detail })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host_config::ServerConfig;

    #[tokio::test]
    async fn control_server_requires_bearer_token_and_tracks_desktop_state() {
        let updater = Arc::new(AppUpdater::new(std::env::temp_dir()));
        let supervisor = Arc::new(ServerSupervisor::new(ServerConfig::default(), None, None));
        let endpoint = start_control_server(updater, supervisor, None).await.unwrap();
        let client = reqwest::Client::new();

        // Rejects missing auth.
        let unauth = client
            .get(format!("{}/updates/status", endpoint.url))
            .send()
            .await
            .unwrap();
        assert_eq!(unauth.status(), reqwest::StatusCode::UNAUTHORIZED);

        // Accepts bearer auth and records desktop heartbeat.
        let hb = client
            .post(format!("{}/updates/desktop/heartbeat", endpoint.url))
            .bearer_auth(&endpoint.token)
            .json(&serde_json::json!({ "version": "0.1.0", "can_self_update": true }))
            .send()
            .await
            .unwrap();
        assert_eq!(hb.status(), reqwest::StatusCode::OK);
        let body: serde_json::Value = hb.json().await.unwrap();
        assert_eq!(body["desktop"]["connected"], true);
        assert_eq!(body["desktop"]["current_version"], "0.1.0");
    }
}
