mod config;
mod control;
mod hardware;
mod supervisor;
mod tray;
mod updater;

use config::{load_config, save_config, ServerConfig};
use hardware::{collect_hardware, HardwareSummary};
use std::path::PathBuf;
use std::sync::Arc;
use supervisor::{LogEntry, ServerStatusSnapshot, ServerSupervisor, ServiceId};
use tauri::{Manager, State, WindowEvent};
use tauri_plugin_autostart::ManagerExt;
use updater::{AppUpdater, ApplyResult, UpdateCheckResult, UpdateStatus};

struct AppState {
    supervisor: Arc<ServerSupervisor>,
    updater: Arc<AppUpdater>,
}

#[tauri::command]
fn get_config() -> ServerConfig {
    load_config()
}

#[tauri::command]
async fn update_config(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    config: ServerConfig,
    restart: bool,
) -> Result<ServerConfig, String> {
    save_config(&config)?;
    let _ = if config.launch_at_login {
        app.autolaunch().enable()
    } else {
        app.autolaunch().disable()
    };
    state.updater.set_data_dir(PathBuf::from(&config.data_dir));
    state.supervisor.update_config(config.clone());
    if restart {
        state.supervisor.start_all().await?;
    }
    Ok(config)
}

#[tauri::command]
fn get_status(state: State<'_, AppState>) -> ServerStatusSnapshot {
    state.supervisor.snapshot()
}

#[tauri::command]
fn get_hardware() -> HardwareSummary {
    collect_hardware()
}

#[tauri::command]
fn get_logs(state: State<'_, AppState>, service: Option<String>) -> Vec<LogEntry> {
    state.supervisor.logs(service.as_deref())
}

#[tauri::command]
async fn start_server(state: State<'_, AppState>) -> Result<ServerStatusSnapshot, String> {
    state.supervisor.start_all().await?;
    Ok(state.supervisor.snapshot())
}

#[tauri::command]
async fn stop_server(state: State<'_, AppState>) -> Result<ServerStatusSnapshot, String> {
    state.supervisor.stop_all().await;
    Ok(state.supervisor.snapshot())
}

#[tauri::command]
async fn restart_service(
    state: State<'_, AppState>,
    service: String,
) -> Result<ServerStatusSnapshot, String> {
    let id = ServiceId::from_key(&service).ok_or_else(|| format!("Unknown service: {service}"))?;
    state.supervisor.restart_service(id).await?;
    Ok(state.supervisor.snapshot())
}

#[tauri::command]
async fn check_updates(state: State<'_, AppState>) -> Result<UpdateCheckResult, String> {
    state.updater.check_for_update().await
}

#[tauri::command]
fn get_update_status(state: State<'_, AppState>) -> UpdateStatus {
    state.updater.status()
}

#[tauri::command]
async fn install_update(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<ApplyResult, String> {
    let staged = state.updater.stage_update(None).await?;
    state.updater.apply_update(
        state.supervisor.clone(),
        Some(app),
        Some(staged.target_version),
        true,
    )
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ))
        .setup(|app| {
            let cfg = load_config();
            let auto_start = cfg.auto_start_server;
            let data_dir = PathBuf::from(&cfg.data_dir);
            let resource_dir = app.path().resource_dir().ok();
            let updater = Arc::new(AppUpdater::new(data_dir));
            let supervisor = Arc::new(ServerSupervisor::new(cfg, resource_dir, None));
            let handle = app.handle().clone();

            let control_endpoint = tauri::async_runtime::block_on(control::start_control_server(
                updater.clone(),
                supervisor.clone(),
                Some(handle.clone()),
            ))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            supervisor.set_control_endpoint(control_endpoint);

            tray::setup_tray(&handle, supervisor.clone())?;

            if auto_start {
                let sup = supervisor.clone();
                tauri::async_runtime::spawn(async move {
                    let _ = sup.start_all().await;
                });
            }

            app.manage(AppState {
                supervisor,
                updater,
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .invoke_handler(tauri::generate_handler![
            get_config,
            update_config,
            get_status,
            get_hardware,
            get_logs,
            start_server,
            stop_server,
            restart_service,
            check_updates,
            get_update_status,
            install_update
        ])
        .run(tauri::generate_context!())
        .expect("error while running Kadan Server");
}
