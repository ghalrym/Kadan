mod config;
mod control;
mod hardware;
mod host_config;
mod host_updater;
mod proxy;
mod supervisor;
mod tray;
mod updater;
mod work_sidecar;

use config::RunMode;
use host_updater::AppUpdater;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use supervisor::ServerSupervisor;
use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};
use work_sidecar::WorkSidecar;

fn resolve_web_root(app: &tauri::AppHandle) -> PathBuf {
    if let Ok(res_dir) = app.path().resource_dir() {
        let bundled = res_dir.join("web");
        if bundled.join("index.html").is_file() && bundled.join("assets").is_dir() {
            return bundled;
        }
    }
    let dev_dist = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../services/frontend/dist");
    if dev_dist.join("index.html").is_file() {
        return dev_dist;
    }
    PathBuf::from("web")
}

/// The app-wide pieces the window, tray and exit handler share.
struct AppShared {
    config: Arc<RwLock<config::DesktopConfig>>,
    supervisor: Arc<ServerSupervisor>,
    work_sidecar: WorkSidecar,
}

pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ))
        .setup(|app| {
            let handle = app.handle().clone();
            let cfg = config::load_config();
            let host_cfg = host_config::load_config();
            let web_root = resolve_web_root(&handle);
            let resource_dir = handle.path().resource_dir().ok();

            // Kadan's own services: started here when this computer hosts them.
            let host_updater = Arc::new(AppUpdater::new(PathBuf::from(&host_cfg.data_dir)));
            let supervisor = Arc::new(ServerSupervisor::new(
                host_cfg.clone(),
                resource_dir.clone(),
                None,
            ));
            let control_endpoint = tauri::async_runtime::block_on(control::start_control_server(
                host_updater.clone(),
                supervisor.clone(),
                Some(handle.clone()),
            ))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            supervisor.set_control_endpoint(control_endpoint);

            let hosting = cfg.run_mode() == RunMode::Host;
            if hosting && host_cfg.auto_start_server {
                let sup = supervisor.clone();
                tauri::async_runtime::spawn(async move {
                    let _ = sup.start_all().await;
                });
            }

            let work_sidecar = WorkSidecar::new(resource_dir);
            let sidecar_warmup = work_sidecar.clone();
            let initial_upstream = cfg.upstream(host_cfg.server_port);
            let (addr, shared_config) = tauri::async_runtime::block_on(proxy::start_loopback_proxy(
                cfg,
                web_root,
                Some(handle.clone()),
                work_sidecar.clone(),
                supervisor.clone(),
                host_updater,
            ))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

            // Warm up the local Work runtime in the background so WorkRunner starts
            // immediately for local harnesses and agent-launched coding threads.
            tauri::async_runtime::spawn(async move {
                if let Ok(client) = reqwest::Client::builder()
                    .redirect(reqwest::redirect::Policy::none())
                    .build()
                {
                    let _ = sidecar_warmup.ensure_running(&client, &initial_upstream).await;
                }
            });

            tray::setup_tray(&handle, supervisor.clone())?;
            app.manage(AppShared {
                config: shared_config,
                supervisor,
                work_sidecar,
            });

            let url: url::Url = format!("http://{addr}/")
                .parse()
                .expect("valid loopback URL");
            WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url))
                .title("Kadan")
                .inner_size(1200.0, 820.0)
                .min_inner_size(820.0, 580.0)
                .resizable(true)
                .build()?;

            Ok(())
        })
        .on_window_event(|window, event| {
            // While this computer hosts Kadan, closing the window keeps it running in the tray.
            if let WindowEvent::CloseRequested { api, .. } = event {
                let hosting = window
                    .try_state::<AppShared>()
                    .map(|shared| shared.config.read().unwrap().run_mode() == RunMode::Host)
                    .unwrap_or(false);
                if hosting {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building Kadan");

    app.run(move |app_handle, event| {
        if let RunEvent::Exit = event {
            if let Some(shared) = app_handle.try_state::<AppShared>() {
                let supervisor = shared.supervisor.clone();
                let sidecar = shared.work_sidecar.clone();
                tauri::async_runtime::block_on(async move {
                    sidecar.shutdown().await;
                    supervisor.stop_all().await;
                });
            }
        }
    });
}
