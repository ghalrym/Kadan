mod config;
mod proxy;
mod updater;
mod work_sidecar;

use std::path::PathBuf;
use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};
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

pub fn run() {
    let mut sidecar_for_exit: Option<WorkSidecar> = None;

    let app = tauri::Builder::default()
        .setup(|app| {
            let handle = app.handle().clone();
            let cfg = config::load_config();
            let web_root = resolve_web_root(&handle);
            let resource_dir = handle.path().resource_dir().ok();
            let work_sidecar = WorkSidecar::new(resource_dir);
            let sidecar_warmup = work_sidecar.clone();
            let initial_host = cfg.active_host.clone();
            app.manage(work_sidecar.clone());

            let (addr, _shared) = tauri::async_runtime::block_on(async move {
                proxy::start_loopback_proxy(cfg, web_root, Some(handle), work_sidecar).await
            })
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

            // Warm up the local Work runtime in the background so WorkRunner starts
            // immediately for local harnesses and agent-launched coding threads.
            tauri::async_runtime::spawn(async move {
                if let Ok(client) = reqwest::Client::builder()
                    .redirect(reqwest::redirect::Policy::none())
                    .build()
                {
                    let _ = sidecar_warmup.ensure_running(&client, &initial_host).await;
                }
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
        .build(tauri::generate_context!())
        .expect("error while building Kadan Desktop");

    if let Some(sc) = app.try_state::<WorkSidecar>() {
        sidecar_for_exit = Some(sc.inner().clone());
    }

    app.run(move |_app_handle, event| {
        if let RunEvent::Exit = event {
            if let Some(sc) = &sidecar_for_exit {
                let sc = sc.clone();
                tauri::async_runtime::block_on(async move {
                    sc.shutdown().await;
                });
            }
        }
    });
}
