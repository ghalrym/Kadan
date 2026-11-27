mod config;
mod proxy;
mod updater;

use std::path::PathBuf;
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

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
    tauri::Builder::default()
        .setup(|app| {
            let handle = app.handle().clone();
            let cfg = config::load_config();
            let web_root = resolve_web_root(&handle);

            let (addr, _shared) = tauri::async_runtime::block_on(async move {
                proxy::start_loopback_proxy(cfg, web_root, Some(handle)).await
            })
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

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
        .run(tauri::generate_context!())
        .expect("error while running Kadan Desktop");
}
