use crate::supervisor::ServerSupervisor;
use std::sync::Arc;
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Manager};

/// The tray keeps Kadan reachable while it hosts with its window closed.
pub fn setup_tray(app: &AppHandle, supervisor: Arc<ServerSupervisor>) -> tauri::Result<()> {
    let open = MenuItem::with_id(app, "open", "Open Kadan", true, None::<&str>)?;
    let sep = PredefinedMenuItem::separator(app)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Kadan", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&open, &sep, &quit])?;

    let mut builder = TrayIconBuilder::new()
        .menu(&menu)
        .tooltip("Kadan")
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                show_main_window(tray.app_handle());
            }
        })
        .on_menu_event(move |app_handle, event| match event.id.as_ref() {
            "open" => show_main_window(app_handle),
            "quit" => {
                let sup = supervisor.clone();
                let handle = app_handle.clone();
                tauri::async_runtime::spawn(async move {
                    sup.stop_all().await;
                    handle.exit(0);
                });
            }
            _ => {}
        });

    if let Some(img) = app.default_window_icon().cloned() {
        builder = builder.icon(img);
    }
    builder.build(app)?;
    Ok(())
}

pub fn show_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}
