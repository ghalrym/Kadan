#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    #[cfg(target_os = "linux")]
    {
        // Avoid Wayland Error 71 (Protocol error) on NVIDIA drivers without
        // disabling WebKitGTK's DMA-BUF renderer (which triggers a WebKitGTK 2.52
        // DrawingAreaProxy segfault on interaction).
        if std::env::var_os("__NV_DISABLE_EXPLICIT_SYNC").is_none() {
            unsafe {
                std::env::set_var("__NV_DISABLE_EXPLICIT_SYNC", "1");
            }
        }
    }
    kadan_server_lib::run();
}
