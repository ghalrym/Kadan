#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    kadan_server_lib::run();
}
