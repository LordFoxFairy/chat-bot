#![cfg_attr(
  all(not(debug_assertions), target_os = "windows"),
  windows_subsystem = "windows"
)]

use std::process::Command;
use tauri::{
    CustomMenuItem, Manager, SystemTray, SystemTrayEvent, SystemTrayMenu, SystemTrayMenuItem,
};

fn main() {
  tauri::Builder::default()
    .setup(|app| {
      let window = app.get_window("main").unwrap();

      // Start python backend process
      // In production this would launch the bundled binary
      // In development we might want to just rely on running the python script separately
      // or using a shell command

      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
