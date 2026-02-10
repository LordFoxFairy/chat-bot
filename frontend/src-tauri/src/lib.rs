use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::{
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Manager, WindowEvent,
};

struct PythonProcess(Mutex<Option<Child>>);

#[tauri::command]
fn start_python_server(state: tauri::State<PythonProcess>) -> Result<String, String> {
    let mut process = state.0.lock().map_err(|e| e.to_string())?;

    if process.is_some() {
        return Ok("Python server already running".to_string());
    }

    // In development, start from project root
    #[cfg(debug_assertions)]
    let child = Command::new("python3")
        .arg("app.py")
        .current_dir("../..")
        .spawn()
        .map_err(|e| format!("Failed to start Python server: {}", e))?;

    // In production, use bundled binary
    #[cfg(not(debug_assertions))]
    let child = Command::new("./binaries/python-server")
        .spawn()
        .map_err(|e| format!("Failed to start Python server: {}", e))?;

    *process = Some(child);
    Ok("Python server started".to_string())
}

#[tauri::command]
fn stop_python_server(state: tauri::State<PythonProcess>) -> Result<String, String> {
    let mut process = state.0.lock().map_err(|e| e.to_string())?;

    if let Some(ref mut child) = *process {
        child.kill().map_err(|e| format!("Failed to kill Python server: {}", e))?;
        *process = None;
        Ok("Python server stopped".to_string())
    } else {
        Ok("Python server not running".to_string())
    }
}

#[tauri::command]
fn check_python_server(state: tauri::State<PythonProcess>) -> Result<bool, String> {
    let mut process = state.0.lock().map_err(|e| e.to_string())?;

    if let Some(ref mut child) = *process {
        match child.try_wait() {
            Ok(Some(_)) => {
                // Process has exited
                *process = None;
                Ok(false)
            }
            Ok(None) => Ok(true), // Still running
            Err(e) => Err(format!("Failed to check process: {}", e)),
        }
    } else {
        Ok(false)
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_process::init())
        .manage(PythonProcess(Mutex::new(None)))
        .setup(|app| {
            // Create system tray
            let _tray = TrayIconBuilder::new()
                .tooltip("Chat Bot")
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        let app = tray.app_handle();
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                })
                .build(app)?;

            // Auto-start Python server in development
            #[cfg(debug_assertions)]
            {
                let state = app.state::<PythonProcess>();
                let _ = start_python_server(state);
            }

            Ok(())
        })
        .on_window_event(|window, event| {
            // Hide window instead of closing when clicking the close button
            if let WindowEvent::CloseRequested { api, .. } = event {
                let _ = window.hide();
                api.prevent_close();
            }
        })
        .invoke_handler(tauri::generate_handler![
            start_python_server,
            stop_python_server,
            check_python_server
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
