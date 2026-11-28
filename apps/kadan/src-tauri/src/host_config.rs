use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub bind_all_interfaces: bool,
    pub server_port: u16,
    pub worker_token: String,
    pub data_dir: String,
    pub auto_start_server: bool,
    pub launch_at_login: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_all_interfaces: true,
            server_port: 8000,
            worker_token: "insecure-api-key".to_string(),
            data_dir: default_data_dir().to_string_lossy().to_string(),
            auto_start_server: true,
            launch_at_login: false,
        }
    }
}

pub fn default_data_dir() -> PathBuf {
    if let Some(dirs) = ProjectDirs::from("ai", "Kadan", "KadanServer") {
        return dirs.data_local_dir().to_path_buf();
    }
    PathBuf::from("kadan-server-data")
}

pub fn config_path() -> PathBuf {
    if let Some(dirs) = ProjectDirs::from("ai", "Kadan", "KadanServer") {
        return dirs.config_dir().join("server-config.json");
    }
    PathBuf::from("server-config.json")
}

pub fn validate_config(cfg: &ServerConfig) -> Result<(), String> {
    if cfg.server_port == 0 {
        return Err("Server port must be between 1 and 65535.".to_string());
    }
    if cfg.data_dir.trim().is_empty() {
        return Err("Choose a folder to store Kadan's models and data.".to_string());
    }
    if cfg.worker_token.trim().is_empty() {
        return Err("Worker token cannot be empty.".to_string());
    }
    Ok(())
}

pub fn load_config() -> ServerConfig {
    let path = config_path();
    let Ok(raw) = fs::read_to_string(&path) else {
        return ServerConfig::default();
    };
    serde_json::from_str::<ServerConfig>(&raw).unwrap_or_default()
}

pub fn save_config(cfg: &ServerConfig) -> Result<(), String> {
    validate_config(cfg)?;
    let path = config_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Could not create settings folder: {e}"))?;
    }
    let json =
        serde_json::to_string_pretty(cfg).map_err(|e| format!("Config error: {e}"))?;
    fs::write(&path, json).map_err(|e| format!("Could not write settings: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let cfg = ServerConfig::default();
        assert!(validate_config(&cfg).is_ok());
        assert_eq!(cfg.server_port, 8000);
    }

    #[test]
    fn rejects_zero_port() {
        let mut cfg = ServerConfig::default();
        cfg.server_port = 0;
        assert!(validate_config(&cfg).is_err());
    }

    #[test]
    fn ignores_legacy_workload_fields() {
        let legacy = r#"{"server_port": 9000, "enable_local_worker": false, "workloads": ["llm"]}"#;
        let parsed: ServerConfig = serde_json::from_str(legacy).expect("parse legacy config");
        assert_eq!(parsed.server_port, 9000);
    }
}
