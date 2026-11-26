use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use url::Url;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SavedServer {
    pub id: String,
    pub label: String,
    pub host: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DesktopConfig {
    pub active_host: String,
    #[serde(default)]
    pub saved_servers: Vec<SavedServer>,
}

impl Default for DesktopConfig {
    fn default() -> Self {
        Self {
            active_host: "http://127.0.0.1:8000".to_string(),
            saved_servers: vec![SavedServer {
                id: "local".to_string(),
                label: "This computer".to_string(),
                host: "http://127.0.0.1:8000".to_string(),
            }],
        }
    }
}

pub fn normalize_host(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("Enter a server address such as http://127.0.0.1:8630.".to_string());
    }
    let with_scheme = if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    };
    let parsed = Url::parse(&with_scheme)
        .map_err(|_| "Enter a valid http:// or https:// server address.".to_string())?;
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        return Err("Only http:// and https:// server addresses are supported.".to_string());
    }
    if parsed.host_str().is_none() {
        return Err("Server address must include a host name or IP address.".to_string());
    }
    Ok(parsed.as_str().trim_end_matches('/').to_string())
}

pub fn config_path() -> PathBuf {
    if let Some(dirs) = ProjectDirs::from("ai", "Kadan", "KadanDesktop") {
        return dirs.config_dir().join("servers.json");
    }
    PathBuf::from("servers.json")
}

pub fn load_config() -> DesktopConfig {
    let path = config_path();
    let Ok(raw) = fs::read_to_string(&path) else {
        return DesktopConfig::default();
    };
    serde_json::from_str::<DesktopConfig>(&raw).unwrap_or_default()
}

pub fn save_config(mut config: DesktopConfig) -> Result<DesktopConfig, String> {
    config.active_host = normalize_host(&config.active_host)?;
    for server in &mut config.saved_servers {
        server.host = normalize_host(&server.host)?;
        if server.label.trim().is_empty() {
            server.label = server
                .host
                .trim_start_matches("http://")
                .trim_start_matches("https://")
                .to_string();
        }
    }
    let path = config_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Could not create config folder: {e}"))?;
    }
    let payload =
        serde_json::to_string_pretty(&config).map_err(|e| format!("Config error: {e}"))?;
    fs::write(&path, payload).map_err(|e| format!("Could not save settings: {e}"))?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_hosts_cleanly() {
        assert_eq!(
            normalize_host("192.168.1.50:8630/").unwrap(),
            "http://192.168.1.50:8630"
        );
        assert_eq!(
            normalize_host("https://kadan.home.arpa///").unwrap(),
            "https://kadan.home.arpa"
        );
        assert!(normalize_host("").is_err());
        assert!(normalize_host("ftp://example.com").is_err());
    }
}
