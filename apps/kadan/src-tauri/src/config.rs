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

/// Where Kadan's services run: here (this app starts them) or on another computer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    Host,
    Connect,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DesktopConfig {
    /// Missing in settings saved before Kadan and Kadan Server merged; see `load_config`.
    #[serde(default)]
    pub mode: Option<RunMode>,
    /// The server to use in `Connect` mode.
    pub active_host: String,
    #[serde(default)]
    pub saved_servers: Vec<SavedServer>,
}

impl Default for DesktopConfig {
    fn default() -> Self {
        Self {
            mode: Some(RunMode::Host),
            active_host: "http://127.0.0.1:8000".to_string(),
            saved_servers: vec![SavedServer {
                id: "local".to_string(),
                label: "This computer".to_string(),
                host: "http://127.0.0.1:8000".to_string(),
            }],
        }
    }
}

impl DesktopConfig {
    pub fn run_mode(&self) -> RunMode {
        self.mode.unwrap_or(RunMode::Host)
    }

    /// The address every request goes to: this computer's services, or the chosen server.
    pub fn upstream(&self, host_port: u16) -> String {
        match self.run_mode() {
            RunMode::Host => format!("http://127.0.0.1:{host_port}"),
            RunMode::Connect => self.active_host.clone(),
        }
    }
}

pub fn normalize_host(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("Enter a server address such as http://192.168.1.50:8000.".to_string());
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

fn is_loopback(host: &str) -> bool {
    Url::parse(host)
        .ok()
        .and_then(|url| url.host_str().map(|h| h == "127.0.0.1" || h == "localhost" || h == "::1"))
        .unwrap_or(false)
}

pub fn config_path() -> PathBuf {
    if let Some(dirs) = ProjectDirs::from("ai", "Kadan", "KadanDesktop") {
        return dirs.config_dir().join("servers.json");
    }
    PathBuf::from("servers.json")
}

/// Settings saved before the merge have no mode. A client pointed at another computer
/// keeps connecting to it; one pointed at this computer runs Kadan here. A computer that
/// only ran Kadan Server has no client settings at all and gets the default, hosting.
pub fn resolve_mode(saved: Option<RunMode>, active_host: &str) -> RunMode {
    saved.unwrap_or(if is_loopback(active_host) {
        RunMode::Host
    } else {
        RunMode::Connect
    })
}

pub fn load_config() -> DesktopConfig {
    let path = config_path();
    let Ok(raw) = fs::read_to_string(&path) else {
        return DesktopConfig::default();
    };
    let mut config = serde_json::from_str::<DesktopConfig>(&raw).unwrap_or_default();
    config.mode = Some(resolve_mode(config.mode, &config.active_host));
    config
}

pub fn save_config(mut config: DesktopConfig) -> Result<DesktopConfig, String> {
    config.mode = Some(config.run_mode());
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

    #[test]
    fn older_settings_keep_doing_what_they_did() {
        assert_eq!(resolve_mode(None, "http://192.168.0.19:8000"), RunMode::Connect);
        assert_eq!(resolve_mode(None, "http://127.0.0.1:8000"), RunMode::Host);
        assert_eq!(resolve_mode(None, "http://localhost:9000"), RunMode::Host);
        assert_eq!(resolve_mode(Some(RunMode::Connect), "http://127.0.0.1:8000"), RunMode::Connect);
        let legacy: DesktopConfig =
            serde_json::from_str(r#"{"active_host":"http://10.0.0.2:8000","saved_servers":[]}"#).unwrap();
        assert_eq!(legacy.mode, None);
    }

    #[test]
    fn host_mode_talks_to_this_computer() {
        let mut config = DesktopConfig::default();
        config.active_host = "http://10.0.0.2:8000".to_string();
        assert_eq!(config.upstream(8123), "http://127.0.0.1:8123");
        config.mode = Some(RunMode::Connect);
        assert_eq!(config.upstream(8123), "http://10.0.0.2:8000");
    }
}
