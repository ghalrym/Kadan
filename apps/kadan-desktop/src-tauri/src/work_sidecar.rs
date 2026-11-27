use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

#[derive(Debug, Clone)]
pub struct SidecarPaths {
    pub python_bin: Option<PathBuf>,
    pub services_root: Option<PathBuf>,
}

fn find_first_existing(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|p| p.exists()).cloned()
}

pub fn discover_sidecar_paths(resource_dir: Option<&Path>) -> SidecarPaths {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));
    let dev_repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf());

    let mut python_candidates = Vec::new();
    let mut services_candidates = Vec::new();

    if let Ok(explicit_py) = std::env::var("KADAN_DESKTOP_PYTHON") {
        if !explicit_py.trim().is_empty() {
            python_candidates.push(PathBuf::from(explicit_py));
        }
    }
    if let Ok(explicit_services) = std::env::var("KADAN_DESKTOP_SERVICES_DIR") {
        if !explicit_services.trim().is_empty() {
            services_candidates.push(PathBuf::from(explicit_services));
        }
    }

    if let Some(res) = resource_dir {
        let rt = res.join("work-runtime");
        #[cfg(windows)]
        python_candidates.push(rt.join("venv/Scripts/python.exe"));
        #[cfg(not(windows))]
        python_candidates.push(rt.join("venv/bin/python"));
        services_candidates.push(rt.join("services"));
    }

    if let Some(exe) = exe_dir {
        let rt = exe.join("work-runtime");
        #[cfg(windows)]
        python_candidates.push(rt.join("venv/Scripts/python.exe"));
        #[cfg(not(windows))]
        python_candidates.push(rt.join("venv/bin/python"));
        services_candidates.push(rt.join("services"));
    }

    if let Some(repo) = dev_repo {
        #[cfg(windows)]
        python_candidates.push(repo.join(".venv/Scripts/python.exe"));
        #[cfg(not(windows))]
        python_candidates.push(repo.join(".venv/bin/python"));
        services_candidates.push(repo.join("services"));
    }

    SidecarPaths {
        python_bin: find_first_existing(&python_candidates),
        services_root: find_first_existing(&services_candidates),
    }
}

fn allocate_loopback_port() -> Result<u16, String> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|e| format!("Could not allocate a local port for Work: {e}"))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("Could not read allocated local port for Work: {e}"))?
        .port();
    drop(listener);
    Ok(port)
}

fn build_pythonpath(services_root: &Path) -> String {
    let mut parts: Vec<PathBuf> = vec![
        services_root.join("kadan"),
        services_root.join("tracing"),
    ];
    if let Ok(existing) = std::env::var("PYTHONPATH") {
        if !existing.trim().is_empty() {
            parts.push(PathBuf::from(existing));
        }
    }
    std::env::join_paths(parts)
        .ok()
        .and_then(|os| os.into_string().ok())
        .unwrap_or_else(|| services_root.join("kadan").display().to_string())
}

struct RunningSidecar {
    base_url: String,
    active_host: String,
    child: Option<Child>,
}

#[derive(Clone)]
pub struct WorkSidecar {
    resource_dir: Option<PathBuf>,
    fixed_url: Option<String>,
    state: Arc<Mutex<Option<RunningSidecar>>>,
}

impl WorkSidecar {
    pub fn new(resource_dir: Option<PathBuf>) -> Self {
        Self {
            resource_dir,
            fixed_url: None,
            state: Arc::new(Mutex::new(None)),
        }
    }

    #[cfg(test)]
    pub fn fixed_url(url: String) -> Self {
        Self {
            resource_dir: None,
            fixed_url: Some(url.trim_end_matches('/').to_string()),
            state: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn ensure_running(
        &self,
        client: &reqwest::Client,
        active_host: &str,
    ) -> Result<String, String> {
        if let Some(url) = &self.fixed_url {
            return Ok(url.clone());
        }

        let mut guard = self.state.lock().await;
        if let Some(running) = guard.as_mut() {
            let still_alive = match running.child.as_mut() {
                Some(child) => matches!(child.try_wait(), Ok(None)),
                None => true,
            };
            if still_alive {
                let url = running.base_url.clone();
                if running.active_host != active_host {
                    let target = format!("{url}/api/work/upstream");
                    let _ = client
                        .put(&target)
                        .json(&serde_json::json!({ "active_host": active_host }))
                        .timeout(Duration::from_secs(3))
                        .send()
                        .await;
                    running.active_host = active_host.to_string();
                }
                return Ok(url);
            }
            *guard = None;
        }

        let paths = discover_sidecar_paths(self.resource_dir.as_deref());
        let python_bin = paths.python_bin.ok_or_else(|| {
            "Local Work runtime is not available on this computer (Python environment not found)."
                .to_string()
        })?;
        let services_root = paths.services_root.ok_or_else(|| {
            "Local Work runtime files were not found on this computer.".to_string()
        })?;

        let port = allocate_loopback_port()?;
        let base_url = format!("http://127.0.0.1:{port}");
        let trimmed_host = active_host.trim_end_matches('/');
        let tsuchi_url = format!("{trimmed_host}/tsuchi");
        let hikari_url = format!("{trimmed_host}/hikari");

        let mut cmd = Command::new(&python_bin);
        cmd.args([
            "-m",
            "kadan.work.desktop_server",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--active-host",
            trimmed_host,
        ])
        .env("PYTHONPATH", build_pythonpath(&services_root))
        .env("KADAN_TSUCHI_URL", &tsuchi_url)
        .env("KADAN_HIKARI_URL", &hikari_url)
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Could not start local Work runtime: {e}"))?;

        let health_url = format!("{base_url}/api/work/health");
        let mut ready = false;
        for _ in 0..80 {
            if let Ok(Some(status)) = child.try_wait() {
                return Err(format!(
                    "Local Work runtime exited before becoming ready ({status})."
                ));
            }
            if let Ok(resp) = client
                .get(&health_url)
                .timeout(Duration::from_millis(250))
                .send()
                .await
            {
                if resp.status().is_success() {
                    ready = true;
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        if !ready {
            let _ = child.kill().await;
            return Err("Timed out waiting for the local Work runtime to start.".to_string());
        }

        *guard = Some(RunningSidecar {
            base_url: base_url.clone(),
            active_host: active_host.to_string(),
            child: Some(child),
        });
        Ok(base_url)
    }

    pub async fn update_upstream(&self, client: &reqwest::Client, active_host: &str) {
        if self.fixed_url.is_some() {
            return;
        }
        let mut guard = self.state.lock().await;
        if let Some(running) = guard.as_mut() {
            let target = format!("{}/api/work/upstream", running.base_url);
            let _ = client
                .put(&target)
                .json(&serde_json::json!({ "active_host": active_host }))
                .timeout(Duration::from_secs(3))
                .send()
                .await;
            running.active_host = active_host.to_string();
        }
    }

    pub async fn shutdown(&self) {
        let mut guard = self.state.lock().await;
        if let Some(mut running) = guard.take() {
            if let Some(mut child) = running.child.take() {
                let _ = child.kill().await;
                let _ = child.wait().await;
            }
        }
    }
}
