use crate::config::ServerConfig;
use crate::control::ControlEndpoint;
use crate::hardware::collect_hardware;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

const MAX_LOG_LINES: usize = 600;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceId {
    Api,
    Runner,
    Worker,
}

impl ServiceId {
    pub fn all() -> [ServiceId; 3] {
        [ServiceId::Api, ServiceId::Runner, ServiceId::Worker]
    }

    pub fn label(self) -> &'static str {
        match self {
            ServiceId::Api => "Kadan Services (Kadan + Hikari + Tsuchi)",
            ServiceId::Runner => "Kadan Agent Runner",
            ServiceId::Worker => "Local Model Workers",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            ServiceId::Api => "api",
            ServiceId::Runner => "runner",
            ServiceId::Worker => "worker",
        }
    }

    pub fn from_key(key: &str) -> Option<ServiceId> {
        match key {
            "api" => Some(ServiceId::Api),
            "runner" => Some(ServiceId::Runner),
            "worker" => Some(ServiceId::Worker),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServicePhase {
    Stopped,
    Starting,
    Running,
    Error,
}

#[derive(Debug, Clone, Serialize)]
pub struct ServiceStatus {
    pub id: String,
    pub label: String,
    pub phase: ServicePhase,
    pub message: String,
    pub url: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LogEntry {
    pub timestamp: u64,
    pub service: String,
    pub line: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ServerStatusSnapshot {
    pub running: bool,
    pub services: Vec<ServiceStatus>,
    pub server_local_url: String,
    pub hikari_local_url: String,
    pub tsuchi_local_url: String,
    pub worker_binary_found: bool,
    pub python_runtime_found: bool,
}

#[derive(Debug, Clone)]
pub struct RuntimePaths {
    pub python_bin: Option<PathBuf>,
    pub services_root: Option<PathBuf>,
    pub frontend_dir: Option<PathBuf>,
    pub worker_bin: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct CompiledWorkloadInfo {
    id: String,
    #[serde(default)]
    uses_gpu: bool,
}

#[derive(Debug, Deserialize)]
struct WorkerCapabilities {
    #[serde(default)]
    workloads: Vec<CompiledWorkloadInfo>,
}

fn find_first_existing(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|p| p.exists()).cloned()
}

pub fn discover_runtime_paths(resource_dir: Option<&Path>) -> RuntimePaths {
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
    let mut frontend_candidates = Vec::new();
    let mut worker_candidates = Vec::new();

    if let Ok(explicit_py) = std::env::var("KADAN_SERVER_PYTHON") {
        python_candidates.push(PathBuf::from(explicit_py));
    }
    if let Ok(explicit_worker) = std::env::var("KADAN_WORKER_BIN") {
        worker_candidates.push(PathBuf::from(explicit_worker));
    }

    if let Some(res) = resource_dir {
        let rt = res.join("runtime");
        #[cfg(windows)]
        {
            python_candidates.push(rt.join("venv/Scripts/python.exe"));
            python_candidates.push(rt.join("python/python.exe"));
            worker_candidates.push(rt.join("bin/kadan-worker.exe"));
        }
        #[cfg(not(windows))]
        {
            python_candidates.push(rt.join("venv/bin/python"));
            python_candidates.push(rt.join("python/bin/python3"));
            worker_candidates.push(rt.join("bin/kadan-worker"));
        }
        services_candidates.push(rt.join("services"));
        frontend_candidates.push(rt.join("web"));
    }

    if let Some(ed) = &exe_dir {
        #[cfg(windows)]
        worker_candidates.push(ed.join("kadan-worker.exe"));
        #[cfg(not(windows))]
        worker_candidates.push(ed.join("kadan-worker"));
    }

    if let Some(repo) = &dev_repo {
        #[cfg(windows)]
        {
            python_candidates.push(repo.join(".venv/Scripts/python.exe"));
            worker_candidates.push(
                repo.join("apps/kadan-server/src-tauri/resources/runtime/bin/kadan-worker.exe"),
            );
            worker_candidates
                .push(repo.join("services/hikari/inference_worker/build/kadan-worker.exe"));
        }
        #[cfg(not(windows))]
        {
            python_candidates.push(
                repo.join("apps/kadan-server/src-tauri/resources/runtime/venv/bin/python"),
            );
            python_candidates.push(repo.join(".venv/bin/python"));
            worker_candidates.push(
                repo.join("apps/kadan-server/src-tauri/resources/runtime/bin/kadan-worker"),
            );
            worker_candidates
                .push(repo.join("services/hikari/inference_worker/build/kadan-worker"));
        }
        services_candidates.push(repo.join("apps/kadan-server/src-tauri/resources/runtime/services"));
        services_candidates.push(repo.join("services"));
        frontend_candidates.push(repo.join("apps/kadan-server/src-tauri/resources/runtime/web"));
        frontend_candidates.push(repo.join("services/frontend/dist"));
    }

    RuntimePaths {
        python_bin: find_first_existing(&python_candidates),
        services_root: find_first_existing(&services_candidates),
        frontend_dir: find_first_existing(&frontend_candidates),
        worker_bin: find_first_existing(&worker_candidates),
    }
}

struct ManagedState {
    running: bool,
    config: ServerConfig,
    control: Option<ControlEndpoint>,
    statuses: HashMap<ServiceId, ServiceStatus>,
    logs: VecDeque<LogEntry>,
    children: HashMap<ServiceId, Vec<Child>>,
}

pub struct ServerSupervisor {
    paths: RuntimePaths,
    state: Arc<Mutex<ManagedState>>,
    client: reqwest::Client,
}

impl ServerSupervisor {
    pub fn new(
        config: ServerConfig,
        resource_dir: Option<PathBuf>,
        control: Option<ControlEndpoint>,
    ) -> Self {
        let paths = discover_runtime_paths(resource_dir.as_deref());
        let mut statuses = HashMap::new();
        for id in ServiceId::all() {
            statuses.insert(
                id,
                ServiceStatus {
                    id: id.key().to_string(),
                    label: id.label().to_string(),
                    phase: ServicePhase::Stopped,
                    message: "Stopped".to_string(),
                    url: None,
                },
            );
        }
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .expect("http client");
        Self {
            paths,
            state: Arc::new(Mutex::new(ManagedState {
                running: false,
                config,
                control,
                statuses,
                logs: VecDeque::with_capacity(MAX_LOG_LINES),
                children: HashMap::new(),
            })),
            client,
        }
    }

    pub fn set_control_endpoint(&self, control: ControlEndpoint) {
        self.state.lock().unwrap().control = Some(control);
    }

    pub fn update_config(&self, config: ServerConfig) {
        self.state.lock().unwrap().config = config;
    }

    pub fn snapshot(&self) -> ServerStatusSnapshot {
        let guard = self.state.lock().unwrap();
        let services = ServiceId::all()
            .iter()
            .map(|id| guard.statuses[id].clone())
            .collect();
        let base = format!("http://127.0.0.1:{}", guard.config.server_port);
        ServerStatusSnapshot {
            running: guard.running,
            services,
            server_local_url: base.clone(),
            hikari_local_url: format!("{base}/hikari/settings"),
            tsuchi_local_url: format!("{base}/tsuchi"),
            worker_binary_found: self.paths.worker_bin.is_some(),
            python_runtime_found: self.paths.python_bin.is_some()
                && self.paths.services_root.is_some(),
        }
    }

    pub fn logs(&self, filter: Option<&str>) -> Vec<LogEntry> {
        let guard = self.state.lock().unwrap();
        guard
            .logs
            .iter()
            .filter(|entry| filter.map(|f| f == "all" || entry.service == f).unwrap_or(true))
            .cloned()
            .collect()
    }

    fn set_status(
        &self,
        id: ServiceId,
        phase: ServicePhase,
        message: impl Into<String>,
        url: Option<String>,
    ) {
        let mut guard = self.state.lock().unwrap();
        if let Some(slot) = guard.statuses.get_mut(&id) {
            slot.phase = phase;
            slot.message = message.into();
            slot.url = url;
        }
    }

    pub async fn start_all(&self) -> Result<(), String> {
        self.stop_all().await;
        let cfg = self.state.lock().unwrap().config.clone();
        let data_dir = PathBuf::from(&cfg.data_dir);
        for sub in ["tsuchi", "models"] {
            std::fs::create_dir_all(data_dir.join(sub))
                .map_err(|e| format!("Could not prepare data folder: {e}"))?;
        }

        self.state.lock().unwrap().running = true;

        // 1. Combined API (Tsuchi + Hikari SQLite + Kadan + unified frontend on one port)
        self.start_service(ServiceId::Api).await?;
        let base = format!("http://127.0.0.1:{}", cfg.server_port);
        for probe in [
            format!("{base}/tsuchi/api/health"),
            format!("{base}/hikari/health"),
            format!("{base}/kadan/api/health"),
        ] {
            self.wait_healthy(ServiceId::Api, &probe, &base, Duration::from_secs(25))
                .await?;
        }

        // 2. Kadan Agent Runner
        self.start_service(ServiceId::Runner).await?;

        // 3. Native C++ Local Model Workers (Hikari owns all model downloads & lifecycle)
        if let Err(err) = self.start_service(ServiceId::Worker).await {
            push_log_shared(&self.state, "worker", format!("Worker not started: {err}"));
            self.set_status(ServiceId::Worker, ServicePhase::Error, err, None);
        }

        Ok(())
    }

    pub async fn stop_all(&self) {
        self.state.lock().unwrap().running = false;
        for id in [ServiceId::Worker, ServiceId::Runner, ServiceId::Api] {
            self.stop_service(id).await;
        }
    }

    pub async fn restart_service(&self, id: ServiceId) -> Result<(), String> {
        self.stop_service(id).await;
        self.start_service(id).await?;
        if id == ServiceId::Api {
            let cfg = self.state.lock().unwrap().config.clone();
            let base = format!("http://127.0.0.1:{}", cfg.server_port);
            self.wait_healthy(
                id,
                &format!("{base}/kadan/api/health"),
                &base,
                Duration::from_secs(25),
            )
            .await?;
        }
        Ok(())
    }

    async fn stop_service(&self, id: ServiceId) {
        let children = self.state.lock().unwrap().children.remove(&id).unwrap_or_default();
        for mut child in children {
            let _ = child.start_kill();
            let _ = tokio::time::timeout(Duration::from_secs(4), child.wait()).await;
        }
        self.set_status(id, ServicePhase::Stopped, "Stopped", None);
    }

    async fn start_workers(&self, cfg: &ServerConfig) -> Result<(), String> {
        let bin = self.paths.worker_bin.clone().ok_or_else(|| {
            "Local model worker binary was not found in this installation.".to_string()
        })?;
        self.set_status(ServiceId::Worker, ServicePhase::Starting, "Starting…", None);

        let has_gpu = !collect_hardware().gpus.is_empty();
        let probe = Command::new(&bin)
            .arg("--list-workloads")
            .stdin(Stdio::null())
            .output()
            .await
            .map_err(|e| format!("Could not inspect local model worker: {e}"))?;

        let workloads: Vec<String> = if probe.status.success() {
            serde_json::from_slice::<WorkerCapabilities>(&probe.stdout)
                .map(|caps| {
                    caps.workloads
                        .into_iter()
                        .filter(|w| !w.uses_gpu || has_gpu)
                        .map(|w| w.id)
                        .collect()
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        let workloads = if workloads.is_empty() {
            if has_gpu {
                vec![
                    "llm".into(),
                    "speech".into(),
                    "tts".into(),
                    "image".into(),
                ]
            } else {
                Vec::new()
            }
        } else {
            workloads
        };

        let data_dir = PathBuf::from(&cfg.data_dir);
        let ws_url = format!("ws://127.0.0.1:{}/hikari/v1/workers", cfg.server_port);
        let model_dir = data_dir.join("models").to_string_lossy().to_string();
        let worker_data = bin
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("share/kadan"))
            .filter(|p| p.exists());

        let mut spawned = Vec::with_capacity(workloads.len());
        for workload in &workloads {
            let mut cmd = Command::new(&bin);
            cmd.arg(format!("--workload={workload}"))
                .arg(format!("--api-url={ws_url}"))
                .arg(format!("--token={}", cfg.worker_token))
                .arg(format!("--model-dir={model_dir}"))
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true);
            if let Some(ref share) = worker_data {
                cmd.env("KADAN_WORKER_DATA_DIR", share);
            }
            #[cfg(windows)]
            {
                const CREATE_NO_WINDOW: u32 = 0x08000000;
                cmd.creation_flags(CREATE_NO_WINDOW);
            }
            let mut child = cmd
                .spawn()
                .map_err(|e| format!("Failed to start {workload} worker: {e}"))?;
            self.pipe_logs("worker", Some(workload), &mut child);
            spawned.push(child);
        }

        let count = spawned.len();
        self.state
            .lock()
            .unwrap()
            .children
            .insert(ServiceId::Worker, spawned);
        self.set_status(
            ServiceId::Worker,
            ServicePhase::Running,
            format!("Connected ({count} capabilities ready)"),
            None,
        );
        Ok(())
    }

    fn pipe_logs(&self, service: &str, prefix: Option<&str>, child: &mut Child) {
        let state_out = self.state.clone();
        let svc_key = service.to_string();
        let tag = prefix.map(|p| format!("[{p}] ")).unwrap_or_default();
        if let Some(stdout) = child.stdout.take() {
            let s = state_out.clone();
            let k = svc_key.clone();
            let t = tag.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    push_log_shared(&s, &k, format!("{t}{line}"));
                }
            });
        }
        if let Some(stderr) = child.stderr.take() {
            let s = state_out;
            let k = svc_key;
            let t = tag;
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    push_log_shared(&s, &k, format!("{t}{line}"));
                }
            });
        }
    }

    async fn start_service(&self, id: ServiceId) -> Result<(), String> {
        let (cfg, control) = {
            let guard = self.state.lock().unwrap();
            (guard.config.clone(), guard.control.clone())
        };
        if id == ServiceId::Worker {
            return self.start_workers(&cfg).await;
        }

        let bind_host = if cfg.bind_all_interfaces {
            "0.0.0.0"
        } else {
            "127.0.0.1"
        };
        let data_dir = PathBuf::from(&cfg.data_dir);
        let sep = if cfg!(windows) { ";" } else { ":" };

        let mut cmd = match id {
            ServiceId::Worker => unreachable!(),
            ServiceId::Api => {
                let python = self.paths.python_bin.clone().ok_or_else(|| {
                    "Python runtime not found. Run scripts/bundle-kadan-server.sh first."
                        .to_string()
                })?;
                let services = self.paths.services_root.clone().ok_or_else(|| {
                    "Kadan services directory was not found.".to_string()
                })?;
                let pythonpath = [
                    services.join("api"),
                    services.join("hikari"),
                    services.join("tsuchi"),
                    services.join("kadan"),
                    services.join("tracing"),
                ]
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(sep);

                let db_path = data_dir.join("hikari.db");
                let sqlite_url = format!(
                    "sqlite:///{}",
                    db_path.to_string_lossy().replace('\\', "/")
                );
                let local_base = format!("http://127.0.0.1:{}", cfg.server_port);

                let mut c = Command::new(python);
                c.current_dir(services.join("api"))
                    .env("PYTHONUNBUFFERED", "1")
                    .env("PYTHONPATH", pythonpath)
                    .env("TSUCHI_DATA_DIR", data_dir.join("tsuchi"))
                    .env("HIKARI_DATABASE_URL", sqlite_url)
                    .env("HIKARI_AUTO_MIGRATE", "true")
                    .env("HIKARI_WORKER_TOKEN", &cfg.worker_token)
                    .env("HIKARI_MODEL_DIR", data_dir.join("models"))
                    .env("KADAN_TSUCHI_URL", format!("{local_base}/tsuchi"))
                    .env("KADAN_HIKARI_URL", format!("{local_base}/hikari"))
                    .env("KADAN_APP_VERSION", env!("CARGO_PKG_VERSION"))
                    .args([
                        "-m",
                        "uvicorn",
                        "combined.main:app",
                        "--host",
                        bind_host,
                        "--port",
                        &cfg.server_port.to_string(),
                    ]);
                if let Some(ref ctrl) = control {
                    c.env("KADAN_SERVER_CONTROL_URL", &ctrl.url)
                        .env("KADAN_SERVER_CONTROL_TOKEN", &ctrl.token);
                }
                if let Some(web) = &self.paths.frontend_dir {
                    c.env("KADAN_FRONTEND_DIR", web);
                }
                c
            }
            ServiceId::Runner => {
                let python = self.paths.python_bin.clone().ok_or_else(|| {
                    "Python runtime not found.".to_string()
                })?;
                let services = self.paths.services_root.clone().ok_or_else(|| {
                    "Kadan services directory was not found.".to_string()
                })?;
                let kadan_dir = services.join("kadan");
                let tracing = services.join("tracing");
                let local_base = format!("http://127.0.0.1:{}", cfg.server_port);
                let mut c = Command::new(python);
                c.current_dir(&kadan_dir)
                    .env("PYTHONUNBUFFERED", "1")
                    .env(
                        "PYTHONPATH",
                        format!("{}{sep}{}", kadan_dir.display(), tracing.display()),
                    )
                    .env("KADAN_TSUCHI_URL", format!("{local_base}/tsuchi"))
                    .env("KADAN_HIKARI_URL", format!("{local_base}/hikari"))
                    .env("KADAN_APP_VERSION", env!("CARGO_PKG_VERSION"))
                    .env("KADAN_RUNNER_ID", "runner-1")
                    .args(["-m", "kadan.runner"]);
                if let Some(ref ctrl) = control {
                    c.env("KADAN_SERVER_CONTROL_URL", &ctrl.url)
                        .env("KADAN_SERVER_CONTROL_TOKEN", &ctrl.token);
                }
                c
            }
        };

        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        #[cfg(windows)]
        {
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        self.set_status(id, ServicePhase::Starting, "Starting…", None);
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to start {}: {e}", id.label()))?;

        self.pipe_logs(id.key(), None, &mut child);

        self.state.lock().unwrap().children.insert(id, vec![child]);
        if matches!(id, ServiceId::Runner) {
            self.set_status(id, ServicePhase::Running, "Running", None);
        }
        Ok(())
    }

    async fn wait_healthy(
        &self,
        id: ServiceId,
        health_url: &str,
        base_url: &str,
        timeout: Duration,
    ) -> Result<(), String> {
        let deadline = tokio::time::Instant::now() + timeout;
        while tokio::time::Instant::now() < deadline {
            if let Ok(resp) = self.client.get(health_url).send().await {
                if resp.status().is_success() {
                    self.set_status(
                        id,
                        ServicePhase::Running,
                        "Healthy",
                        Some(base_url.to_string()),
                    );
                    return Ok(());
                }
            }
            let exited = {
                let mut guard = self.state.lock().unwrap();
                guard
                    .children
                    .get_mut(&id)
                    .and_then(|list| list.first_mut())
                    .and_then(|child| child.try_wait().ok().flatten())
            };
            if let Some(status) = exited {
                let msg = format!("Exited early with status {status}. Check the Logs tab.");
                self.set_status(id, ServicePhase::Error, msg.clone(), None);
                return Err(format!("{} failed to start: {msg}", id.label()));
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        let msg = format!("Timed out waiting for {health_url}.");
        self.set_status(id, ServicePhase::Error, msg.clone(), None);
        Err(format!("{}: {msg}", id.label()))
    }
}

fn push_log_shared(state: &Arc<Mutex<ManagedState>>, service: &str, line: String) {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut guard = state.lock().unwrap();
    if guard.logs.len() >= MAX_LOG_LINES {
        guard.logs.pop_front();
    }
    guard.logs.push_back(LogEntry {
        timestamp: now,
        service: service.to_string(),
        line,
    });
}
