use crate::supervisor::ServerSupervisor;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

pub const DEFAULT_UPDATE_REPO: &str = "ghalrym/Kadan";
const USER_AGENT: &str = concat!("KadanServer/", env!("CARGO_PKG_VERSION"));
const DESKTOP_STALE_SECONDS: u64 = 120;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstallKind {
    Appimage,
    WindowsNsis,
    MacosApp,
    Unsupported,
}

impl InstallKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            InstallKind::Appimage => "appimage",
            InstallKind::WindowsNsis => "windows_nsis",
            InstallKind::MacosApp => "macos_app",
            InstallKind::Unsupported => "unsupported",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseAssetInfo {
    pub filename: String,
    pub url: String,
    pub sha256: Option<String>,
    pub size: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ReleaseManifest {
    version: String,
    tag: String,
    #[serde(default)]
    published_at: Option<String>,
    #[serde(default)]
    release_url: Option<String>,
    #[serde(default)]
    apps: HashMap<String, HashMap<String, ReleaseAssetInfo>>,
}

#[derive(Debug, Clone, Deserialize)]
struct GithubRelease {
    tag_name: String,
    #[serde(default)]
    html_url: String,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    published_at: Option<String>,
    #[serde(default)]
    assets: Vec<GithubAsset>,
}

#[derive(Debug, Clone, Deserialize)]
struct GithubAsset {
    name: String,
    browser_download_url: String,
    #[serde(default)]
    size: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopClientInfo {
    pub connected: bool,
    pub current_version: Option<String>,
    pub update_available: bool,
    pub can_self_update: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingDesktopUpdate {
    pub request_id: String,
    pub target_version: String,
    pub requested_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateCheckResult {
    pub current_version: String,
    pub latest_version: Option<String>,
    pub update_available: bool,
    pub release_tag: Option<String>,
    pub release_url: Option<String>,
    pub release_notes: Option<String>,
    pub published_at: Option<String>,
    pub can_self_update: bool,
    pub install_kind: String,
    pub reason: Option<String>,
    pub staged_version: Option<String>,
    pub desktop: DesktopClientInfo,
    #[serde(skip_serializing)]
    pub server_asset: Option<ReleaseAssetInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStatus {
    pub state: String,
    pub current_version: String,
    pub staged_version: Option<String>,
    pub message: Option<String>,
    pub desktop: DesktopClientInfo,
    pub pending_desktop_update: Option<PendingDesktopUpdate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageResult {
    pub ok: bool,
    pub target_version: String,
    pub server_staged: bool,
    pub desktop_update_available: bool,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyResult {
    pub ok: bool,
    pub target_version: String,
    pub server_restarting: bool,
    pub desktop_update_requested: bool,
    pub pending_desktop_update: Option<PendingDesktopUpdate>,
    pub message: String,
}

#[derive(Debug, Clone)]
struct StagedUpdate {
    version: String,
    asset_path: PathBuf,
    install_kind: InstallKind,
}

#[derive(Debug, Clone)]
struct DesktopHeartbeat {
    version: String,
    can_self_update: bool,
    seen_at: u64,
}

#[derive(Debug, Default)]
struct UpdaterState {
    phase: String,
    message: Option<String>,
    staged: Option<StagedUpdate>,
    desktop: Option<DesktopHeartbeat>,
    pending_desktop_update: Option<PendingDesktopUpdate>,
    latest_known_version: Option<String>,
}

pub struct AppUpdater {
    client: reqwest::Client,
    current_version: String,
    data_dir: Arc<Mutex<PathBuf>>,
    state: Arc<Mutex<UpdaterState>>,
}

fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Parse a release version such as "v0.1.20261126", "0.1.20261126-2", or "0.2.0"
/// into comparable numeric parts `(major, minor, patch, revision)`.
pub fn parse_version(raw: &str) -> Option<(u64, u64, u64, u64)> {
    let trimmed = raw.trim().trim_start_matches(['v', 'V']);
    if trimmed.is_empty() {
        return None;
    }
    let (core, suffix) = match trimmed.split_once('-') {
        Some((c, s)) => (c, Some(s)),
        None => (trimmed, None),
    };
    let mut parts = core.split('.');
    let major = parts.next()?.parse::<u64>().ok()?;
    let minor = parts.next().unwrap_or("0").parse::<u64>().ok()?;
    let patch = parts.next().unwrap_or("0").parse::<u64>().ok()?;
    if parts.next().is_some() {
        return None;
    }
    let revision = match suffix {
        None => 0,
        Some(s) => s
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse::<u64>()
            .unwrap_or(0),
    };
    Some((major, minor, patch, revision))
}

pub fn is_newer_version(current: &str, latest: &str) -> bool {
    match (parse_version(current), parse_version(latest)) {
        (Some(cur), Some(lat)) => lat > cur,
        _ => {
            let c = current.trim().trim_start_matches(['v', 'V']);
            let l = latest.trim().trim_start_matches(['v', 'V']);
            !l.is_empty() && l != c
        }
    }
}

pub fn detect_install_kind() -> (InstallKind, Option<String>) {
    #[cfg(target_os = "linux")]
    {
        if let Ok(path) = std::env::var("APPIMAGE") {
            if !path.trim().is_empty() && Path::new(&path).exists() {
                return (InstallKind::Appimage, None);
            }
        }
        return (
            InstallKind::Unsupported,
            Some(
                "Automatic updates work with the Kadan AppImage. This installation needs to be updated manually."
                    .to_string(),
            ),
        );
    }

    #[cfg(windows)]
    {
        if let Ok(exe) = std::env::current_exe() {
            let lower = exe.to_string_lossy().to_ascii_lowercase();
            if lower.contains("\\target\\debug\\") || lower.contains("\\target\\release\\") {
                return (
                    InstallKind::Unsupported,
                    Some("Development builds cannot be updated automatically.".to_string()),
                );
            }
            return (InstallKind::WindowsNsis, None);
        }
        return (
            InstallKind::Unsupported,
            Some("Could not locate the installed Kadan application.".to_string()),
        );
    }

    #[cfg(target_os = "macos")]
    {
        if find_macos_app_bundle().is_some() {
            return (InstallKind::MacosApp, None);
        }
        return (
            InstallKind::Unsupported,
            Some(
                "Automatic updates work when running from the Kadan app bundle."
                    .to_string(),
            ),
        );
    }

    #[cfg(not(any(target_os = "linux", windows, target_os = "macos")))]
    {
        (
            InstallKind::Unsupported,
            Some("Automatic updates are not available on this platform yet.".to_string()),
        )
    }
}

#[cfg(target_os = "macos")]
fn find_macos_app_bundle() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    for ancestor in exe.ancestors() {
        if ancestor.extension().and_then(|e| e.to_str()) == Some("app") {
            return Some(ancestor.to_path_buf());
        }
    }
    None
}

pub fn platform_manifest_key(install_kind: &InstallKind) -> Option<String> {
    let arch = if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "x86_64"
    };
    match install_kind {
        InstallKind::Appimage => Some(format!("linux-{arch}-appimage")),
        InstallKind::WindowsNsis => Some(format!("windows-{arch}-nsis")),
        InstallKind::MacosApp => Some(format!("macos-{arch}-app")),
        InstallKind::Unsupported => None,
    }
}

fn match_github_asset(
    assets: &[GithubAsset],
    app_filter: &str,
    install_kind: &InstallKind,
) -> Option<ReleaseAssetInfo> {
    let want_ext = match install_kind {
        InstallKind::Appimage => ".appimage",
        InstallKind::WindowsNsis => ".exe",
        InstallKind::MacosApp => ".app.tar.gz",
        InstallKind::Unsupported => return None,
    };
    let is_arm = cfg!(target_arch = "aarch64");
    for asset in assets {
        let low = asset.name.to_ascii_lowercase();
        // The merged app ships as "Kadan"; older releases also had a separate "Kadan Server".
        let is_server = low.contains("server");
        if (app_filter == "kadan-server") != is_server {
            continue;
        }
        if !low.ends_with(want_ext) {
            continue;
        }
        let asset_arm = low.contains("aarch64") || low.contains("arm64");
        if asset_arm != is_arm && (low.contains("x86_64") || low.contains("amd64") || low.contains("x64") || asset_arm) {
            continue;
        }
        return Some(ReleaseAssetInfo {
            filename: asset.name.clone(),
            url: asset.browser_download_url.clone(),
            sha256: None,
            size: asset.size,
        });
    }
    None
}

fn truncate_notes(raw: Option<String>) -> Option<String> {
    let text = raw?.trim().to_string();
    if text.is_empty() {
        return None;
    }
    let max_chars = 1500;
    if text.chars().count() <= max_chars {
        Some(text)
    } else {
        let clipped: String = text.chars().take(max_chars).collect();
        Some(format!("{clipped}…"))
    }
}

impl AppUpdater {
    pub fn new(data_dir: PathBuf) -> Self {
        let client = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .expect("updater http client");
        Self {
            client,
            current_version: env!("CARGO_PKG_VERSION").to_string(),
            data_dir: Arc::new(Mutex::new(data_dir)),
            state: Arc::new(Mutex::new(UpdaterState {
                phase: "idle".to_string(),
                ..Default::default()
            })),
        }
    }

    pub fn set_data_dir(&self, data_dir: PathBuf) {
        *self.data_dir.lock().unwrap() = data_dir;
    }

    fn release_api_url(&self) -> String {
        if let Ok(explicit) = std::env::var("KADAN_UPDATE_API_URL") {
            let trimmed = explicit.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
        let repo = std::env::var("KADAN_UPDATE_REPO")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_UPDATE_REPO.to_string());
        format!("https://api.github.com/repos/{repo}/releases/latest")
    }

    pub fn record_desktop_heartbeat(&self, version: String, can_self_update: bool) {
        let mut guard = self.state.lock().unwrap();
        let trimmed = version.trim().to_string();
        if trimmed.is_empty() {
            return;
        }
        if let Some(pending) = &guard.pending_desktop_update {
            if !is_newer_version(&trimmed, &pending.target_version) {
                guard.pending_desktop_update = None;
            }
        }
        guard.desktop = Some(DesktopHeartbeat {
            version: trimmed,
            can_self_update,
            seen_at: now_epoch(),
        });
    }

    pub fn ack_desktop_update(&self, request_id: &str) -> bool {
        let mut guard = self.state.lock().unwrap();
        if let Some(pending) = &guard.pending_desktop_update {
            if pending.request_id == request_id {
                guard.pending_desktop_update = None;
                return true;
            }
        }
        false
    }

    fn desktop_info_locked(guard: &UpdaterState, latest: Option<&str>) -> DesktopClientInfo {
        let now = now_epoch();
        let Some(hb) = guard.desktop.as_ref() else {
            return DesktopClientInfo {
                connected: false,
                current_version: None,
                update_available: false,
                can_self_update: false,
            };
        };
        if now.saturating_sub(hb.seen_at) > DESKTOP_STALE_SECONDS {
            return DesktopClientInfo {
                connected: false,
                current_version: Some(hb.version.clone()),
                update_available: false,
                can_self_update: hb.can_self_update,
            };
        }
        let latest_ver = latest.or(guard.latest_known_version.as_deref());
        let update_available = latest_ver
            .map(|l| is_newer_version(&hb.version, l))
            .unwrap_or(false);
        DesktopClientInfo {
            connected: true,
            current_version: Some(hb.version.clone()),
            update_available,
            can_self_update: hb.can_self_update,
        }
    }

    pub fn status(&self) -> UpdateStatus {
        let guard = self.state.lock().unwrap();
        let desktop = Self::desktop_info_locked(&guard, None);
        UpdateStatus {
            state: guard.phase.clone(),
            current_version: self.current_version.clone(),
            staged_version: guard.staged.as_ref().map(|s| s.version.clone()),
            message: guard.message.clone(),
            desktop,
            pending_desktop_update: guard.pending_desktop_update.clone(),
        }
    }

    pub async fn check_for_update(&self) -> Result<UpdateCheckResult, String> {
        let url = self.release_api_url();
        let resp = self
            .client
            .get(&url)
            .header("Accept", "application/vnd.github+json")
            .send()
            .await
            .map_err(|e| format!("Could not check for updates right now: {e}"))?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            let (install_kind, reason) = detect_install_kind();
            let guard = self.state.lock().unwrap();
            let desktop = Self::desktop_info_locked(&guard, None);
            return Ok(UpdateCheckResult {
                current_version: self.current_version.clone(),
                latest_version: None,
                update_available: false,
                release_tag: None,
                release_url: None,
                release_notes: None,
                published_at: None,
                can_self_update: install_kind != InstallKind::Unsupported,
                install_kind: install_kind.as_str().to_string(),
                reason: reason.or_else(|| Some("No published releases were found yet.".to_string())),
                staged_version: guard.staged.as_ref().map(|s| s.version.clone()),
                desktop,
                server_asset: None,
            });
        }

        if !resp.status().is_success() {
            return Err(format!(
                "Update check failed with HTTP {}. Try again later.",
                resp.status().as_u16()
            ));
        }

        let release: GithubRelease = resp
            .json()
            .await
            .map_err(|e| format!("Could not read release information: {e}"))?;

        let latest_version = release
            .tag_name
            .trim()
            .trim_start_matches(['v', 'V'])
            .to_string();

        // Try loading release-manifest.json if attached to the release.
        let mut manifest: Option<ReleaseManifest> = None;
        if let Some(manifest_asset) = release
            .assets
            .iter()
            .find(|a| a.name == "release-manifest.json")
        {
            if let Ok(mresp) = self.client.get(&manifest_asset.browser_download_url).send().await {
                if mresp.status().is_success() {
                    manifest = mresp.json::<ReleaseManifest>().await.ok();
                }
            }
        }

        let resolved_version = manifest
            .as_ref()
            .map(|m| m.version.clone())
            .unwrap_or_else(|| latest_version.clone());
        let resolved_tag = manifest
            .as_ref()
            .map(|m| m.tag.clone())
            .unwrap_or_else(|| release.tag_name.clone());
        let resolved_url = manifest
            .as_ref()
            .and_then(|m| m.release_url.clone())
            .or_else(|| {
                if release.html_url.is_empty() {
                    None
                } else {
                    Some(release.html_url.clone())
                }
            });
        let published_at = manifest
            .as_ref()
            .and_then(|m| m.published_at.clone())
            .or(release.published_at.clone());

        let (install_kind, mut reason) = detect_install_kind();
        let server_asset = if install_kind != InstallKind::Unsupported {
            let from_manifest = manifest
                .as_ref()
                .and_then(|m| m.apps.get("kadan"))
                .and_then(|plat_map| {
                    platform_manifest_key(&install_kind)
                        .and_then(|key| plat_map.get(&key).cloned())
                });
            from_manifest.or_else(|| match_github_asset(&release.assets, "kadan", &install_kind))
        } else {
            None
        };

        let server_update_available = is_newer_version(&self.current_version, &resolved_version);
        let can_self_update = install_kind != InstallKind::Unsupported
            && (!server_update_available || server_asset.is_some());
        if server_update_available && install_kind != InstallKind::Unsupported && server_asset.is_none() {
            reason = Some(
                "This release does not include an automatic update package for this platform."
                    .to_string(),
            );
        }

        let mut guard = self.state.lock().unwrap();
        guard.latest_known_version = Some(resolved_version.clone());
        let desktop = Self::desktop_info_locked(&guard, Some(&resolved_version));
        let staged_version = guard.staged.as_ref().map(|s| s.version.clone());

        Ok(UpdateCheckResult {
            current_version: self.current_version.clone(),
            latest_version: Some(resolved_version),
            update_available: server_update_available,
            release_tag: Some(resolved_tag),
            release_url: resolved_url,
            release_notes: truncate_notes(release.body),
            published_at,
            can_self_update,
            install_kind: install_kind.as_str().to_string(),
            reason,
            staged_version,
            desktop,
            server_asset,
        })
    }

    pub async fn stage_update(&self, target_version: Option<String>) -> Result<StageResult, String> {
        {
            let mut guard = self.state.lock().unwrap();
            guard.phase = "checking".to_string();
            guard.message = Some("Checking for updates…".to_string());
        }

        let check = match self.check_for_update().await {
            Ok(c) => c,
            Err(err) => {
                let mut guard = self.state.lock().unwrap();
                guard.phase = "error".to_string();
                guard.message = Some(err.clone());
                return Err(err);
            }
        };

        let Some(latest) = check.latest_version.clone() else {
            let mut guard = self.state.lock().unwrap();
            guard.phase = "idle".to_string();
            guard.message = None;
            return Err("No published Kadan releases were found.".to_string());
        };

        if let Some(expected) = target_version.as_deref() {
            let norm_expected = expected.trim().trim_start_matches(['v', 'V']);
            if !norm_expected.is_empty() && norm_expected != latest {
                return Err(format!(
                    "The latest available release is {latest}, not {norm_expected}. Check for updates again before installing."
                ));
            }
        }

        let desktop_needs_update = check.desktop.connected
            && check.desktop.update_available
            && check.desktop.can_self_update;

        if !check.update_available {
            let mut guard = self.state.lock().unwrap();
            guard.phase = "idle".to_string();
            guard.message = None;
            if desktop_needs_update {
                return Ok(StageResult {
                    ok: true,
                    target_version: latest,
                    server_staged: false,
                    desktop_update_available: true,
                    message: "Kadan is already up to date here; ready to update connected Kadan apps."
                        .to_string(),
                });
            }
            return Err(format!(
                "Kadan is already up to date (version {}).",
                self.current_version
            ));
        }

        if !check.can_self_update {
            let msg = check.reason.unwrap_or_else(|| {
                "Automatic updates are not supported for this installation.".to_string()
            });
            let mut guard = self.state.lock().unwrap();
            guard.phase = "idle".to_string();
            guard.message = Some(msg.clone());
            return Err(msg);
        }

        let Some(asset) = check.server_asset else {
            let msg = "No matching update package was found in the latest release.".to_string();
            let mut guard = self.state.lock().unwrap();
            guard.phase = "error".to_string();
            guard.message = Some(msg.clone());
            return Err(msg);
        };

        // Reuse already-staged file if version matches and file still exists.
        {
            let guard = self.state.lock().unwrap();
            if let Some(existing) = &guard.staged {
                if existing.version == latest && existing.asset_path.exists() {
                    return Ok(StageResult {
                        ok: true,
                        target_version: latest,
                        server_staged: true,
                        desktop_update_available: desktop_needs_update,
                        message: format!("Update {} is already downloaded and verified.", existing.version),
                    });
                }
            }
        }

        let (install_kind, _) = detect_install_kind();
        let staging_dir = self.data_dir.lock().unwrap().join("updates").join("staging");
        tokio::fs::create_dir_all(&staging_dir)
            .await
            .map_err(|e| format!("Could not prepare update folder: {e}"))?;

        let safe_name = Path::new(&asset.filename)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("kadan-update");
        let part_path = staging_dir.join(format!("{safe_name}.part"));
        let final_path = staging_dir.join(safe_name);

        {
            let mut guard = self.state.lock().unwrap();
            guard.phase = "downloading".to_string();
            guard.message = Some(format!("Downloading Kadan {latest}…"));
        }

        let download_result = download_and_verify(&self.client, &asset, &part_path, &final_path).await;
        if let Err(err) = download_result {
            let _ = tokio::fs::remove_file(&part_path).await;
            let mut guard = self.state.lock().unwrap();
            guard.phase = "error".to_string();
            guard.message = Some(err.clone());
            return Err(err);
        }

        {
            let mut guard = self.state.lock().unwrap();
            guard.phase = "staged".to_string();
            guard.message = Some(format!("Kadan {latest} downloaded and verified."));
            guard.staged = Some(StagedUpdate {
                version: latest.clone(),
                asset_path: final_path,
                install_kind,
            });
        }

        Ok(StageResult {
            ok: true,
            target_version: latest,
            server_staged: true,
            desktop_update_available: desktop_needs_update,
            message: "Update downloaded and verified. Ready to install and restart.".to_string(),
        })
    }

    pub fn apply_update(
        self: &Arc<Self>,
        supervisor: Arc<ServerSupervisor>,
        app_handle: Option<tauri::AppHandle>,
        target_version: Option<String>,
        include_desktop: bool,
    ) -> Result<ApplyResult, String> {
        let (staged, desktop_info, latest_known) = {
            let guard = self.state.lock().unwrap();
            let desktop = Self::desktop_info_locked(&guard, target_version.as_deref());
            (
                guard.staged.clone(),
                desktop,
                guard.latest_known_version.clone(),
            )
        };

        let resolved_target = staged
            .as_ref()
            .map(|s| s.version.clone())
            .or(target_version)
            .or(latest_known)
            .ok_or_else(|| "No update has been staged yet. Stage an update first.".to_string())?;

        let should_update_desktop = include_desktop
            && desktop_info.connected
            && desktop_info.can_self_update
            && desktop_info
                .current_version
                .as_deref()
                .map(|v| is_newer_version(v, &resolved_target))
                .unwrap_or(false);

        let pending_desktop = if should_update_desktop {
            let pending = PendingDesktopUpdate {
                request_id: Uuid::new_v4().to_string(),
                target_version: resolved_target.clone(),
                requested_at: now_epoch(),
            };
            self.state.lock().unwrap().pending_desktop_update = Some(pending.clone());
            Some(pending)
        } else {
            None
        };

        let server_restarting = staged.is_some();
        if let Some(staged_update) = staged {
            {
                let mut guard = self.state.lock().unwrap();
                guard.phase = "applying".to_string();
                guard.message = Some(format!(
                    "Installing Kadan {} and restarting…",
                    staged_update.version
                ));
            }
            let updater = self.clone();
            tokio::spawn(async move {
                // Give the HTTP response and any pending desktop poll a brief moment to flush.
                let delay = if should_update_desktop {
                    Duration::from_millis(2500)
                } else {
                    Duration::from_millis(1200)
                };
                tokio::time::sleep(delay).await;
                supervisor.stop_all().await;
                if let Err(err) = install_and_relaunch(&staged_update) {
                    {
                        let mut guard = updater.state.lock().unwrap();
                        guard.phase = "error".to_string();
                        guard.message = Some(err);
                    }
                    // Bring services back up if the swap itself failed before exit.
                    let _ = supervisor.start_all().await;
                    return;
                }
                if let Some(handle) = app_handle {
                    handle.exit(0);
                } else {
                    std::process::exit(0);
                }
            });
        } else if !should_update_desktop {
            return Err("No staged server update or connected desktop update is pending.".to_string());
        }

        let message = match (server_restarting, should_update_desktop) {
            (true, true) => format!(
                "Installing Kadan {resolved_target} here and on connected Kadan apps. They will restart shortly."
            ),
            (true, false) => format!(
                "Installing Kadan {resolved_target}. Kadan will restart shortly."
            ),
            (false, true) => format!(
                "Requested Kadan Desktop to update to {resolved_target} and restart."
            ),
            (false, false) => unreachable!(),
        };

        Ok(ApplyResult {
            ok: true,
            target_version: resolved_target,
            server_restarting,
            desktop_update_requested: should_update_desktop,
            pending_desktop_update: pending_desktop,
            message,
        })
    }
}

pub async fn download_and_verify(
    client: &reqwest::Client,
    asset: &ReleaseAssetInfo,
    part_path: &Path,
    final_path: &Path,
) -> Result<(), String> {
    let resp = client
        .get(&asset.url)
        .send()
        .await
        .map_err(|e| format!("Could not download update package: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!(
            "Update download failed with HTTP {}.",
            resp.status().as_u16()
        ));
    }

    let mut file = tokio::fs::File::create(part_path)
        .await
        .map_err(|e| format!("Could not save update file: {e}"))?;
    let mut hasher = Sha256::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.map_err(|e| format!("Update download was interrupted: {e}"))?;
        hasher.update(&bytes);
        file.write_all(&bytes)
            .await
            .map_err(|e| format!("Could not write update file: {e}"))?;
    }
    file.flush()
        .await
        .map_err(|e| format!("Could not finish writing update file: {e}"))?;
    drop(file);

    if let Some(expected) = asset.sha256.as_deref() {
        let actual = hex::encode(hasher.finalize());
        if !actual.eq_ignore_ascii_case(expected.trim()) {
            return Err(
                "Downloaded update failed integrity verification and was not installed.".to_string(),
            );
        }
    }

    tokio::fs::rename(part_path, final_path)
        .await
        .map_err(|e| format!("Could not finalize downloaded update: {e}"))?;
    Ok(())
}

/// Atomically replace an AppImage file on Linux (handling cross-filesystem moves).
#[cfg(unix)]
pub fn replace_appimage_file(staged_file: &Path, target_appimage: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;

    let parent = target_appimage
        .parent()
        .ok_or_else(|| "Could not determine AppImage folder.".to_string())?;
    let target_name = target_appimage
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("KadanServer.AppImage");
    let sibling_tmp = parent.join(format!(".{target_name}.updating"));

    std::fs::copy(staged_file, &sibling_tmp)
        .map_err(|e| format!("Could not stage updated AppImage next to existing file: {e}"))?;
    std::fs::set_permissions(&sibling_tmp, std::fs::Permissions::from_mode(0o755))
        .map_err(|e| format!("Could not mark updated AppImage executable: {e}"))?;
    std::fs::rename(&sibling_tmp, target_appimage)
        .map_err(|e| format!("Could not replace existing AppImage: {e}"))?;
    let _ = std::fs::remove_file(staged_file);
    Ok(())
}

fn install_and_relaunch(staged: &StagedUpdate) -> Result<(), String> {
    match staged.install_kind {
        InstallKind::Appimage => {
            #[cfg(target_os = "linux")]
            {
                let appimage = std::env::var("APPIMAGE")
                    .map_err(|_| "APPIMAGE path is not set.".to_string())?;
                let target = PathBuf::from(&appimage);
                replace_appimage_file(&staged.asset_path, &target)?;
                std::process::Command::new(&target)
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn()
                    .map_err(|e| format!("Updated AppImage could not be relaunched: {e}"))?;
                Ok(())
            }
            #[cfg(not(target_os = "linux"))]
            {
                Err("AppImage updates are only supported on Linux.".to_string())
            }
        }
        InstallKind::WindowsNsis => {
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                const CREATE_NO_WINDOW: u32 = 0x08000000;

                let current_exe = std::env::current_exe()
                    .map_err(|e| format!("Could not locate current executable: {e}"))?;
                let pid = std::process::id();
                let script_path = std::env::temp_dir().join(format!("kadan-update-{pid}.ps1"));
                let installer = staged.asset_path.to_string_lossy().replace('\'', "''");
                let exe_str = current_exe.to_string_lossy().replace('\'', "''");
                let script_self = script_path.to_string_lossy().replace('\'', "''");
                let script = format!(
                    "$ErrorActionPreference = 'SilentlyContinue'\n\
                     try {{ Wait-Process -Id {pid} -Timeout 30 }} catch {{}}\n\
                     Start-Process -FilePath '{installer}' -ArgumentList '/S' -Wait\n\
                     if (Test-Path '{exe_str}') {{ Start-Process -FilePath '{exe_str}' }}\n\
                     Remove-Item -Force '{installer}'\n\
                     Remove-Item -Force '{script_self}'\n"
                );
                std::fs::write(&script_path, script)
                    .map_err(|e| format!("Could not prepare Windows update helper: {e}"))?;
                std::process::Command::new("powershell")
                    .args([
                        "-NoProfile",
                        "-WindowStyle",
                        "Hidden",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                    ])
                    .arg(&script_path)
                    .creation_flags(CREATE_NO_WINDOW)
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn()
                    .map_err(|e| format!("Could not launch Windows updater: {e}"))?;
                Ok(())
            }
            #[cfg(not(windows))]
            {
                Err("Windows installer updates are only supported on Windows.".to_string())
            }
        }
        InstallKind::MacosApp => {
            #[cfg(target_os = "macos")]
            {
                let app_bundle = find_macos_app_bundle()
                    .ok_or_else(|| "Could not locate macOS app bundle.".to_string())?;
                let parent = app_bundle
                    .parent()
                    .ok_or_else(|| "Could not locate Applications folder.".to_string())?;
                let extract_dir = std::env::temp_dir()
                    .join(format!("kadan-extract-{}", std::process::id()));
                let _ = std::fs::remove_dir_all(&extract_dir);
                std::fs::create_dir_all(&extract_dir)
                    .map_err(|e| format!("Could not create extract directory: {e}"))?;
                let status = std::process::Command::new("/usr/bin/tar")
                    .args(["-xzf"])
                    .arg(&staged.asset_path)
                    .arg("-C")
                    .arg(&extract_dir)
                    .status()
                    .map_err(|e| format!("Could not extract macOS update: {e}"))?;
                if !status.success() {
                    return Err("Failed to unpack the macOS update archive.".to_string());
                }
                let extracted_app = std::fs::read_dir(&extract_dir)
                    .ok()
                    .and_then(|mut entries| {
                        entries.find_map(|e| {
                            let p = e.ok()?.path();
                            (p.extension().and_then(|x| x.to_str()) == Some("app")).then_some(p)
                        })
                    })
                    .ok_or_else(|| "Update archive did not contain an application bundle.".to_string())?;
                let backup = parent.join(".KadanServer.app.old");
                let _ = std::fs::remove_dir_all(&backup);
                std::fs::rename(&app_bundle, &backup)
                    .map_err(|e| format!("Could not move previous app bundle aside: {e}"))?;
                if let Err(e) = std::fs::rename(&extracted_app, &app_bundle) {
                    let _ = std::fs::rename(&backup, &app_bundle);
                    return Err(format!("Could not install updated app bundle: {e}"));
                }
                let _ = std::fs::remove_dir_all(&backup);
                let _ = std::fs::remove_dir_all(&extract_dir);
                let _ = std::fs::remove_file(&staged.asset_path);
                std::process::Command::new("/usr/bin/open")
                    .args(["-n"])
                    .arg(&app_bundle)
                    .spawn()
                    .map_err(|e| format!("Could not relaunch updated app: {e}"))?;
                Ok(())
            }
            #[cfg(not(target_os = "macos"))]
            {
                Err("macOS bundle updates are only supported on macOS.".to_string())
            }
        }
        InstallKind::Unsupported => {
            Err("Automatic updates are not supported for this installation.".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compares_semver_and_date_versions() {
        assert_eq!(parse_version("0.1.0"), Some((0, 1, 0, 0)));
        assert_eq!(parse_version("v0.1.20261126"), Some((0, 1, 20261126, 0)));
        assert_eq!(parse_version("v0.1.20261126-2"), Some((0, 1, 20261126, 2)));
        assert!(is_newer_version("0.1.0", "0.1.20261126"));
        assert!(is_newer_version("0.1.20261126", "0.1.20261126-2"));
        assert!(is_newer_version("0.1.20261126", "0.2.0"));
        assert!(!is_newer_version("0.1.20261126", "0.1.20261126"));
        assert!(!is_newer_version("0.1.20261127", "0.1.20261126"));
    }

    #[test]
    fn matches_the_kadan_app_asset_not_old_server_ones() {
        let assets = vec![
            GithubAsset {
                name: "Kadan_0.1.20261126_amd64.AppImage".to_string(),
                browser_download_url: "https://example/desktop.AppImage".to_string(),
                size: Some(100),
            },
            GithubAsset {
                name: "Kadan Server_0.1.20261126_amd64.AppImage".to_string(),
                browser_download_url: "https://example/server.AppImage".to_string(),
                size: Some(200),
            },
            GithubAsset {
                name: "Kadan Server_0.1.20261126_x64-setup.exe".to_string(),
                browser_download_url: "https://example/server.exe".to_string(),
                size: Some(300),
            },
        ];

        // The merged app updates itself from the "Kadan" file, never an old "Kadan Server" one.
        let linux = match_github_asset(&assets, "kadan", &InstallKind::Appimage).unwrap();
        assert_eq!(linux.url, "https://example/desktop.AppImage");
        assert!(match_github_asset(&assets, "kadan", &InstallKind::WindowsNsis).is_none());
    }

    #[test]
    fn desktop_heartbeat_and_ack_lifecycle() {
        let updater = AppUpdater::new(std::env::temp_dir());
        assert!(!updater.status().desktop.connected);

        updater.record_desktop_heartbeat("0.1.0".to_string(), true);
        let st = updater.status();
        assert!(st.desktop.connected);
        assert_eq!(st.desktop.current_version.as_deref(), Some("0.1.0"));
        assert!(st.desktop.can_self_update);

        {
            let mut guard = updater.state.lock().unwrap();
            guard.pending_desktop_update = Some(PendingDesktopUpdate {
                request_id: "req-1".to_string(),
                target_version: "0.1.20261126".to_string(),
                requested_at: now_epoch(),
            });
        }
        assert!(!updater.ack_desktop_update("wrong"));
        assert!(updater.ack_desktop_update("req-1"));
        assert!(updater.status().pending_desktop_update.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn replaces_appimage_atomically_in_tempdir() {
        let root = std::env::temp_dir().join(format!("kadan-updater-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let target = root.join("KadanServer.AppImage");
        let staged = root.join("staged.AppImage");
        std::fs::write(&target, b"old-binary").unwrap();
        std::fs::write(&staged, b"new-binary").unwrap();

        replace_appimage_file(&staged, &target).unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"new-binary");
        assert!(!staged.exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn download_and_verify_checks_sha256() {
        use axum::{routing::get, Router};
        use tokio::net::TcpListener;

        let payload = b"kadan-release-binary-bytes";
        let expected_sha = hex::encode(Sha256::digest(payload));

        let app = Router::new().route("/pkg.AppImage", get(move || async move { payload.as_slice() }));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let root = std::env::temp_dir().join(format!("kadan-dl-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let client = reqwest::Client::new();

        // Bad checksum must fail cleanly.
        let bad_asset = ReleaseAssetInfo {
            filename: "pkg.AppImage".to_string(),
            url: format!("http://{addr}/pkg.AppImage"),
            sha256: Some("0".repeat(64)),
            size: Some(payload.len() as u64),
        };
        let err = download_and_verify(&client, &bad_asset, &root.join("a.part"), &root.join("a"))
            .await
            .unwrap_err();
        assert!(err.contains("integrity verification"));

        // Matching checksum succeeds.
        let good_asset = ReleaseAssetInfo {
            filename: "pkg.AppImage".to_string(),
            url: format!("http://{addr}/pkg.AppImage"),
            sha256: Some(expected_sha),
            size: Some(payload.len() as u64),
        };
        download_and_verify(&client, &good_asset, &root.join("b.part"), &root.join("b"))
            .await
            .unwrap();
        assert_eq!(std::fs::read(root.join("b")).unwrap(), payload);

        let _ = std::fs::remove_dir_all(&root);
    }
}
