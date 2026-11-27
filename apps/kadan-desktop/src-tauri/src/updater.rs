use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::io::AsyncWriteExt;

pub const DEFAULT_UPDATE_REPO: &str = "ghalrym/Kadan";
const USER_AGENT: &str = concat!("KadanDesktop/", env!("CARGO_PKG_VERSION"));

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
pub struct DesktopUpdateCheck {
    pub current_version: String,
    pub latest_version: Option<String>,
    pub update_available: bool,
    pub release_tag: Option<String>,
    pub release_url: Option<String>,
    pub can_self_update: bool,
    pub install_kind: String,
    pub reason: Option<String>,
    #[serde(skip_serializing)]
    pub asset: Option<ReleaseAssetInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopInstallResponse {
    pub ok: bool,
    pub target_version: Option<String>,
    pub restarting: bool,
    pub message: String,
}

pub fn current_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

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
            Some("Automatic updates work when running from the Kadan app bundle.".to_string()),
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

pub fn can_self_update_locally() -> bool {
    detect_install_kind().0 != InstallKind::Unsupported
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

fn platform_manifest_key(install_kind: &InstallKind) -> Option<String> {
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

fn match_desktop_asset(
    assets: &[GithubAsset],
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
        if low.contains("server") || !low.ends_with(want_ext) {
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

fn release_api_url() -> String {
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

fn build_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .timeout(Duration::from_secs(300))
        .connect_timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))
}

pub async fn check_for_update() -> Result<DesktopUpdateCheck, String> {
    let client = build_client()?;
    let url = release_api_url();
    let resp = client
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| format!("Could not check for updates: {e}"))?;

    let (install_kind, mut reason) = detect_install_kind();
    let cur = current_version().to_string();

    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(DesktopUpdateCheck {
            current_version: cur,
            latest_version: None,
            update_available: false,
            release_tag: None,
            release_url: None,
            can_self_update: install_kind != InstallKind::Unsupported,
            install_kind: install_kind.as_str().to_string(),
            reason: reason.or_else(|| Some("No published releases were found yet.".to_string())),
            asset: None,
        });
    }

    if !resp.status().is_success() {
        return Err(format!(
            "Update check failed with HTTP {}.",
            resp.status().as_u16()
        ));
    }

    let release: GithubRelease = resp
        .json()
        .await
        .map_err(|e| format!("Could not read release information: {e}"))?;

    let mut manifest: Option<ReleaseManifest> = None;
    if let Some(m) = release.assets.iter().find(|a| a.name == "release-manifest.json") {
        if let Ok(mresp) = client.get(&m.browser_download_url).send().await {
            if mresp.status().is_success() {
                manifest = mresp.json::<ReleaseManifest>().await.ok();
            }
        }
    }

    let latest_version = manifest
        .as_ref()
        .map(|m| m.version.clone())
        .unwrap_or_else(|| release.tag_name.trim().trim_start_matches(['v', 'V']).to_string());
    let release_tag = manifest
        .as_ref()
        .map(|m| m.tag.clone())
        .unwrap_or_else(|| release.tag_name.clone());
    let release_url = manifest
        .as_ref()
        .and_then(|m| m.release_url.clone())
        .or_else(|| (!release.html_url.is_empty()).then_some(release.html_url.clone()));

    let asset = if install_kind != InstallKind::Unsupported {
        let from_manifest = manifest
            .as_ref()
            .and_then(|m| m.apps.get("kadan-desktop"))
            .and_then(|plat_map| {
                platform_manifest_key(&install_kind).and_then(|key| plat_map.get(&key).cloned())
            });
        from_manifest.or_else(|| match_desktop_asset(&release.assets, &install_kind))
    } else {
        None
    };

    let update_available = is_newer_version(&cur, &latest_version);
    let can_self_update =
        install_kind != InstallKind::Unsupported && (!update_available || asset.is_some());
    if update_available && install_kind != InstallKind::Unsupported && asset.is_none() {
        reason = Some("No automatic update package is available for this platform.".to_string());
    }

    Ok(DesktopUpdateCheck {
        current_version: cur,
        latest_version: Some(latest_version),
        update_available,
        release_tag: Some(release_tag),
        release_url,
        can_self_update,
        install_kind: install_kind.as_str().to_string(),
        reason,
        asset,
    })
}

pub async fn download_and_install(app_handle: Option<tauri::AppHandle>) -> Result<DesktopInstallResponse, String> {
    let check = check_for_update().await?;
    if !check.update_available {
        return Ok(DesktopInstallResponse {
            ok: true,
            target_version: check.latest_version,
            restarting: false,
            message: format!("Kadan Desktop is already up to date ({}).", check.current_version),
        });
    }
    if !check.can_self_update {
        return Err(check
            .reason
            .unwrap_or_else(|| "Automatic updates are not supported for this installation.".to_string()));
    }
    let asset = check
        .asset
        .ok_or_else(|| "No matching update package was found.".to_string())?;
    let target_version = check.latest_version.clone().unwrap_or_default();
    let (install_kind, _) = detect_install_kind();

    let staging_dir = std::env::temp_dir().join("kadan-desktop-update");
    tokio::fs::create_dir_all(&staging_dir)
        .await
        .map_err(|e| format!("Could not prepare update folder: {e}"))?;
    let safe_name = Path::new(&asset.filename)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("kadan-desktop-update");
    let part_path = staging_dir.join(format!("{safe_name}.part"));
    let final_path = staging_dir.join(safe_name);

    let client = build_client()?;
    if let Err(err) = download_and_verify(&client, &asset, &part_path, &final_path).await {
        let _ = tokio::fs::remove_file(&part_path).await;
        return Err(err);
    }

    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(800)).await;
        if install_and_relaunch(&install_kind, &final_path).is_ok() {
            if let Some(handle) = app_handle {
                handle.exit(0);
            } else {
                std::process::exit(0);
            }
        }
    });

    Ok(DesktopInstallResponse {
        ok: true,
        target_version: Some(target_version.clone()),
        restarting: true,
        message: format!("Installing Kadan {target_version} and restarting…"),
    })
}

async fn download_and_verify(
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

#[cfg(unix)]
fn replace_appimage_file(staged_file: &Path, target_appimage: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;

    let parent = target_appimage
        .parent()
        .ok_or_else(|| "Could not determine AppImage folder.".to_string())?;
    let target_name = target_appimage
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("Kadan.AppImage");
    let sibling_tmp = parent.join(format!(".{target_name}.updating"));

    std::fs::copy(staged_file, &sibling_tmp)
        .map_err(|e| format!("Could not stage updated AppImage: {e}"))?;
    std::fs::set_permissions(&sibling_tmp, std::fs::Permissions::from_mode(0o755))
        .map_err(|e| format!("Could not mark updated AppImage executable: {e}"))?;
    std::fs::rename(&sibling_tmp, target_appimage)
        .map_err(|e| format!("Could not replace existing AppImage: {e}"))?;
    let _ = std::fs::remove_file(staged_file);
    Ok(())
}

fn install_and_relaunch(install_kind: &InstallKind, staged_path: &Path) -> Result<(), String> {
    match install_kind {
        InstallKind::Appimage => {
            #[cfg(target_os = "linux")]
            {
                let appimage = std::env::var("APPIMAGE")
                    .map_err(|_| "APPIMAGE path is not set.".to_string())?;
                let target = PathBuf::from(&appimage);
                replace_appimage_file(staged_path, &target)?;
                std::process::Command::new(&target)
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn()
                    .map_err(|e| format!("Could not relaunch Kadan: {e}"))?;
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
                let script_path = std::env::temp_dir().join(format!("kadan-desktop-update-{pid}.ps1"));
                let installer = staged_path.to_string_lossy().replace('\'', "''");
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
                    .join(format!("kadan-desktop-extract-{}", std::process::id()));
                let _ = std::fs::remove_dir_all(&extract_dir);
                std::fs::create_dir_all(&extract_dir)
                    .map_err(|e| format!("Could not create extract directory: {e}"))?;
                let status = std::process::Command::new("/usr/bin/tar")
                    .args(["-xzf"])
                    .arg(staged_path)
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
                let backup = parent.join(".Kadan.app.old");
                let _ = std::fs::remove_dir_all(&backup);
                std::fs::rename(&app_bundle, &backup)
                    .map_err(|e| format!("Could not move previous app bundle aside: {e}"))?;
                if let Err(e) = std::fs::rename(&extracted_app, &app_bundle) {
                    let _ = std::fs::rename(&backup, &app_bundle);
                    return Err(format!("Could not install updated app bundle: {e}"));
                }
                let _ = std::fs::remove_dir_all(&backup);
                let _ = std::fs::remove_dir_all(&extract_dir);
                let _ = std::fs::remove_file(staged_path);
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
    fn desktop_asset_matching_skips_server_packages() {
        let assets = vec![
            GithubAsset {
                name: "Kadan Server_0.1.20261126_amd64.AppImage".to_string(),
                browser_download_url: "https://example/server.AppImage".to_string(),
                size: Some(200),
            },
            GithubAsset {
                name: "Kadan_0.1.20261126_amd64.AppImage".to_string(),
                browser_download_url: "https://example/desktop.AppImage".to_string(),
                size: Some(100),
            },
        ];
        let matched = match_desktop_asset(&assets, &InstallKind::Appimage).unwrap();
        assert_eq!(matched.url, "https://example/desktop.AppImage");
    }

    #[test]
    fn compares_desktop_versions() {
        assert!(is_newer_version("0.1.0", "v0.1.20261126"));
        assert!(!is_newer_version("0.1.20261126", "0.1.20261126"));
    }
}
