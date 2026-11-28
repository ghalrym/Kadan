use serde::Serialize;
use std::net::UdpSocket;
use std::process::Command;
use sysinfo::System;

#[derive(Debug, Clone, Default, Serialize)]
pub struct GpuInfo {
    pub name: String,
    pub total_vram_mb: u64,
    pub free_vram_mb: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct HardwareSummary {
    pub hostname: String,
    pub lan_ip: Option<String>,
    pub cpu_cores: usize,
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    pub gpus: Vec<GpuInfo>,
}

pub fn detect_lan_ip() -> Option<String> {
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    let ip = socket.local_addr().ok()?.ip();
    if ip.is_loopback() || ip.is_unspecified() {
        return None;
    }
    Some(ip.to_string())
}

fn detect_nvidia_gpus() -> Vec<GpuInfo> {
    let Ok(output) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split(',').map(|p| p.trim()).collect();
            if parts.len() < 3 {
                return None;
            }
            Some(GpuInfo {
                name: parts[0].to_string(),
                total_vram_mb: parts[1].parse().ok()?,
                free_vram_mb: parts[2].parse().ok()?,
            })
        })
        .collect()
}

pub fn collect_hardware() -> HardwareSummary {
    let mut sys = System::new_all();
    sys.refresh_memory();
    let total_ram_mb = sys.total_memory() / (1024 * 1024);
    let available_ram_mb = sys.available_memory() / (1024 * 1024);
    let cpu_cores = sys.cpus().len();
    let hostname = System::host_name().unwrap_or_else(|| "this-computer".to_string());
    HardwareSummary {
        hostname,
        lan_ip: detect_lan_ip(),
        cpu_cores,
        total_ram_mb,
        available_ram_mb,
        gpus: detect_nvidia_gpus(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collects_system_memory_and_cpus() {
        let hw = collect_hardware();
        assert!(hw.cpu_cores > 0);
        assert!(hw.total_ram_mb > 0);
    }
}
