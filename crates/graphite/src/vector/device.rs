//! Hardware execution device selection (CPU vs CUDA/GPU) with graceful fallback and installer hints.

use serde::{Deserialize, Serialize};

/// Target hardware acceleration device for embedding and reranking neural models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DeviceType {
    /// Automatically detects CUDA-capable GPU; falls back to CPU SIMD if unavailable.
    #[default]
    Auto,
    /// Standard multi-threaded CPU execution (SIMD AVX2/AVX-512).
    Cpu,
    /// NVIDIA GPU hardware acceleration via CUDA Execution Provider.
    Cuda(u32),
}

/// Status of CUDA hardware and driver availability on the host system.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CudaStatus {
    /// NVIDIA GPU is detected and CUDA driver/runtime is fully operational.
    Available { device_count: usize },
    /// NVIDIA GPU hardware is physically present, but NVIDIA drivers or CUDA libraries are missing or uninitialized.
    GpuDetectedDriverMissing {
        distro_id: String,
        install_command: String,
    },
    /// No NVIDIA GPU detected on this system (Intel / AMD / Virtualization / CPU-only).
    NoGpuDetected,
}

impl DeviceType {
    /// Returns the canonical string representation of the device.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda(_) => "cuda",
        }
    }

    /// Parses a device string into `DeviceType`.
    pub fn from_str_name(name: &str) -> Self {
        match name.trim().to_lowercase().as_str() {
            "cuda" | "gpu" | "nvidia" => Self::Cuda(0),
            "cpu" => Self::Cpu,
            _ => Self::Auto,
        }
    }

    /// Returns true if this device explicitly requests or resolves to CUDA.
    pub fn is_cuda(&self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    /// Checks whether NVIDIA drivers and CUDA runtime are actively available in the current environment.
    pub fn is_cuda_available() -> bool {
        // 1. Check if NVIDIA driver module or device is accessible in /dev/nvidia*
        let dev_exists = std::path::Path::new("/dev/nvidia0").exists()
            || std::path::Path::new("/dev/nvidiactl").exists()
            || std::path::Path::new("/proc/driver/nvidia/version").exists();

        if dev_exists {
            return true;
        }

        // 2. Check if CUDA_VISIBLE_DEVICES or NVIDIA_VISIBLE_DEVICES is set and valid
        if let Ok(val) = std::env::var("CUDA_VISIBLE_DEVICES") {
            let trimmed = val.trim();
            if !trimmed.is_empty() && trimmed != "-1" {
                return true;
            }
        }

        // 3. Optional quick execution check for nvidia-smi if available
        if let Ok(output) = std::process::Command::new("nvidia-smi").output() {
            if output.status.success() {
                return true;
            }
        }

        false
    }

    /// Resolves `Auto` to either `Cuda(0)` or `Cpu` based on live hardware detection.
    pub fn resolve(&self) -> Self {
        match self {
            Self::Auto => {
                if Self::is_cuda_available() {
                    Self::Cuda(0)
                } else {
                    Self::Cpu
                }
            }
            other => *other,
        }
    }
}

impl CudaStatus {
    /// Detects the current host system CUDA and NVIDIA hardware status.
    pub fn detect() -> Self {
        // 1. Check if CUDA is active and working
        if DeviceType::is_cuda_available() {
            return Self::Available { device_count: 1 };
        }

        // 2. Check if an NVIDIA GPU is physically present on the PCI bus
        let mut nvidia_gpu_found = false;

        #[cfg(target_os = "linux")]
        {
            if let Ok(entries) = std::fs::read_dir("/sys/bus/pci/devices") {
                for entry in entries.flatten() {
                    let vendor_path = entry.path().join("vendor");
                    if let Ok(vendor_str) = std::fs::read_to_string(vendor_path) {
                        if vendor_str.trim().eq_ignore_ascii_case("0x10de") {
                            nvidia_gpu_found = true;
                            break;
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            if std::path::Path::new("C:\\Program Files\\NVIDIA Corporation").exists() {
                nvidia_gpu_found = true;
            }
        }

        if nvidia_gpu_found {
            let (distro_id, install_command) = Self::detect_install_command();
            Self::GpuDetectedDriverMissing {
                distro_id,
                install_command,
            }
        } else {
            Self::NoGpuDetected
        }
    }

    /// Determines the recommended install command based on the OS or Linux distribution.
    fn detect_install_command() -> (String, String) {
        #[cfg(target_os = "linux")]
        {
            if let Ok(os_release) = std::fs::read_to_string("/etc/os-release") {
                let lower = os_release.to_lowercase();
                if lower.contains("arch") || lower.contains("omarchy") || lower.contains("manjaro") {
                    return ("arch".to_string(), "sudo pacman -S nvidia cuda".to_string());
                } else if lower.contains("ubuntu") || lower.contains("debian") || lower.contains("pop") || lower.contains("mint") {
                    return (
                        "ubuntu".to_string(),
                        "sudo apt update && sudo apt install nvidia-driver-535 nvidia-cuda-toolkit".to_string(),
                    );
                } else if lower.contains("fedora") || lower.contains("rhel") || lower.contains("centos") {
                    return (
                        "fedora".to_string(),
                        "sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda".to_string(),
                    );
                } else if lower.contains("suse") {
                    return (
                        "opensuse".to_string(),
                        "sudo zypper install nvidia-open-driver-G06-signed-kmp-default cuda".to_string(),
                    );
                }
            }
            (
                "linux".to_string(),
                "Install NVIDIA proprietary drivers and CUDA toolkit from your distribution package manager or https://developer.nvidia.com/cuda-downloads".to_string(),
            )
        }
        #[cfg(target_os = "windows")]
        {
            (
                "windows".to_string(),
                "Download and install NVIDIA GPU Drivers and CUDA Toolkit from https://developer.nvidia.com/cuda-downloads".to_string(),
            )
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            (
                "other".to_string(),
                "Install NVIDIA CUDA toolkit from https://developer.nvidia.com/cuda-downloads".to_string(),
            )
        }
    }
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda(id) => write!(f, "cuda:{}", id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type_parsing() {
        assert_eq!(DeviceType::from_str_name("cuda"), DeviceType::Cuda(0));
        assert_eq!(DeviceType::from_str_name("GPU"), DeviceType::Cuda(0));
        assert_eq!(DeviceType::from_str_name("cpu"), DeviceType::Cpu);
        assert_eq!(DeviceType::from_str_name("auto"), DeviceType::Auto);
    }

    #[test]
    fn test_device_type_resolve() {
        assert_eq!(DeviceType::Cpu.resolve(), DeviceType::Cpu);
        assert_eq!(DeviceType::Cuda(0).resolve(), DeviceType::Cuda(0));
        let resolved = DeviceType::Auto.resolve();
        assert!(resolved == DeviceType::Cpu || matches!(resolved, DeviceType::Cuda(_)));
    }

    #[test]
    fn test_cuda_status_detect() {
        let status = CudaStatus::detect();
        match status {
            CudaStatus::Available { device_count } => {
                assert!(device_count >= 1);
            }
            CudaStatus::GpuDetectedDriverMissing { distro_id, install_command } => {
                assert!(!distro_id.is_empty());
                assert!(!install_command.is_empty());
            }
            CudaStatus::NoGpuDetected => {
                // Expected on systems without NVIDIA GPU
            }
        }
    }
}
