//! System hardware, GPU detection, and CUDA doctor command.

use anyhow::Result;
use graphite::vector::device::CudaStatus;
use inquire::Confirm;

use crate::args::DoctorArgs;

pub fn execute_doctor(args: &DoctorArgs) -> Result<()> {
    println!("------------------------------------------------------------");
    println!("  Graphite System & Hardware Doctor");
    println!("------------------------------------------------------------");

    println!("  Operating System:      {}", std::env::consts::OS);
    println!("  Architecture:          {}", std::env::consts::ARCH);

    let cuda_status = CudaStatus::detect();
    println!();
    match cuda_status {
        CudaStatus::Available { device_count } => {
            println!("  [OK] CUDA Acceleration: Available ({} GPU device(s) ready)", device_count);
            println!("       Graphite will automatically use GPU Tensor Cores for 10x-50x speedup.");
        }
        CudaStatus::GpuDetectedDriverMissing {
            distro_id,
            install_command,
        } => {
            println!("  [!] NVIDIA GPU Detected, but CUDA Runtime / Drivers are not active.");
            println!("      Detected distribution/system: {}", distro_id);
            println!("      Recommended installation command:");
            println!("        {}", install_command);
            println!();

            let should_install = if args.install {
                true
            } else {
                Confirm::new("Would you like to execute this installation command now?")
                    .with_default(false)
                    .prompt()
                    .unwrap_or(false)
            };

            if should_install {
                println!("\nExecuting: {}\n", install_command);
                let status = std::process::Command::new("sh")
                    .arg("-c")
                    .arg(&install_command)
                    .status();

                match status {
                    Ok(s) if s.success() => {
                        println!("\n  [OK] Installation completed successfully. Please reboot or reload drivers.");
                    }
                    Ok(s) => {
                        println!("\n  [!] Installation exited with status: {}", s);
                    }
                    Err(e) => {
                        println!("\n  [!] Failed to execute installation command: {}", e);
                    }
                }
            } else {
                println!("  [Info] Graphite will run seamlessly on CPU with AVX2/AVX-512 SIMD acceleration.");
            }
        }
        CudaStatus::NoGpuDetected => {
            println!("  [Info] GPU Acceleration: No NVIDIA GPU detected on this system.");
            println!("         Graphite is running with optimized CPU SIMD parallelism.");
        }
    }
    println!("------------------------------------------------------------");
    Ok(())
}
