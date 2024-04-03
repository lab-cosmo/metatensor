//! utility functions to run metatensor-torch tests from Cargo
//! (used in `run-torch-tests.rs` and `torch-install-check.rs`)

#![allow(clippy::needless_return)]
#![allow(dead_code)]

use std::path::PathBuf;
use std::process::Command;

#[path = "../../../metatensor-core/tests/utils/mod.rs"]
mod core_utils;

pub use core_utils::cmake_build;
pub use core_utils::cmake_config;
pub use core_utils::ctest;


/// Find the path to the python/python3 binary on the user system
fn find_python() -> PathBuf {
    if let Ok(python) = which::which("python") {
        let output = Command::new(&python)
            .arg("-c")
            .arg("import sys; print(sys.version_info.major)")
            .output()
            .expect("could not run python");

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);

            if stdout.trim() == "3" {
                // we found Python 3
                return python;
            }
        }
    }

    // try python3
    let python = which::which("python3").expect("could not find python");
    let output = Command::new(&python)
        .arg("-c")
        .arg("import sys; print(sys.version_info.major)")
        .output()
        .expect("could not run python");

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.trim() == "3" {
            // we found Python 3
            return python;
        }
    }

    panic!("could not find Python 3")
}


/// Download PyTorch in a fresh Python virtualenv, and return the
/// CMAKE_PREFIX_PATH for the corresponding libtorch
pub fn setup_pytorch(build_dir: PathBuf) -> PathBuf {
    // create virtual env
    let status = Command::new(find_python())
        .arg("-m")
        .arg("venv")
        .arg(&build_dir)
        .status()
        .expect("failed to run python");
    assert!(status.success(), "failed to run `python -m venv`");

    // Use the virtual env python to update pip and install torch
    let mut python = build_dir;
    if cfg!(target_os = "windows") {
        python.extend(["Scripts", "python.exe"]);
    } else {
        python.extend(["bin", "python"]);
    }

    let status = Command::new(&python)
        .arg("-m")
        .arg("pip")
        .arg("install")
        .arg("--upgrade")
        .arg("pip")
        .status()
        .expect("failed to run python");
    assert!(status.success(), "failed to run `python -m pip install --upgrade pip`");

    let torch_version = std::env::var("METATENSOR_TESTS_TORCH_VERSION").unwrap_or("2.2.*".into());
    let status = Command::new(&python)
        .arg("-m")
        .arg("pip")
        .arg("install")
        .arg(format!("torch == {}", torch_version))
        .status()
        .expect("failed to run python");
    assert!(status.success(), "failed to run `python -m pip install torch`");

    let output = Command::new(&python)
        .arg("-c")
        .arg("import torch; print(torch.utils.cmake_prefix_path)")
        .output()
        .expect("failed to run python");
    assert!(output.status.success(), "failed to get torch cmake prefix");

    let stdout = String::from_utf8_lossy(&output.stdout);

    let prefix = PathBuf::from(stdout.trim());
    if !prefix.exists() {
        panic!("'torch.utils.cmake_prefix' at '{}' does not exists", prefix.display());
    }

    return prefix;
}

/// Build metatensor-core in `build_dir`, and return the installation prefix
pub fn setup_metatensor(build_dir: PathBuf) -> PathBuf {
    let mut metatensor_source_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    metatensor_source_dir.extend(["..", "metatensor-core"]);

    // configure cmake for metatensor
    let mut cmake_config = cmake_config(&metatensor_source_dir, &build_dir);

    let install_prefix = build_dir.join("usr");
    cmake_config.arg(format!("-DCMAKE_INSTALL_PREFIX={}", install_prefix.display()));

    let status = cmake_config.status().expect("could not run cmake");
    assert!(status.success(), "failed to run metatensor cmake configuration");

    // build and install metatensor
    let mut cmake_build = cmake_build(&build_dir);
    cmake_build.arg("--target");
    cmake_build.arg("install");

    let status = cmake_build.status().expect("could not run cmake");
    assert!(status.success(), "failed to run metatensor cmake build");

    return install_prefix;
}
