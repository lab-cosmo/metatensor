#![allow(dead_code)]
#![allow(clippy::needless_return)]

use std::path::Path;
use std::process::Command;

fn build_type() -> &'static str {
    // assume that debug assertion means that we are building the code in
    // debug mode, even if that could be not true in some cases
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

pub fn cmake_config(source_dir: &Path, build_dir: &Path) -> Command {
    let cmake = which::which("cmake").expect("could not find cmake");

    let mut cmake_config = Command::new(cmake);
    cmake_config.current_dir(build_dir);
    cmake_config.arg(source_dir);
    cmake_config.arg("--no-warn-unused-cli");
    cmake_config.arg(format!("-DCMAKE_BUILD_TYPE={}", build_type()));

    // the cargo executable currently running
    let cargo_exe = std::env::var("CARGO").expect("CARGO env var is not set");
    cmake_config.arg(format!("-DCARGO_EXE={}", cargo_exe));

    return cmake_config;
}

pub fn cmake_build(build_dir: &Path) -> Command {
    let cmake = which::which("cmake").expect("could not find cmake");

    let mut cmake_build = Command::new(cmake);
    cmake_build.current_dir(build_dir);
    cmake_build.arg("--build");
    cmake_build.arg(".");
    cmake_build.arg("--parallel");
    cmake_build.arg("--config");
    cmake_build.arg(build_type());

    return cmake_build;
}


pub fn ctest(build_dir: &Path) -> Command {
    let ctest = which::which("ctest").expect("could not find ctest");

    let mut ctest = Command::new(ctest);
    ctest.current_dir(build_dir);
    ctest.arg("--output-on-failure");
    ctest.arg("--build-config");
    ctest.arg(build_type());

    return ctest
}
