//! Link directives for the optional `blas` feature.
//!
//! This crate **never vendors, downloads, or builds a BLAS**.  It links one
//! that already exists on the system, which is why there is no `*-src`
//! dependency: those crates exist to *supply* a BLAS, and carry a compiler and
//! an HTTP stack to do it.  All that is actually needed to call into an
//! installed BLAS is a link directive, which is what this script emits.
//!
//! Resolution order:
//!
//! 1. `FAST_HNSW_BLAS_LIB` — explicit override, always wins.  Accepts a plain
//!    library name (`openblas`, `mkl_rt`, `blis`), a `framework=Name` form for
//!    Apple frameworks, or several entries separated by commas.
//! 2. **Apple targets** (macOS, iOS, tvOS, watchOS, visionOS) link the
//!    Accelerate framework, which ships in every Apple SDK.  Nothing to
//!    install, and it is the only practical option on iOS, where third-party
//!    native libraries cannot be installed system-wide.
//! 3. `pkg-config`, invoked as a subprocess so it is not a build dependency.
//!    Probes `openblas` then `cblas`.
//! 4. A per-target default, with a diagnostic describing how to override it.
//!
//! `FAST_HNSW_BLAS_LIB_DIR` adds a link-search path in every case.

use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for var in [
        "FAST_HNSW_BLAS_LIB",
        "FAST_HNSW_BLAS_LIB_DIR",
        "FAST_HNSW_BLAS_STATIC",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    // Cargo sets this only when the feature is enabled; without it the crate
    // has no BLAS calls to resolve and must not emit any link directives.
    if env::var_os("CARGO_FEATURE_BLAS").is_none() {
        return;
    }

    if let Some(dir) = env::var_os("FAST_HNSW_BLAS_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", dir.to_string_lossy());
    }

    let kind = if env::var_os("FAST_HNSW_BLAS_STATIC").is_some() {
        "static"
    } else {
        "dylib"
    };

    // 1 · Explicit override.
    if let Ok(spec) = env::var("FAST_HNSW_BLAS_LIB") {
        for entry in spec.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            match entry.strip_prefix("framework=") {
                Some(framework) => println!("cargo:rustc-link-lib=framework={framework}"),
                None => println!("cargo:rustc-link-lib={kind}={entry}"),
            }
        }
        return;
    }

    // 2 · Apple platforms: Accelerate is part of the SDK on every one of them,
    //     including iOS where nothing else can realistically be linked.
    if env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        return;
    }

    // 3 · pkg-config, as a subprocess. Preferring it over a hardcoded name
    //     picks up non-standard prefixes (Homebrew on Linux, Nix, Spack).
    //
    //     Only CBLAS-providing packages are probed. Netlib's `blas` is
    //     Fortran-only and does *not* export the `cblas_*` symbols this crate
    //     calls, so probing it would "succeed" and then fail at link time.
    for package in ["openblas", "cblas"] {
        if probe_pkg_config(package) {
            return;
        }
    }

    // 4 · Fall back to a per-target guess and say what to set if it is wrong.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "windows" {
        println!(
            "cargo:warning=fast-hnsw: no BLAS found via pkg-config. Windows has no \
             system BLAS; install one (vcpkg install openblas, or Intel oneAPI MKL) \
             and set FAST_HNSW_BLAS_LIB=openblas (or mkl_rt) plus \
             FAST_HNSW_BLAS_LIB_DIR=<path>. Defaulting to `openblas`."
        );
    } else {
        println!(
            "cargo:warning=fast-hnsw: pkg-config found no openblas/cblas. Install one \
             (apt install libopenblas-dev, dnf install openblas-devel) or set \
             FAST_HNSW_BLAS_LIB / FAST_HNSW_BLAS_LIB_DIR. Defaulting to `openblas`."
        );
    }
    println!("cargo:rustc-link-lib={kind}=openblas");
}

/// Ask `pkg-config` for `package` and translate its answer into cargo
/// directives.  Returns `false` if pkg-config is missing or does not know it.
fn probe_pkg_config(package: &str) -> bool {
    // Cross-compiling with the host's pkg-config would find host libraries, so
    // honour the usual per-target wrapper convention and otherwise stay out of
    // the way.
    let target = env::var("TARGET").unwrap_or_default();
    let host = env::var("HOST").unwrap_or_default();
    let tool = env::var(format!("PKG_CONFIG_{}", target.replace('-', "_")))
        .or_else(|_| env::var("PKG_CONFIG"))
        .unwrap_or_else(|_| "pkg-config".to_string());
    if target != host && env::var_os("PKG_CONFIG_ALLOW_CROSS").is_none() {
        return false;
    }

    let Ok(output) = Command::new(&tool).args(["--libs", package]).output() else {
        return false; // pkg-config not installed
    };
    if !output.status.success() {
        return false; // package unknown
    }

    let flags = String::from_utf8_lossy(&output.stdout);
    let mut linked_any = false;
    for flag in flags.split_whitespace() {
        if let Some(path) = flag.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={path}");
        } else if let Some(lib) = flag.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={lib}");
            linked_any = true;
        }
    }
    linked_any
}
