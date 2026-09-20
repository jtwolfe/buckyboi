//! Shared ORT session builder: intra-op cap + Level1.

use std::path::Path;
use std::sync::Once;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

/// Default 1; `BUCKYBOI_ORT_THREADS` clamped to 1..=4.
pub fn intra_threads() -> usize {
    std::env::var("BUCKYBOI_ORT_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, 4)
}

fn apply_omp(n: usize) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if std::env::var_os("OMP_NUM_THREADS").is_none() {
            // Microsoft prebuilt ORT may ignore intra-op when linked with OpenMP.
            std::env::set_var("OMP_NUM_THREADS", n.to_string());
        }
    });
}

pub fn session_from_file(path: &Path) -> Option<Session> {
    let intra = intra_threads();
    apply_omp(intra);
    Session::builder()
        .ok()?
        .with_intra_threads(intra)
        .ok()?
        .with_inter_threads(1)
        .ok()?
        .with_optimization_level(GraphOptimizationLevel::Level1)
        .ok()?
        .commit_from_file(path)
        .ok()
}
