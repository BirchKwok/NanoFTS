//! Durable persistence helpers (fsync + atomic rename).
//!
//! Crash-consistency pattern used across flush/compact/sidecar writes:
//! 1. Write payload to a temporary file
//! 2. `sync_all` the temp file (data + metadata)
//! 3. Atomically `rename` temp → final
//! 4. On Unix, `sync_all` the parent directory so the rename itself is durable
//!    (Windows does not expose a portable directory fsync; NTFS rename is atomic
//!    within a volume, so we skip this step there).

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

/// Sync the parent directory of `path` so a preceding rename is durable.
///
/// On Windows this is a no-op: opening a directory handle and calling `sync_all`
/// returns `ERROR_ACCESS_DENIED` (5) on typical runners and developer machines.
pub fn sync_parent_dir(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let dir = if parent.as_os_str().is_empty() {
            File::open(".")?
        } else {
            File::open(parent)?
        };
        dir.sync_all()?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

/// Atomically replace `final_path` with `tmp_path`, then fsync the parent directory.
///
/// On Windows, `rename` cannot overwrite an existing destination, so we remove
/// `final_path` first when it already exists (same volume, best-effort atomicity).
pub fn durable_rename(tmp_path: &Path, final_path: &Path) -> io::Result<()> {
    #[cfg(windows)]
    {
        if final_path.exists() {
            fs::remove_file(final_path)?;
        }
    }
    fs::rename(tmp_path, final_path)?;
    sync_parent_dir(final_path)?;
    Ok(())
}

/// Write `bytes` to `final_path` via temp file + fsync + durable rename.
pub fn durable_write(final_path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut tmp = final_path.as_os_str().to_os_string();
    tmp.push(".tmp");
    let tmp_path = std::path::PathBuf::from(tmp);

    {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    durable_rename(&tmp_path, final_path)
}

/// Sync an already-open file to durable storage.
pub fn sync_file(file: &File) -> io::Result<()> {
    file.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn durable_write_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("payload.bin");
        durable_write(&path, b"hello-durable").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"hello-durable");
    }

    #[test]
    fn durable_rename_replaces_target() {
        let dir = tempdir().unwrap();
        let final_path = dir.path().join("final.dat");
        let tmp_path = dir.path().join("final.dat.tmp");
        {
            let mut f = File::create(&tmp_path).unwrap();
            f.write_all(b"new").unwrap();
            f.sync_all().unwrap();
        }
        // Pre-create destination so Windows takes the remove-then-rename path.
        fs::write(&final_path, b"old").unwrap();
        durable_rename(&tmp_path, &final_path).unwrap();
        assert_eq!(fs::read(&final_path).unwrap(), b"new");
        assert!(!tmp_path.exists());
    }
}
