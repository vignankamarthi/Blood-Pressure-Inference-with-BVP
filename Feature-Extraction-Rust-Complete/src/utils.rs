use anyhow::Result;
use std::fs;
use std::path::Path;

use crate::types::ExtractionCheckpoint;

/// Atomic file write: write to .tmp, then rename.
/// Prevents corruption if process is killed mid-write.
pub fn atomic_write(path: &Path, content: &str) -> Result<()> {
    let tmp_path = path.with_extension("tmp");
    fs::write(&tmp_path, content)?;
    fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Save checkpoint state atomically.
pub fn save_checkpoint(path: &Path, checkpoint: &ExtractionCheckpoint) -> Result<()> {
    let json = serde_json::to_string_pretty(checkpoint)?;
    atomic_write(path, &json)
}

/// Load checkpoint if it exists.
pub fn load_checkpoint(path: &Path) -> Result<Option<ExtractionCheckpoint>> {
    if path.exists() {
        let content = fs::read_to_string(path)?;
        let checkpoint: ExtractionCheckpoint = serde_json::from_str(&content)?;
        Ok(Some(checkpoint))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn test_atomic_write() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.json");
        atomic_write(&path, r#"{"key": "value"}"#).unwrap();
        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, r#"{"key": "value"}"#);
        // tmp file should not exist
        assert!(!path.with_extension("tmp").exists());
    }

    #[test]
    fn test_checkpoint_save_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("checkpoint.json");

        let mut checkpoint = ExtractionCheckpoint::new(100);
        checkpoint.mark_completed("subject_001".to_string());
        checkpoint.mark_completed("subject_002".to_string());
        checkpoint.mark_failed("subject_003".to_string(), "NaN threshold exceeded".to_string());

        save_checkpoint(&path, &checkpoint).unwrap();
        let loaded = load_checkpoint(&path).unwrap().unwrap();

        assert_eq!(loaded.completed_subjects.len(), 2);
        assert_eq!(loaded.failed_subjects.len(), 1);
        assert_eq!(loaded.total_subjects, 100);
        assert!(loaded.is_completed("subject_001"));
        assert!(!loaded.is_completed("subject_999"));
    }

    #[test]
    fn test_load_checkpoint_nonexistent() {
        let result = load_checkpoint(Path::new("/nonexistent/checkpoint.json")).unwrap();
        assert!(result.is_none());
    }
}
