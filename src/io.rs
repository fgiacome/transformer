use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

/// Saves a serializable model to a file using bincode
///
/// # Arguments
/// * `model` - The model to save (must implement Serialize)
/// * `path` - Path to the output file
///
/// # Returns
/// Result with () on success or io::Error on failure
///
/// # Example
/// ```no_run
/// use transformer::{Feedforward, io};
///
/// let model = Feedforward::new(784, 128);
/// io::save(&model, "model.bin").unwrap();
/// ```
pub fn save<T: Serialize>(model: &T, path: impl AsRef<Path>) -> io::Result<()> {
    let mut file = File::create(path)?;
    let encoded = bincode::serde::encode_to_vec(model, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    file.write_all(&encoded)?;
    Ok(())
}

/// Loads a serializable model from a file using bincode
///
/// # Arguments
/// * `path` - Path to the input file
///
/// # Returns
/// Result with the deserialized model or io::Error on failure
///
/// # Example
/// ```no_run
/// use transformer::{Feedforward, io};
///
/// let model: Feedforward = io::load("model.bin").unwrap();
/// ```
pub fn load<T: for<'de> Deserialize<'de>>(path: impl AsRef<Path>) -> io::Result<T> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let (decoded, _) = bincode::serde::decode_from_slice(&buffer, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    Ok(decoded)
}
