use serde::Deserialize;
use std::fs::read_to_string;

#[derive(Debug, Deserialize)]
pub struct InternalConfig {
    pub(crate) engine: Engine,
}

#[derive(Debug, Deserialize)]
pub struct Engine {
    pub scale: f32,
}

impl InternalConfig {
    pub fn load_internal_config(filename: &str) -> InternalConfig {
        let toml_str = read_to_string(filename).unwrap();
        toml::from_str(&toml_str).unwrap()
    }
}
