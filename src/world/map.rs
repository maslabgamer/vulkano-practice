use std::fs::File;
use std::io::{BufRead, BufReader, ErrorKind};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::options::InternalConfig;
use crate::world::chunk::Chunk;
use cgmath::Point3;
use dashmap::DashMap;
use vulkano::device::Device;
use vulkano::pipeline::GraphicsPipelineAbstract;

pub struct Map {
    pub spawn_location: [f32; 3],
    pub chunks: Arc<DashMap<[i32; 3], Chunk>>,
}

impl Map {
    pub fn load_from_file(
        device: Arc<Device>,
        filename: &str,
        world_scale: f32,
        internal_config: &Arc<InternalConfig>,
        default_shader: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    ) -> Result<Map, &'static str> {
        let f = match File::open(filename) {
            Ok(file) => file,
            Err(error) => {
                return match error.kind() {
                    ErrorKind::NotFound => Err("Map file not found."),
                    _ => Err("Could not open world file."),
                }
            }
        };
        let file = BufReader::new(&f);

        let mut lines = file.lines();

        // Spawn location is first line in file. Parse that first
        let spawn_location = match lines.next() {
            None => return Err("Problem reading world file."),
            Some(spawn_location) => match spawn_location {
                Ok(parse_spawn_coordinates) => Map::parse_as_coordinates(&parse_spawn_coordinates),
                Err(_) => return Err("There was a problem reading the world file."),
            },
        };

        let mut handles = vec![];

        // Each chunk is 32x32
        // Chunk start denoted by line starting with "c" and a set of coordinates
        // that mark the center of the chunk
        // Following coordinates values are actually attributes for each
        let chunks: Arc<DashMap<[i32; 3], Chunk>> = Arc::new(DashMap::new());
        let chunk_centers = Arc::new(Mutex::new(vec![]));

        let mut chunk_coords: Option<[i32; 3]> = None;
        for line in lines.into_iter() {
            if let Ok(line) = line {
                let line = Map::strip_comments(&line).to_string();
                if line.starts_with("c") {
                    let mut line = line.split_ascii_whitespace();
                    line.next();
                    let chunk_location: Vec<i32> = line
                        .into_iter()
                        .map(|el| el.parse::<i32>().unwrap())
                        .collect();
                    let chunk_location = [chunk_location[0], chunk_location[1], chunk_location[2]];

                    chunk_coords = Some(chunk_location);
                } else {
                    match chunk_coords {
                        Some(chunk_coords) => {
                            let chunks = Arc::clone(&chunks);
                            let chunk_centers = Arc::clone(&chunk_centers);
                            let internal_config = Arc::clone(&internal_config);
                            let device = Arc::clone(&device);
                            let default_shader = Arc::clone(&default_shader);
                            let handle = thread::spawn(move || {
                                let line: Vec<u8> = line
                                    .split_ascii_whitespace()
                                    .map(|el| el.parse::<u8>().unwrap())
                                    .collect();

                                let new_chunk = Chunk::new(
                                    device,
                                    chunk_coords,
                                    &line,
                                    world_scale,
                                    &internal_config,
                                    default_shader.clone(),
                                );
                                let new_chunk_center = new_chunk.location.clone();
                                chunks.insert(chunk_coords, new_chunk);
                                let mut chunk_center = chunk_centers.lock().unwrap();
                                chunk_center.push(new_chunk_center);
                            });
                            handles.push(handle);
                        }
                        None => return Err("Could not parse world file due to formatting!"),
                    }
                }
            }
        }

        for handle in handles {
            handle.join().unwrap();
        }

        Ok(Map {
            spawn_location,
            chunks,
        })
    }

    fn parse_as_coordinates(line: &str) -> [f32; 3] {
        let line = Map::strip_comments(line);
        let line: Vec<f32> = line
            .split_ascii_whitespace()
            .map(|e| e.parse::<f32>().unwrap())
            .collect();
        [line[0], line[1], line[2]]
    }

    fn strip_comments(line: &str) -> &str {
        line.split("#").next().unwrap().trim()
    }

    pub fn spawn_location_as_point(&self) -> Point3<f32> {
        Point3::new(
            self.spawn_location[0],
            self.spawn_location[1],
            self.spawn_location[2],
        )
    }
}
