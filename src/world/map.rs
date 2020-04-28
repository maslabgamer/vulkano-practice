use std::io::{BufReader, ErrorKind, BufRead};
use std::fs::File;
use std::collections::HashMap;
use crate::world::chunk::Chunk;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use std::sync::Arc;
use vulkano::memory::pool::{PotentialDedicatedAllocation, StdMemoryPoolAlloc};
use vulkano::device::Device;
use crate::world::vertex::Vertex;
use std::time::Instant;

#[derive(Debug)]
pub struct Map{
    pub spawn_location: [f32; 3],
    pub chunks: HashMap<[i32; 3], Chunk>,
}

impl Map {
    pub fn load_from_file(filename: &str, world_scale: f32) -> Result<Map, &'static str> {
        let f = match File::open(filename) {
            Ok(file) => file,
            Err(error) => return match error.kind() {
                ErrorKind::NotFound => Err("Map file not found."),
                _ => Err("Could not open world file."),
            },
        };
        let file = BufReader::new(&f);

        let mut lines = file.lines();

        // Spawn location is first line in file. Parse that first
        let spawn_location = match lines.next() {
            None => return Err("Problem reading world file."),
            Some(spawn_location) => {
                match spawn_location {
                    Ok(parse_spawn_coordinates) => Map::parse_as_coordinates(&parse_spawn_coordinates),
                    Err(_) => return Err("There was a problem reading the world file."),
                }
            },
        };

        // Each chunk is 32x32
        // Chunk start denoted by line starting with "c" and a set of coordinates
        // that mark the center of the chunk
        // Following coordinates values are actually attributes for each
        let mut chunks: HashMap<[i32; 3], Chunk> = HashMap::new();
        let mut chunk_coords: Option<[i32; 3]> = None;
        for line in lines.into_iter() {
            if let Ok(line) = line {
                let line = Map::strip_comments(&line);
                if line.starts_with("c") {
                    let mut line = line.split_ascii_whitespace();
                    line.next();
                    let chunk_location: Vec<i32> = line.into_iter().map(|el| el.parse::<i32>().unwrap()).collect();
                    let chunk_location = [chunk_location[0], chunk_location[1], chunk_location[2]];

                    chunk_coords = Some(chunk_location);
                } else {
                    match chunk_coords {
                        Some(chunk_coords) => {
                            let line: Vec<u8> = line.split_ascii_whitespace()
                                .map(|el| el.parse::<u8>().unwrap())
                                .collect();

                            let new_chunk = Chunk::new(chunk_coords, &line, world_scale);
                            chunks.insert(chunk_coords, new_chunk);
                        },
                        None => return Err("Could not parse world file due to formatting!"),
                    }
                }
            }
        }

        Ok(Map {
            spawn_location,
            chunks
        })
    }

    fn parse_as_coordinates(line: &str) -> [f32; 3] {
        let line = Map::strip_comments(line);
        let line: Vec<f32> = line.split_ascii_whitespace()
            .map(|e| e.parse::<f32>().unwrap()).collect();
        [line[0], line[1], line[2]]
    }

    fn strip_comments(line: &str) -> &str {
        line.split("#").next().unwrap().trim()
    }

    // Note: This renders everything
    // May want to play around with more efficiently built algorithms later
    pub fn render_map(&self, device: &Arc<Device>) -> (Vec<Arc<CpuAccessibleBuffer<[Vertex], PotentialDedicatedAllocation<StdMemoryPoolAlloc>>>>, Vec<Arc<CpuAccessibleBuffer<[u32], PotentialDedicatedAllocation<StdMemoryPoolAlloc>>>>) {
        let mut vertex_buffers = vec![];
        let mut index_buffers = vec![];

        let start = Instant::now();
        for chunk in self.chunks.values() {
            let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, chunk.chunk_vertices.iter().cloned()).unwrap();
            vertex_buffers.push(vertex_buffer);

            let index_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, chunk.get_indices().iter().cloned()).unwrap();
            index_buffers.push(index_buffer);
        }
        println!("Marshalling took {:?}", start.elapsed());

        (vertex_buffers, index_buffers)
    }
}