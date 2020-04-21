use std::io::{BufReader, ErrorKind, BufRead};
use std::fs::File;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Map{
    pub spawn_location: [f32; 3],
    pub chunks: HashMap<[i32; 3], Vec<u8>>,
}

impl Map {
    pub fn load_from_file(filename: &str) -> Result<Map, &'static str> {
        let f = match File::open(filename) {
            Ok(file) => file,
            Err(error) => return match error.kind() {
                ErrorKind::NotFound => Err("Map file not found."),
                _ => Err("Could not open map file."),
            },
        };
        let file = BufReader::new(&f);

        let mut lines = file.lines();

        // Spawn location is first line in file. Parse that first
        let spawn_location = match lines.next() {
            None => return Err("Problem reading map file."),
            Some(spawn_location) => {
                match spawn_location {
                    Ok(parse_spawn_coordinates) => Map::parse_as_coordinates(&parse_spawn_coordinates),
                    Err(_) => return Err("There was a problem reading the map file."),
                }
            },
        };

        // Each chunk is 32x32
        // Chunk start denoted by line starting with "c" and a set of coordinates
        // that mark the center of the chunk
        // Following coordinates values are actually attributes for each
        let mut chunks: HashMap<[i32; 3], Vec<u8>> = HashMap::new();
        let mut chunk_coords: Option<[i32; 3]> = None;
        for line in lines.into_iter() {
            if let Ok(line) = line {
                let line = Map::strip_comments(&line);
                if line.starts_with("c") {
                    let mut line = line.split_ascii_whitespace();
                    line.next();
                    let chunk_location: Vec<i32> = line.into_iter().map(|el| el.parse::<i32>().unwrap()).collect();
                    let chunk_location = [chunk_location[0], chunk_location[1], chunk_location[2]];
                    if !chunks.contains_key(&chunk_location) {
                        chunks.insert(chunk_location, Vec::with_capacity(1024));
                    }
                    chunk_coords = Some(chunk_location);
                } else {
                    match chunk_coords {
                        None => return Err("Could not parse map file due to formatting!"),
                        Some(chunk_coords) => {
                            match chunks.get_mut(&chunk_coords) {
                                Some(chunk) => {
                                    // let line: Vec<_> = line.split("  ")
                                    //     .map(|el| el.split_ascii_whitespace()
                                    //         .map(|att| att.parse::<u8>().unwrap())
                                    //         .collect::<Vec<u8>>())
                                    //     .collect();
                                    let line: Vec<u8> = line.split_ascii_whitespace()
                                        .map(|el| el.parse::<u8>().unwrap())
                                        .collect();

                                    chunk.extend(line);
                                },
                                None => return Err("Could not parse map file due to formatting!"),
                            }
                        },
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
}