use crate::options::InternalConfig;
use crate::world::cube::Cube;
use crate::world::vertex::Vertex;
use cgmath::Point3;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::device::Device;
use vulkano::pipeline::GraphicsPipelineAbstract;

const CUBES_PER_SIDE: u32 = 32;
const VERTICES_PER_SIDE: u32 = CUBES_PER_SIDE + 1;
const OFFSET_MULTIPLIER: i32 = 640;
const DISTANCE_FROM_CENTER: i32 = 20 * 16; // Width of cube times number of cubes from center

pub struct Chunk {
    pub chunk_attrs: Vec<u8>,
    pub chunk_vertices: Vec<Vertex>,
    pub location: Point3<f32>,
    // cubes: Vec<Cube>,
    pub(crate) collision_detection_distance: f32,
    pub vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    pub index_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
    pub shader_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
}

impl Chunk {
    pub fn new(
        device: Arc<Device>,
        chunk_coordinates: [i32; 3],
        chunk_attrs: &Vec<u8>,
        world_scale: f32,
        internal_config: &Arc<InternalConfig>,
        default_shader: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    ) -> Chunk {
        let x_offset = chunk_coordinates[0] * OFFSET_MULTIPLIER;
        let y_offset = chunk_coordinates[1] * OFFSET_MULTIPLIER;
        let z_offset = chunk_coordinates[2] * OFFSET_MULTIPLIER;

        let mut chunk_vertices: Vec<Vertex> = vec![];
        for y_vert in (-DISTANCE_FROM_CENTER..=DISTANCE_FROM_CENTER).step_by(20) {
            for z_vert in (-DISTANCE_FROM_CENTER..=DISTANCE_FROM_CENTER).step_by(20) {
                for x_vert in (-DISTANCE_FROM_CENTER..=DISTANCE_FROM_CENTER).step_by(20) {
                    chunk_vertices.push(Vertex {
                        position: [
                            (x_vert + x_offset) as f32,
                            (y_vert + y_offset) as f32,
                            (z_vert + z_offset) as f32,
                        ],
                        normal: [0.0, 0.0, 0.0],
                    })
                }
            }
        }

        // Now we have vertices, figure out triangles
        // For triangles with two vertices on top, the third vertex will be the next row down with the
        // same x coordinate as the left hand vertex
        // For triangles with two vertices on bottom, the third vertex will be the row up with the
        // same x coordinate as the right hand vertex
        // Remember to keep vertices logged clockwise consistently!
        // There will be 32 triangles of each orientation on each row
        let mut cubes: Vec<Cube> = vec![];
        // For now we're just figuring out the "top" surface
        for tri_layer_idx in 0..CUBES_PER_SIDE {
            for tri_row_idx in 0..CUBES_PER_SIDE {
                for tri_col_idx in 0..CUBES_PER_SIDE {
                    let chunk_val = chunk_attrs[(CUBES_PER_SIDE * CUBES_PER_SIDE * tri_layer_idx
                        + (tri_col_idx + (CUBES_PER_SIDE * tri_row_idx)))
                        as usize];
                    // Get indices for current square's vertices on top
                    // first triangle will be first, second and third vertices
                    // Second triangle will be fourth, third, and second vertices
                    let first_idx = VERTICES_PER_SIDE * VERTICES_PER_SIDE * tri_layer_idx
                        + (tri_col_idx + (VERTICES_PER_SIDE * tri_row_idx));
                    let second_idx = VERTICES_PER_SIDE * VERTICES_PER_SIDE * tri_layer_idx
                        + (tri_col_idx + 1 + (VERTICES_PER_SIDE * tri_row_idx));
                    let third_idx = VERTICES_PER_SIDE * VERTICES_PER_SIDE * tri_layer_idx
                        + (tri_col_idx + (VERTICES_PER_SIDE * (tri_row_idx + 1)));
                    let fourth_idx = VERTICES_PER_SIDE * VERTICES_PER_SIDE * tri_layer_idx
                        + (tri_col_idx + 1 + (VERTICES_PER_SIDE * (tri_row_idx + 1)));

                    // Next two indices will let us calculate the "left" triangles
                    // First triangle will be first, third, fifth
                    // Second triangle will be sixth, fifth, third
                    let fifth_idx = VERTICES_PER_SIDE * VERTICES_PER_SIDE * (tri_layer_idx + 1)
                        + (tri_col_idx + (VERTICES_PER_SIDE * tri_row_idx));
                    let sixth_idx = VERTICES_PER_SIDE * VERTICES_PER_SIDE * (tri_layer_idx + 1)
                        + (tri_col_idx + (VERTICES_PER_SIDE * (tri_row_idx + 1)));

                    // Next two indices will let us calculate the "right" triangles. As this will give
                    // Us all eight vertices we can then easily get the "back", "front", and "bottom"
                    // First triangle will be fourth, second, eighth
                    // Second triangle will be seventh, eighth, second
                    let seventh_idx = VERTICES_PER_SIDE * VERTICES_PER_SIDE * (tri_layer_idx + 1)
                        + (tri_col_idx + 1 + (VERTICES_PER_SIDE * tri_row_idx));
                    let eighth_idx = VERTICES_PER_SIDE * VERTICES_PER_SIDE * (tri_layer_idx + 1)
                        + (tri_col_idx + 1 + (VERTICES_PER_SIDE * (tri_row_idx + 1)));

                    cubes.push(Cube {
                        attribute: chunk_val,
                        vertex_indices: vec![
                            // Top triangles
                            first_idx,
                            second_idx,
                            third_idx,
                            fourth_idx,
                            third_idx,
                            second_idx,
                            // Left triangles
                            first_idx,
                            third_idx,
                            fifth_idx,
                            sixth_idx,
                            fifth_idx,
                            third_idx,
                            // Right triangles
                            fourth_idx,
                            second_idx,
                            eighth_idx,
                            seventh_idx,
                            eighth_idx,
                            second_idx,
                            // Back triangles
                            first_idx,
                            fifth_idx,
                            second_idx,
                            seventh_idx,
                            second_idx,
                            fifth_idx,
                            // Front triangles
                            fourth_idx,
                            eighth_idx,
                            third_idx,
                            sixth_idx,
                            third_idx,
                            eighth_idx,
                            // Bottom triangles
                            seventh_idx,
                            fifth_idx,
                            eighth_idx,
                            sixth_idx,
                            eighth_idx,
                            fifth_idx,
                        ],
                    });
                }
            }
        }

        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            chunk_vertices.iter().cloned(),
        )
        .expect("Could not create chunk vertex buffer");

        let indices: Vec<u32> = cubes
            .iter()
            .map(|cube| cube.render_vertices())
            .filter_map(|cube| cube)
            .flatten()
            .map(|cube| *cube)
            .collect();

        let index_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            indices.iter().cloned(),
        )
        .expect("Could not create chunk index buffer");

        Chunk {
            chunk_attrs: chunk_attrs.clone(),
            // cubes, // Commenting out for now as will likely usue later
            chunk_vertices,
            location: Point3::new(
                x_offset as f32 * world_scale,
                y_offset as f32 * world_scale,
                z_offset as f32 * world_scale,
            ),
            collision_detection_distance: DISTANCE_FROM_CENTER as f32
                * internal_config.engine.scale
                * 2.0,
            vertex_buffer,
            index_buffer,
            shader_pipeline: default_shader,
        }
    }

    pub fn set_shader(&mut self, shader: Arc<dyn GraphicsPipelineAbstract + Send + Sync>) {
        self.shader_pipeline = shader;
    }
}
