#[derive(Debug)]
pub(crate) struct Cube {
    pub(crate) attribute: u8,
    // pub(crate) vertex_indices: [u32; 36],
    pub(crate) vertex_indices: Vec<u32>,
}

impl Cube {
    pub fn render_vertices(&self) -> Vec<u32> {
        if self.attribute > 0 {
            self.vertex_indices.clone()
        } else {
            vec![]
        }
    }
}