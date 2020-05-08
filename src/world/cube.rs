#[derive(Debug)]
pub(crate) struct Cube {
    pub(crate) attribute: u8,
    pub(crate) vertex_indices: Vec<u32>,
}

impl Cube {
    pub fn render_vertices(&self) -> Option<&Vec<u32>> {
        if self.attribute > 0 {
            Some(&self.vertex_indices)
        } else {
            None
        }
    }
}
