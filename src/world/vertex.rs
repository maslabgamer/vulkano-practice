use std::ops::{Sub, Add};

#[derive(Debug, Default, PartialEq, Clone)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color_val: [f32;3],
}

vulkano::impl_vertex!(Vertex, position, normal, color_val);

impl Vertex {
    pub fn set_color_val(&mut self, r: f32, g: f32, b: f32) {
        self.color_val = [r, g, b];
    }
}

impl Sub for Vertex {
    type Output = Vertex;

    fn sub(self, rhs: Self) -> Self::Output {
        Vertex {
            position: [self.position[0] - rhs.position[0], self.position[1] - rhs.position[1], self.position[2] - rhs.position[2]],
            normal: [0.0; 3],
            color_val: [0.0, 1.0, 0.0]
        }
    }
}

impl Add for Vertex {
    type Output = Vertex;

    fn add(self, rhs: Self) -> Self::Output {
        Vertex {
            position: [self.position[0] + rhs.position[0], self.position[1] + rhs.position[1], self.position[2] + rhs.position[2]],
            normal: [0.0; 3],
            color_val: [0.0, 1.0, 0.0]
        }
    }
}
