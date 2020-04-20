use std::ops::Sub;

#[derive(Debug, Default, PartialEq, Clone)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

vulkano::impl_vertex!(Vertex, position, normal);

impl Sub for Vertex {
    type Output = Vertex;

    fn sub(self, rhs: Self) -> Self::Output {
        Vertex {
            position: [self.position[0] - rhs.position[0], self.position[1] - rhs.position[1], self.position[2] - rhs.position[2]],
            normal: [0.0, 1.0, 0.0]
        }
    }
}

impl Vertex {
    pub fn cross_product(&self, v: &Vertex) -> Vertex {
        Vertex {
            position: [
                self.position[1] * v.position[2] - self.position[2] * v.position[1],
                self.position[2] * v.position[0] - self.position[0] * v.position[2],
                self.position[0] * v.position[1] - self.position[1] * v.position[0],
            ],
            normal: [0.0, 1.0, 0.0]
        }
    }
}
