use cgmath::Point3;

pub(crate) struct Player {
    pub(crate) location: Point3<f32>,
    pub(crate) yaw: f32,
    pub(crate) pitch: f32,
}