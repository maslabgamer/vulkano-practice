use cgmath::Point3;

pub(crate) struct Player {
    pub(crate) location: Point3<f32>,
    pub(crate) yaw: f32,
    pub(crate) pitch: f32,
}

impl Player {
    pub fn move_up(&mut self, units: f32) {
        self.location.y += units;
    }

    pub fn move_down(&mut self, units: f32) {
        self.location.y -= units;
    }
}