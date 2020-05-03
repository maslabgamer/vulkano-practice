use cgmath::Point3;

pub(crate) struct Player {
    pub(crate) location: Point3<f32>,
    // pub(crate) previous_location: Point3<f32>,
    pub(crate) yaw: f32,
    pub(crate) pitch: f32,
    pub(crate) collision_detection_distance: f32
}

impl Player {
    pub fn new(spawn_location: Point3<f32>, yaw: f32, pitch: f32) -> Player {
        Player {
            location: spawn_location,
            // previous_location: spawn_location,
            yaw,
            pitch,
            collision_detection_distance: 1.0
        }
    }

    pub fn move_up(&mut self, units: f32) {
        self.location.y += units;
    }

    pub fn move_down(&mut self, units: f32) {
        self.location.y -= units;
    }
}