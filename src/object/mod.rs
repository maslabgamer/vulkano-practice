use cgmath::Point3;

pub trait GameObject {
    fn collision_detection_distance(&self) -> f32;

    fn location(&self) -> Point3<f32>;

    fn distance_between(&self, other: &dyn GameObject) -> f32 {
        f32::sqrt(
            (other.location().x - self.location().x).powi(2)
                + (other.location().y - self.location().y).powi(2)
                + (other.location().z - self.location().z).powi(2),
        )
    }
}

#[macro_export]
macro_rules! game_object {
    ($($class:ident),* $(,)?) => {
        $(
            impl GameObject for $class {
                fn location(&self) -> Point3<f32> {
                    self.location
                }

                fn collision_detection_distance(&self) -> f32 {
                    self.collision_detection_distance
                }
            }
        )*
    };
}
