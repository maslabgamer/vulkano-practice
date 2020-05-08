use crate::vertex_shader;
use crate::world::vertex::Vertex;
use dashmap::DashMap;
use std::iter;
use std::sync::Arc;
use vulkano::device::Device;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::framebuffer::Subpass;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};

#[derive(PartialEq, Eq, Hash)]
pub enum ShaderType {
    Default,        // green
    CollisionCheck, // red
}

pub struct Shaders {
    pub shaders: DashMap<ShaderType, Arc<dyn GraphicsPipelineAbstract + Send + Sync>>,
}

impl Shaders {
    pub fn load_shaders(
        device: Arc<Device>,
        dimensions: [u32; 2],
        render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
        vs: &vertex_shader::Shader,
    ) -> Shaders {
        let gfs = green_frag_shader::Shader::load(device.clone()).unwrap();
        let rfs = red_frag_shader::Shader::load(device.clone()).unwrap();

        // let shaders_to_load = vec![(ShaderType::Default, gfs), (ShaderType::CollisionCheck, rfs)];

        let shaders: DashMap<ShaderType, Arc<dyn GraphicsPipelineAbstract + Send + Sync>> =
            DashMap::with_capacity(2);

        let pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .viewports(iter::once(Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                    depth_range: 0.0..1.0,
                }))
                .fragment_shader(gfs.main_entry_point(), ())
                .depth_stencil_simple_depth()
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(device.clone())
                .unwrap(),
        );

        shaders.insert(ShaderType::Default, pipeline);

        let pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .viewports(iter::once(Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                    depth_range: 0.0..1.0,
                }))
                .fragment_shader(rfs.main_entry_point(), ())
                .depth_stencil_simple_depth()
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(device.clone())
                .unwrap(),
        );

        shaders.insert(ShaderType::CollisionCheck, pipeline);

        Shaders { shaders }
    }
}

// Create the shaders
mod green_frag_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "resources/shaders/chunk.fs"
    }
}

mod red_frag_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "resources/shaders/red_chunk.fs"
    }
}
