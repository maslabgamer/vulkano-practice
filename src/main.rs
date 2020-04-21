mod map;

use crate::map::map::Map;
use crate::map::vertex::Vertex;
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano_win::VkSurfaceBuild;
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::{WindowBuilder, Window};
use vulkano::device::{DeviceExtensions, Device};
use vulkano::format::Format;
use vulkano::swapchain::{Swapchain, SurfaceTransform, PresentMode, FullscreenExclusive, ColorSpace, SwapchainCreationError, AcquireError};
use vulkano::buffer::{CpuAccessibleBuffer, BufferUsage, CpuBufferPool};
use std::sync::Arc;
use vulkano::image::{SwapchainImage, AttachmentImage};
use vulkano::pipeline::{GraphicsPipelineAbstract, GraphicsPipeline};
use vulkano::framebuffer::{FramebufferAbstract, RenderPassAbstract, Framebuffer, Subpass};
use vulkano::pipeline::viewport::Viewport;
use std::iter;
use vulkano::sync;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::swapchain;
use winit::event::{WindowEvent, Event};
use cgmath::{Matrix3, Rad, Matrix4, Point3, Vector3};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use winit::event::VirtualKeyCode::{Space, LShift, Escape};

const CUBES_PER_SIDE: usize = 32;
const VERTICES_PER_SIDE: usize = CUBES_PER_SIDE + 1;

fn main() {
    // Load basic map
    let map = match Map::load_from_file("resources/map_file.map") {
        Ok(map) => map,
        Err(e) => panic!("There was a problem loading the map. Can't continue: {}", e),
    };

    println!("Map chunk at [0, 0, 0] has {} elements", map.chunks.get(&[0, 0, 0]).unwrap().len());
    let distance_from_center = 20 * 16; // Width of cube times number of cubes from center

    let mut flat_vertices: Vec<Vertex> = vec![];
    for y_vert in (-distance_from_center..=distance_from_center).step_by(20) {
        for z_vert in (-distance_from_center..=distance_from_center).step_by(20) {
            for x_vert in (-distance_from_center..=distance_from_center).step_by(20) {
                flat_vertices.push(Vertex { position: [x_vert as f32, y_vert as f32, z_vert as f32], normal: [0.0, 0.0, 0.0] })
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
    let mut indices: Vec<u32> = Vec::new();
    // For now we're just figuring out the "top" surface
    let chunk_to_render = map.chunks.get(&[0, 0, 0]).unwrap();
    for tri_layer_idx in 0..CUBES_PER_SIDE {
        for tri_row_idx in 0..CUBES_PER_SIDE {
            for tri_col_idx in 0..CUBES_PER_SIDE {
                // println!("[{}, {}, {}] = {}", tri_layer_idx, tri_row_idx, tri_col_idx, chunk_to_render[CUBES_PER_SIDE * CUBES_PER_SIDE * tri_layer_idx + (tri_col_idx + (CUBES_PER_SIDE * tri_row_idx))]);
                let chunk_val = chunk_to_render[CUBES_PER_SIDE * CUBES_PER_SIDE * tri_layer_idx + (tri_col_idx + (CUBES_PER_SIDE * tri_row_idx))];
                if chunk_val > 0 {
                    // Get indices for current square's vertices on top
                    // first triangle will be first, second and third vertices
                    // Second triangle will be fourth, third, and second vertices
                    let first_idx: usize = VERTICES_PER_SIDE * VERTICES_PER_SIDE * tri_layer_idx + (tri_col_idx + (VERTICES_PER_SIDE * tri_row_idx));
                    let second_idx: usize = VERTICES_PER_SIDE * VERTICES_PER_SIDE * tri_layer_idx + (tri_col_idx + 1 + (VERTICES_PER_SIDE * tri_row_idx));
                    let third_idx: usize = VERTICES_PER_SIDE * VERTICES_PER_SIDE * tri_layer_idx + (tri_col_idx + (VERTICES_PER_SIDE * (tri_row_idx + 1)));
                    let fourth_idx: usize = VERTICES_PER_SIDE * VERTICES_PER_SIDE * tri_layer_idx + (tri_col_idx + 1 + (VERTICES_PER_SIDE * (tri_row_idx + 1)));

                    // Next two indices will let us calculate the "left" triangles
                    // First triangle will be first, third, fifth
                    // Second triangle will be sixth, fifth, third
                    let fifth_idx: usize = VERTICES_PER_SIDE * VERTICES_PER_SIDE * (tri_layer_idx + 1) + (tri_col_idx + (VERTICES_PER_SIDE * tri_row_idx));
                    let sixth_idx: usize = VERTICES_PER_SIDE * VERTICES_PER_SIDE * (tri_layer_idx + 1) + (tri_col_idx + (VERTICES_PER_SIDE * (tri_row_idx + 1)));

                    // Next two indices will let us calculate the "right" triangles. As this will give
                    // Us all eight vertices we can then easily get the "back", "front", and "bottom"
                    // First triangle will be fourth, second, eighth
                    // Second triangle will be seventh, eighth, second
                    let seventh_idx = VERTICES_PER_SIDE * VERTICES_PER_SIDE * (tri_layer_idx + 1) + (tri_col_idx + 1 + (VERTICES_PER_SIDE * tri_row_idx));
                    let eighth_idx = VERTICES_PER_SIDE * VERTICES_PER_SIDE * (tri_layer_idx + 1) + (tri_col_idx + 1 + (VERTICES_PER_SIDE * (tri_row_idx + 1)));

                    // Push indices for top triangles
                    indices.push(first_idx as u32);
                    indices.push(second_idx as u32);
                    indices.push(third_idx as u32);

                    indices.push(fourth_idx as u32);
                    indices.push(third_idx as u32);
                    indices.push(second_idx as u32);

                    // push indices for left triangles
                    indices.push(first_idx as u32);
                    indices.push(third_idx as u32);
                    indices.push(fifth_idx as u32);

                    indices.push(sixth_idx as u32);
                    indices.push(fifth_idx as u32);
                    indices.push(third_idx as u32);

                    // push indices for right triangles
                    indices.push(fourth_idx as u32);
                    indices.push(second_idx as u32);
                    indices.push(eighth_idx as u32);

                    indices.push(seventh_idx as u32);
                    indices.push(eighth_idx as u32);
                    indices.push(second_idx as u32);

                    // push indices for back triangles
                    // first triangle will be first, fifth, second
                    // second triangle will be seventh, second, fifth
                    indices.push(first_idx as u32);
                    indices.push(fifth_idx as u32);
                    indices.push(second_idx as u32);

                    indices.push(seventh_idx as u32);
                    indices.push(second_idx as u32);
                    indices.push(fifth_idx as u32);

                    // push indices for front triangles
                    // first triangle will be four, eight, three
                    // second triangle will be six, third, eight
                    indices.push(fourth_idx as u32);
                    indices.push(eighth_idx as u32);
                    indices.push(third_idx as u32);

                    indices.push(sixth_idx as u32);
                    indices.push(third_idx as u32);
                    indices.push(eighth_idx as u32);

                    // push indices for bottom triangles
                    // first triangle will be seven, five, eight
                    // second triangle will be six, eight, five
                    indices.push(seventh_idx as u32);
                    indices.push(fifth_idx as u32);
                    indices.push(eighth_idx as u32);

                    indices.push(sixth_idx as u32);
                    indices.push(eighth_idx as u32);
                    indices.push(fifth_idx as u32);
                }
            }
        }
    }
    // println!("indices = {:?}", indices);

    // Now let's try rendering our basic triangles
    // Get required extensions to draw window
    let required_extensions = vulkano_win::required_extensions();

    // Create vulkano instance
    let instance = Instance::new(None, &required_extensions, None).unwrap();
    // Get physical device
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    // Print the device name and type (verifies we've gotten it)
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    // Create window to actually display shit
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new().with_title("ChunkRenderer").build_vk_surface(&event_loop, instance.clone()).unwrap();

    // Pick a GPU queue to execute draw commands
    // Devices can provide multiple queues to execute commands in parallel
    let queue_family = physical.queue_families().find(|&q| {
        // Take first queue that supports drawing the window
        q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
    }).unwrap();

    // Initialize device with required parameters
    let device_ext = DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::none() };
    let (device, mut queues) = Device::new(physical, physical.supported_features(), &device_ext,
                                           [(queue_family, 0.5)].iter().cloned()).unwrap();
    // Take the first queue and throw the rest away
    let queue = queues.next().unwrap();

    // Create a swapchain
    let dimensions: [u32; 2] = surface.window().inner_size().into();
    let (mut swapchain, images) = {
        // Query the capabilities of the surface
        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;

        // Alpha mode indicates how the alpha value of the final image will behave
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        // Choosing the internal format that the images will have
        let format = caps.supported_formats[0].0;

        Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format,
                       dimensions, 1, usage, &queue, SurfaceTransform::Identity, alpha,
                       PresentMode::Fifo, FullscreenExclusive::Default, true, ColorSpace::SrgbNonLinear).unwrap()
    };

    // Create buffer of vertices
    let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, flat_vertices.iter().cloned()).unwrap();

    // Use indices to build triangles
    let index_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, indices.iter().cloned()).unwrap();

    let uniform_buffer = CpuBufferPool::<vertex_shader::ty::Data>::new(device.clone(), BufferUsage::all());

    let vs = vertex_shader::Shader::load(device.clone()).unwrap();
    let fs = frag_shader::Shader::load(device.clone()).unwrap();

    // Do render pass to describe where output of graphics pipeline will go
    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        ).unwrap()
    );

    // Need to create actual framebuffers
    let (mut pipeline, mut framebuffers) = window_size_dependent_setup(device.clone(), &vs, &fs, &images, render_pass.clone());

    // Flag swapchain recreation in case window is resized
    let mut recreate_swapchain = false;

    // Start submitting commands in loop below
    // Hold a place to store the submission of the previous frame
    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

    // let timer = Instant::now();
    let mut eye = Point3::new(0.3, 0.3, 7.5);

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event: WindowEvent::KeyboardInput { input, .. }, .. } => {
                // println!("input = {:?}", input);
                match input.virtual_keycode {
                    Some(Escape) => *control_flow = ControlFlow::Exit,
                    Some(Space) => eye.y -= 0.5,
                    Some(LShift) => eye.y += 0.5,
                    _ => (),
                }
            }
            // Event::WindowEvent { event: WindowEvent::CursorMoved { device_id, position, .. }, .. } => {
            //     println!("device_id = {:?}, position: {:?}", device_id, position);
            // }
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
                recreate_swapchain = true;
            }
            Event::RedrawEventsCleared => {
                // Clean up unused resources
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                // Whenever window resizes we need to recreate everything dependent on the window size.
                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) = match swapchain.recreate_with_dimensions(dimensions) {
                        Ok(r) => r,
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e)
                    };

                    swapchain = new_swapchain;
                    let (new_pipeline, new_framebuffers) = window_size_dependent_setup(device.clone(), &vs, &fs, &new_images, render_pass.clone());
                    pipeline = new_pipeline;
                    framebuffers = new_framebuffers;
                    recreate_swapchain = false;
                }

                let uniform_buffer_subbuffer = {
                    let rotation = Matrix3::from_angle_y(Rad(0.0));

                    let aspect_ratio = dimensions[0] as f32 / dimensions[1] as f32;
                    let proj = cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.01, 100.0);
                    let view = Matrix4::look_at(eye, Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0));
                    let scale = Matrix4::from_scale(0.01);

                    let uniform_data = vertex_shader::ty::Data {
                        world: Matrix4::from(rotation).into(),
                        view: (view * scale).into(),
                        proj: proj.into(),
                    };

                    uniform_buffer.next(uniform_data).unwrap()
                };

                let layout = pipeline.descriptor_set_layout(0).unwrap();
                let set = Arc::new(PersistentDescriptorSet::start(layout.clone())
                    .add_buffer(uniform_buffer_subbuffer).unwrap()
                    .build().unwrap()
                );

                // Have to acquire an images from the swapchain before we can draw it
                let (image_num, suboptimal, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e)
                };

                // if suboptimal, recreate the swapchain
                if suboptimal {
                    recreate_swapchain = true;
                }

                // Have to build a command buffer in order to draw
                let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
                    .begin_render_pass(framebuffers[image_num].clone(), false,
                                       vec![
                                           [0.0, 0.0, 1.0, 1.0].into(),
                                           1f32.into()
                                       ]).unwrap()
                    .draw_indexed(
                        pipeline.clone(),
                        &DynamicState::none(),
                        vec!(vertex_buffer.clone()),
                        index_buffer.clone(), set.clone(), (),
                    ).unwrap()
                    .end_render_pass().unwrap()
                    .build().unwrap();

                let future = previous_frame_end.take().unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer).unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(Box::new(future) as Box<_>)
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>)
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                    }
                }
            }
            _ => ()
        }
    });
}

// Called during initialization, then whenever window is resized
fn window_size_dependent_setup(
    device: Arc<Device>,
    vs: &vertex_shader::Shader,
    fs: &frag_shader::Shader,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
) -> (Arc<dyn GraphicsPipelineAbstract + Send + Sync>, Vec<Arc<dyn FramebufferAbstract + Send + Sync>>) {
    let dimensions = images[0].dimensions();

    let depth_buffer = AttachmentImage::transient(device.clone(), dimensions, Format::D16Unorm).unwrap();

    let framebuffers = images.iter().map(|image| {
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .add(depth_buffer.clone()).unwrap()
                .build().unwrap()
        ) as Arc<dyn FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>();

    let pipeline = Arc::new(GraphicsPipeline::start()
        .vertex_input_single_buffer::<Vertex>()
        .vertex_shader(vs.main_entry_point(), ())
        .triangle_list()
        .viewports_dynamic_scissors_irrelevant(1)
        .viewports(iter::once(Viewport {
            origin: [0.0, 0.0],
            dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            depth_range: 0.0..1.0,
        }))
        .fragment_shader(fs.main_entry_point(), ())
        .depth_stencil_simple_depth()
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap()
    );

    (pipeline, framebuffers)
}

// Create the shaders
mod vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "resources/shaders/vert.glsl"
    }
}

mod frag_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "resources/shaders/frag.glsl"
    }
}