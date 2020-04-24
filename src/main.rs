#![feature(clamp)]

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
use vulkano::buffer::{BufferUsage, CpuBufferPool};
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
use cgmath::{Matrix3, Rad, Matrix4, Point3, Vector3, Vector4};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use winit::event::VirtualKeyCode::{Space, LShift, Escape};
use winit::dpi::{LogicalSize, PhysicalPosition};
use std::time::Instant;

fn main() {
    println!("Loading map...");
    // Load basic map
    let map = match Map::load_from_file("resources/map_file.map") {
        Ok(map) => map,
        Err(e) => panic!("There was a problem loading the map. Can't continue: {}", e),
    };
    println!("map keys are: {:?}", map.chunks.keys());

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
    surface.window().set_inner_size(LogicalSize::new(1920, 1080));
    match surface.window().set_cursor_grab(true) {
        Ok(_) => println!("Got cursor lock on window."),
        Err(_) => panic!("Couldn't get cursor lock on window!"),
    }
    surface.window().set_cursor_visible(false); // Since we assume window is in focus by default

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
    println!("Dimensions are: {:?}", dimensions);
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

    let (vertex_buffers, index_buffers) = map.render_map(&device);

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
    let mut eye = Point3::new(0.3, 0.3, 25.0);
    let mut eye_yaw: f32 = 0.0;
    let mut eye_pitch: f32 = 0.0;

    // Set default position for mouse
    let mut default_mouse_position = PhysicalPosition { x: dimensions[0] / 2, y: dimensions[1] / 2 };
    let sensitivity = 1.0;

    // Set up elapsed time timer
    let mut timer = Instant::now();

    let mut window_is_focused = true; // Assume focused at startup

    event_loop.run(move |event, _, control_flow| {
        let delta_time = timer.elapsed().as_nanos() as f64 / 1_000_000_000.0;
        timer = Instant::now();
        match event {
            Event::WindowEvent { event: WindowEvent::KeyboardInput { input, .. }, .. } => {
                // println!("input = {:?}", input);
                match input.virtual_keycode {
                    Some(Escape) => *control_flow = ControlFlow::Exit,
                    Some(Space) => eye.y += 0.5,
                    Some(LShift) => eye.y -= 0.5,
                    _ => (),
                }
            }
            Event::WindowEvent { event: WindowEvent::Focused(_0), .. } => {
                window_is_focused = _0;
                surface.window().set_cursor_visible(!window_is_focused);
            }
            Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, .. } => {
                if window_is_focused {
                    let x_difference = position.x - default_mouse_position.x as f64;
                    let y_difference = position.y - default_mouse_position.y as f64;

                    eye_yaw += (x_difference * delta_time * sensitivity) as f32;
                    eye_pitch += (y_difference * delta_time * sensitivity) as f32;
                    eye_pitch = eye_pitch.clamp(-(std::f32::consts::FRAC_PI_2 - 0.00001), std::f32::consts::FRAC_PI_2 - 0.00001);

                    match surface.window().set_cursor_position(default_mouse_position) {
                        Ok(_) => {}
                        Err(_) => panic!("Could not set cursor position!"),
                    }
                }
            }
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
                    default_mouse_position = PhysicalPosition { x: dimensions[0] / 2, y: dimensions[1] / 2 };
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

                    // Create our rotation matrices
                    let horizontal_rotation = Matrix4::from_angle_y(Rad(eye_yaw));
                    let vertical_rotation = Matrix4::from_angle_x(Rad(eye_pitch));
                    let camera_rotation = horizontal_rotation * vertical_rotation;

                    // Target is basically "right in front" of the camera
                    let target = Vector4::new(0.0, 0.0, 1.0, 1.0);

                    // Multiply target by the rotation vector.
                    let view_rotation: Vector4<f32> = camera_rotation * target;

                    // Since we're rotating the "look at" vector around the origin, we need to move
                    // back to the "player"
                    let final_target = Point3::new(view_rotation.x + eye.x, view_rotation.y + eye.y, view_rotation.z + eye.z);

                    let view = Matrix4::look_at(eye, final_target, Vector3::new(0.0, -1.0, 0.0));

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
                let mut command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
                    .begin_render_pass(framebuffers[image_num].clone(), false,
                                       vec![
                                           [0.0, 0.0, 1.0, 1.0].into(),
                                           1f32.into()
                                       ]).unwrap();
                for (vertex_buffer, index_buffer) in vertex_buffers.iter().zip(index_buffers.iter()) {
                    command_buffer = command_buffer.draw_indexed(
                        pipeline.clone(),
                        &DynamicState::none(),
                        vec!(vertex_buffer.clone()),
                        index_buffer.clone(), set.clone(), (),
                    ).unwrap();
                }
                let command_buffer = command_buffer.end_render_pass().unwrap().build().unwrap();

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