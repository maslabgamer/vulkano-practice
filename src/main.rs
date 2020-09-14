#![feature(clamp)]

mod object;
mod options;
mod player;
mod shaders;
mod world;

use crate::object::GameObject;
use crate::options::InternalConfig;
use crate::player::Player;
use crate::shaders::ShaderType;
use crate::shaders::Shaders;
use crate::world::chunk::Chunk;
use crate::world::map::Map;
use cgmath::{Matrix3, Matrix4, Point3, Rad, Vector3, Vector4};
use device_query::{DeviceQuery, DeviceState, Keycode};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use vulkano::buffer::{BufferUsage, CpuBufferPool};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract};
use vulkano::image::{AttachmentImage, SwapchainImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::swapchain;
use vulkano::swapchain::{
    AcquireError, ColorSpace, FullscreenExclusive, PresentMode, SurfaceTransform, Swapchain,
    SwapchainCreationError,
};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};
use vulkano_win::VkSurfaceBuild;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

const UP_VECTOR: Vector3<f32> = Vector3::new(0.0, -1.0, 0.0);

game_object![Player, Chunk];

fn main() {
    // Load engine configuration
    let internal_config: Arc<InternalConfig> = Arc::new(InternalConfig::load_internal_config(
        "resources/settings.toml",
    ));
    println!("internal_config = {:?}", internal_config);

    let world_scale = internal_config.engine.scale;

    // Get required extensions to draw window
    let required_extensions = vulkano_win::required_extensions();

    // Create vulkano instance
    let instance = Instance::new(None, &required_extensions, None).unwrap();
    // Get physical device
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    // Print the device name and type (verifies we've gotten it)
    println!(
        "Using device: {} (type: {:?})",
        physical.name(),
        physical.ty()
    );

    // Create window to actually display shit
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .with_title("ChunkRenderer")
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();
    surface
        .window()
        .set_inner_size(LogicalSize::new(1920, 1080));
    match surface.window().set_cursor_grab(true) {
        Ok(_) => println!("Got cursor lock on window."),
        Err(_) => panic!("Couldn't get cursor lock on window!"),
    }
    surface.window().set_cursor_visible(false); // Since we assume window is in focus by default

    // Pick a GPU queue to execute draw commands
    // Devices can provide multiple queues to execute commands in parallel
    let queue_family = physical
        .queue_families()
        .find(|&q| {
            // Take first queue that supports drawing the window
            q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
        })
        .unwrap();

    // Initialize device with required parameters
    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
        .unwrap();
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

        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count,
            format,
            dimensions,
            1,
            usage,
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Fifo,
            FullscreenExclusive::Default,
            true,
            ColorSpace::SrgbNonLinear,
        )
            .unwrap()
    };

    let uniform_buffer =
        CpuBufferPool::<vertex_shader::ty::Data>::new(device.clone(), BufferUsage::all());

    let vs = vertex_shader::Shader::load(device.clone()).unwrap();

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
        )
            .unwrap(),
    );

    // Need to create actual framebuffers
    let (mut pipelines, mut framebuffers) =
        window_size_dependent_setup(device.clone(), &vs, &images, render_pass.clone());

    // Flag swapchain recreation in case window is resized
    let mut recreate_swapchain = false;

    // Start submitting commands in loop below
    // Hold a place to store the submission of the previous frame
    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

    // Set up keyboard handler
    let device_state = DeviceState::new();
    let mut input_timer = Instant::now();

    // ! Load the map
    println!("Loading world...");
    let world_load_time = Instant::now();
    // Load basic world
    let map = match Map::load_from_file(
        device.clone(),
        "resources/map_file.map",
        world_scale,
        &internal_config,
        pipelines.shaders.get(&ShaderType::Default).unwrap().clone(),
    ) {
        Ok(map) => map,
        Err(e) => panic!(
            "There was a problem loading the world. Can't continue: {}",
            e
        ),
    };
    println!("Map took {:?} to load.", world_load_time.elapsed());

    // Set up player
    let spawn_location = map.spawn_location_as_point();

    let mut player = Player::new(spawn_location, 0.0, 0.0);

    let mut view_rotation: Vector4<f32> = Vector4 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
        w: 1.0,
    };

    // Set default position for mouse
    let mut default_mouse_position = PhysicalPosition {
        x: dimensions[0] / 2,
        y: dimensions[1] / 2,
    };
    let sensitivity = 1.3;

    // Set up elapsed time timer
    let mut timer = Instant::now();

    let mut window_is_focused = true; // Assume focused at startup

    // Set up debug timer for printing debug info
    let mut debug_timer = Instant::now();

    // Main game loop
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        // Update delta_time
        let delta_time = timer.elapsed().as_nanos() as f64 / 1_000_000_000.0;
        timer = Instant::now();

        // Handle input
        if input_timer.elapsed().as_millis() > 5 {
            // Get keys
            let keys: Vec<Keycode> = device_state.get_keys();
            if !keys.is_empty() {
                let forward: Vector4<f32> = view_rotation;
                let right = view_rotation.truncate().cross(UP_VECTOR);

                for key in keys {
                    match key {
                        Keycode::Escape => *control_flow = ControlFlow::Exit,
                        Keycode::W => player.location += forward.truncate(),
                        Keycode::S => player.location -= forward.truncate(),
                        Keycode::A => player.location -= right,
                        Keycode::D => player.location += right,
                        Keycode::Space => player.move_up(0.5),
                        Keycode::LShift => player.move_down(0.5),
                        _ => {}
                    }
                }
                input_timer = Instant::now();
            }
        }

        // Calculate distance from player to chunks
        // let mut player_check_collision_chunks: Vec<([i32; 3], f32)> = vec![];
        let mut player_check_collision_chunks: HashSet<[i32; 3]> =
            HashSet::with_capacity(map.chunks.len());
        for mut chunk in map.chunks.iter_mut() {
            let distance = player.distance_between(&*chunk);

            // Only need to cehck collision against chunks within a certain radius of the player
            if distance < chunk.collision_detection_distance {
                // player_check_collision_chunks.push((chunk.key().clone(), distance));
                player_check_collision_chunks.insert(chunk.key().clone());
                chunk.set_shader(
                    pipelines
                        .shaders
                        .get(&ShaderType::CollisionCheck)
                        .unwrap()
                        .clone(),
                );
            } else {
                chunk.set_shader(pipelines.shaders.get(&ShaderType::Default).unwrap().clone())
            }
        }

        // Now check events sent to the window (update view, etc)
        match event {
            Event::WindowEvent {
                event: WindowEvent::Focused(in_focus),
                ..
            } => {
                window_is_focused = in_focus;
                surface.window().set_cursor_visible(!window_is_focused);
            }
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                if window_is_focused {
                    let x_difference = position.x - default_mouse_position.x as f64;
                    let y_difference = position.y - default_mouse_position.y as f64;

                    player.yaw += (x_difference * delta_time * sensitivity) as f32;
                    player.pitch += (y_difference * delta_time * sensitivity) as f32;
                    // If I don't do this the world view disappears as it combines poorly with the
                    // "up" vector
                    // I fucking hate that this works and it feels hacky as hell
                    player.pitch = player.pitch.clamp(
                        -(std::f32::consts::FRAC_PI_2 - 0.00001),
                        std::f32::consts::FRAC_PI_2 - 0.00001,
                    );

                    if let Err(_) = surface.window().set_cursor_position(default_mouse_position) {
                        panic!("Could not set cursor position!");
                    }
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::RedrawEventsCleared => {
                // Clean up unused resources
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                // Whenever window resizes we need to recreate everything dependent on the window size.
                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    default_mouse_position = PhysicalPosition {
                        x: dimensions[0] / 2,
                        y: dimensions[1] / 2,
                    };
                    let (new_swapchain, new_images) =
                        match swapchain.recreate_with_dimensions(dimensions) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::UnsupportedDimensions) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    swapchain = new_swapchain;
                    let (new_pipelines, new_framebuffers) = window_size_dependent_setup(
                        device.clone(),
                        &vs,
                        &new_images,
                        render_pass.clone(),
                    );
                    pipelines = new_pipelines;
                    framebuffers = new_framebuffers;
                    recreate_swapchain = false;
                }

                // Have to acquire an images from the swapchain before we can draw it
                let (image_num, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                // if suboptimal, recreate the swapchain
                if suboptimal {
                    recreate_swapchain = true;
                }

                let uniform_buffer_subbuffer = {
                    let rotation = Matrix3::from_angle_y(Rad(0.0));

                    let aspect_ratio = dimensions[0] as f32 / dimensions[1] as f32;
                    let proj = cgmath::perspective(
                        Rad(std::f32::consts::FRAC_PI_2),
                        aspect_ratio,
                        0.01,
                        1000.0,
                    );

                    // Create our rotation matrices
                    let horizontal_rotation = Matrix4::from_angle_y(Rad(player.yaw));
                    let vertical_rotation = Matrix4::from_angle_x(Rad(player.pitch));
                    let camera_rotation = horizontal_rotation * vertical_rotation;

                    // Target is basically "right in front" of the camera
                    let target = Vector4::new(0.0, 0.0, 1.0, 1.0);

                    // Multiply target by the rotation vector.
                    view_rotation = camera_rotation * target;

                    let view =
                        Matrix4::look_at_dir(player.location, view_rotation.truncate(), UP_VECTOR);

                    let scale = Matrix4::from_scale(world_scale);

                    let uniform_data = vertex_shader::ty::Data {
                        world: Matrix4::from(rotation).into(),
                        view: (view * scale).into(),
                        proj: proj.into(),
                    };

                    Arc::new(uniform_buffer.next(uniform_data).unwrap())
                };

                // Have to build a command buffer in order to draw
                let mut command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(
                    device.clone(),
                    queue.family(),
                )
                    .unwrap();
                command_buffer.begin_render_pass(
                    framebuffers[image_num].clone(),
                    false,
                    vec![[0.0, 0.0, 0.0, 1.0].into(), 1f32.into()],
                )
                    .unwrap();
                // Iterate through chunks to render
                for chunk in map.chunks.iter() {
                    let layout = chunk.shader_pipeline.descriptor_set_layout(0).unwrap();
                    let set = Arc::new(
                        PersistentDescriptorSet::start(layout.clone())
                            .add_buffer(uniform_buffer_subbuffer.clone())
                            .unwrap()
                            .build()
                            .unwrap(),
                    );

                    command_buffer
                        .draw_indexed(
                            chunk.shader_pipeline.clone(),
                            &DynamicState::none(),
                            vec![chunk.vertex_buffer.clone()],
                            chunk.index_buffer.clone(),
                            set.clone(),
                            (),
                        )
                        .unwrap();
                }
                command_buffer.end_render_pass().unwrap();
                let command_buffer = command_buffer.build().unwrap();
                // End render pass

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => previous_frame_end = Some(Box::new(future) as Box<_>),
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
            _ => (),
        }

        // Debug info
        if debug_timer.elapsed().as_secs() > 1 {
            println!("player position = {:?}", player.location);
            println!(
                "player_distance_to_chunks = {:?}",
                player_check_collision_chunks
            );

            debug_timer = Instant::now();
        }
    });
}

// Called during initialization, then whenever window is resized
fn window_size_dependent_setup(
    device: Arc<Device>,
    vs: &vertex_shader::Shader,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
) -> (Shaders, Vec<Arc<dyn FramebufferAbstract + Send + Sync>>) {
    let dimensions = images[0].dimensions();

    let depth_buffer =
        AttachmentImage::transient(device.clone(), dimensions, Format::D16Unorm).unwrap();

    let framebuffers = images
        .iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .unwrap()
                    .add(depth_buffer.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>();

    let pipelines = Shaders::load_shaders(
        device.clone(),
        [dimensions[0], dimensions[1]],
        render_pass.clone(),
        &vs,
    );

    (pipelines, framebuffers)
}

// Create the shaders
mod vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "resources/shaders/chunk.vs"
    }
}
