use std::cell::RefCell;
use std::rc::Rc;

use js_sys::Math;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{
    CanvasRenderingContext2d, HtmlCanvasElement, KeyboardEvent, MouseEvent, TouchEvent, Window,
};

#[derive(Clone)]
struct Brick {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    alive: bool,
    hue: f64,
}

struct Particle {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    life: f64,
    radius: f64,
}

struct Star {
    x: f64,
    y: f64,
    r: f64,
    phase: f64,
}

struct Game {
    window: Window,
    canvas: HtmlCanvasElement,
    ctx: CanvasRenderingContext2d,
    width: f64,
    height: f64,
    scale: f64,

    paddle_x: f64,
    paddle_y: f64,
    paddle_w: f64,
    paddle_h: f64,
    paddle_target_x: f64,
    pointer_active: bool,

    ball_x: f64,
    ball_y: f64,
    ball_vx: f64,
    ball_vy: f64,
    ball_r: f64,
    ball_speed: f64,
    started: bool,

    rows: usize,
    cols: usize,
    bricks: Vec<Brick>,

    particles: Vec<Particle>,
    stars: Vec<Star>,

    input_left: bool,
    input_right: bool,
    lives: i32,
    score: i32,
    last_time: f64,
    time: f64,
}

impl Game {
    fn new(window: Window, canvas: HtmlCanvasElement, ctx: CanvasRenderingContext2d) -> Self {
        Self {
            window,
            canvas,
            ctx,
            width: 0.0,
            height: 0.0,
            scale: 1.0,
            paddle_x: 0.0,
            paddle_y: 0.0,
            paddle_w: 180.0,
            paddle_h: 16.0,
            paddle_target_x: 0.0,
            pointer_active: false,
            ball_x: 0.0,
            ball_y: 0.0,
            ball_vx: 0.0,
            ball_vy: 0.0,
            ball_r: 10.0,
            ball_speed: 420.0,
            started: false,
            rows: 6,
            cols: 10,
            bricks: Vec::new(),
            particles: Vec::new(),
            stars: Vec::new(),
            input_left: false,
            input_right: false,
            lives: 3,
            score: 0,
            last_time: 0.0,
            time: 0.0,
        }
    }

    fn reset_game(&mut self) {
        self.score = 0;
        self.lives = 3;
        self.ball_speed = 420.0;
        self.layout_bricks();
        self.reset_ball();
        self.started = false;
    }

    fn reset_ball(&mut self) {
        self.ball_x = self.paddle_x + self.paddle_w * 0.5;
        self.ball_y = self.paddle_y - self.ball_r - 6.0;
        self.ball_vx = 0.0;
        self.ball_vy = 0.0;
    }

    fn layout_bricks(&mut self) {
        let margin_x = 40.0;
        let top = 80.0;
        let spacing = 12.0;
        let cols = self.cols as f64;
        let usable_w = (self.width - margin_x * 2.0 - spacing * (cols - 1.0)).max(200.0);
        let brick_w = usable_w / cols;
        let brick_h = 28.0;

        self.bricks.clear();
        for row in 0..self.rows {
            for col in 0..self.cols {
                let x = margin_x + col as f64 * (brick_w + spacing);
                let y = top + row as f64 * (brick_h + spacing);
                let hue = 200.0 + (row as f64 * 12.0) + (col as f64 * 4.0);
                self.bricks.push(Brick {
                    x,
                    y,
                    w: brick_w,
                    h: brick_h,
                    alive: true,
                    hue,
                });
            }
        }
    }

    fn generate_stars(&mut self) {
        self.stars.clear();
        let count = ((self.width * self.height) / 12000.0).round().max(60.0) as usize;
        for _ in 0..count {
            self.stars.push(Star {
                x: Math::random() * self.width,
                y: Math::random() * self.height,
                r: 0.7 + Math::random() * 1.8,
                phase: Math::random() * std::f64::consts::PI * 2.0,
            });
        }
    }

    fn resize(&mut self) {
        let width = self
            .window
            .inner_width()
            .ok()
            .and_then(|v| v.as_f64())
            .unwrap_or(960.0);
        let height = self
            .window
            .inner_height()
            .ok()
            .and_then(|v| v.as_f64())
            .unwrap_or(540.0);
        let dpr = self.window.device_pixel_ratio().max(1.0);

        self.width = width;
        self.height = height;
        self.scale = dpr;

        self.canvas.set_width((width * dpr) as u32);
        self.canvas.set_height((height * dpr) as u32);
        let _ = self
            .canvas
            .style()
            .set_property("width", &format!("{}px", width));
        let _ = self
            .canvas
            .style()
            .set_property("height", &format!("{}px", height));

        let _ = self.ctx.set_transform(dpr, 0.0, 0.0, dpr, 0.0, 0.0);

        self.paddle_w = (self.width * 0.18).clamp(120.0, 240.0);
        self.paddle_h = 16.0;
        self.paddle_y = self.height - 50.0;
        self.paddle_x = (self.width - self.paddle_w) * 0.5;
        self.paddle_target_x = self.paddle_x;

        self.ball_r = (self.width * 0.012).clamp(8.0, 14.0);

        self.layout_bricks();
        self.generate_stars();
        self.reset_ball();
    }

    fn launch_ball(&mut self) {
        let angle = (Math::random() * 0.7 + 0.15) * std::f64::consts::PI;
        self.ball_vx = self.ball_speed * angle.cos();
        self.ball_vy = -self.ball_speed * angle.sin();
        self.started = true;
    }

    fn tick(&mut self, now: f64) {
        if self.last_time == 0.0 {
            self.last_time = now;
        }
        let mut dt = (now - self.last_time) / 1000.0;
        self.last_time = now;
        if dt > 0.05 {
            dt = 0.05;
        }
        self.time += dt;

        self.update(dt);
        self.draw();
    }

    fn update(&mut self, dt: f64) {
        let paddle_speed = 820.0;

        if self.pointer_active {
            self.paddle_x = self.paddle_target_x;
        } else {
            if self.input_left {
                self.paddle_x -= paddle_speed * dt;
            }
            if self.input_right {
                self.paddle_x += paddle_speed * dt;
            }
        }

        self.paddle_x = self.paddle_x.clamp(0.0, self.width - self.paddle_w);

        if !self.started {
            self.ball_x = self.paddle_x + self.paddle_w * 0.5;
            self.ball_y = self.paddle_y - self.ball_r - 6.0;
        } else {
            self.ball_x += self.ball_vx * dt;
            self.ball_y += self.ball_vy * dt;
        }

        self.handle_collisions();
        self.update_particles(dt);

        if self.bricks.iter().all(|b| !b.alive) {
            self.ball_speed *= 1.06;
            self.layout_bricks();
            self.started = false;
        }
    }

    fn handle_collisions(&mut self) {
        let r = self.ball_r;

        // Wall collisions (unchanged)
        if self.ball_x - r <= 0.0 {
            self.ball_x = r;
            self.ball_vx = self.ball_vx.abs();
        }
        if self.ball_x + r >= self.width {
            self.ball_x = self.width - r;
            self.ball_vx = -self.ball_vx.abs();
        }
        if self.ball_y - r <= 0.0 {
            self.ball_y = r;
            self.ball_vy = self.ball_vy.abs();
        }
        if self.ball_y - r > self.height {
            self.lives -= 1;
            if self.lives <= 0 {
                self.reset_game();
            } else {
                self.started = false;
                self.reset_ball();
            }
            return;
        }

        // Paddle collision (unchanged)
        if self.ball_vy > 0.0
            && self.ball_y + r >= self.paddle_y
            && self.ball_y + r <= self.paddle_y + self.paddle_h + 4.0
            && self.ball_x >= self.paddle_x
            && self.ball_x <= self.paddle_x + self.paddle_w
        {
            let hit = (self.ball_x - (self.paddle_x + self.paddle_w * 0.5)) / (self.paddle_w * 0.5);
            let hit = hit.clamp(-1.0, 1.0);
            let max_angle = 75.0_f64.to_radians();
            let angle = hit * max_angle;
            let speed = (self.ball_vx * self.ball_vx + self.ball_vy * self.ball_vy)
                .sqrt()
                .max(self.ball_speed * 0.85);
            self.ball_vx = speed * angle.sin();
            self.ball_vy = -speed * angle.cos();
            self.ball_y = self.paddle_y - r - 1.0;
        }

        // Brick collisions
        let mut hit_index: Option<usize> = None;
        for (i, brick) in self.bricks.iter().enumerate() {
            if !brick.alive {
                continue;
            }
            let closest_x = self.ball_x.clamp(brick.x, brick.x + brick.w);
            let closest_y = self.ball_y.clamp(brick.y, brick.y + brick.h);
            let dx = self.ball_x - closest_x;
            let dy = self.ball_y - closest_y;
            if dx * dx + dy * dy <= r * r {
                hit_index = Some(i);
                break;
            }
        }

        if let Some(i) = hit_index {
            let (center_x, center_y, hue, hit_horizontal) = {
                let brick = &mut self.bricks[i];
                brick.alive = false;
                self.score += 100;
                let center_x = brick.x + brick.w * 0.5;
                let center_y = brick.y + brick.h * 0.5;

                // --- FIX STARTS HERE ---
                // Calculate absolute distance from centers
                let dist_x = (self.ball_x - center_x).abs();
                let dist_y = (self.ball_y - center_y).abs();

                // Calculate overlap on each axis
                // (Sum of half-widths) - distance
                let overlap_x = (brick.w * 0.5 + r) - dist_x;
                let overlap_y = (brick.h * 0.5 + r) - dist_y;

                // The collision happened on the axis with the *smallest* overlap
                // (i.e., the path of least resistance)
                let hit_horizontal = overlap_x < overlap_y;
                // --- FIX ENDS HERE ---

                (center_x, center_y, brick.hue, hit_horizontal)
            };

            if hit_horizontal {
                self.ball_vx = -self.ball_vx;
            } else {
                self.ball_vy = -self.ball_vy;
            }

            self.ball_speed *= 1.003;
            let speed = (self.ball_vx * self.ball_vx + self.ball_vy * self.ball_vy).sqrt();
            if speed > 0.0 {
                let scale = self.ball_speed / speed;
                self.ball_vx *= scale;
                self.ball_vy *= scale;
            }
            self.spawn_hit_particles(center_x, center_y, hue);
        }
    }

    fn spawn_hit_particles(&mut self, x: f64, y: f64, _hue: f64) {
        for _ in 0..12 {
            let angle = Math::random() * std::f64::consts::PI * 2.0;
            let speed = 60.0 + Math::random() * 180.0;
            self.particles.push(Particle {
                x,
                y,
                vx: speed * angle.cos(),
                vy: speed * angle.sin(),
                life: 0.6 + Math::random() * 0.6,
                radius: 1.5 + Math::random() * 2.5,
            });
        }
        if self.particles.len() > 260 {
            self.particles.drain(0..80);
        }
    }

    fn update_particles(&mut self, dt: f64) {
        if self.started {
            let count = 3;
            for _ in 0..count {
                let jitter = (Math::random() - 0.5) * self.ball_r * 0.8;
                self.particles.push(Particle {
                    x: self.ball_x + jitter,
                    y: self.ball_y + jitter,
                    vx: (Math::random() - 0.5) * 20.0,
                    vy: 40.0 + Math::random() * 60.0,
                    life: 0.4 + Math::random() * 0.4,
                    radius: 1.5 + Math::random() * 2.5,
                });
            }
        }

        for p in &mut self.particles {
            p.x += p.vx * dt;
            p.y += p.vy * dt;
            p.life -= dt;
            p.radius *= 0.98;
        }
        self.particles.retain(|p| p.life > 0.0 && p.radius > 0.4);
    }

    fn draw(&self) {
        let ctx = &self.ctx;
        ctx.set_global_alpha(1.0);

        let gradient = ctx.create_linear_gradient(0.0, 0.0, 0.0, self.height);
        gradient.add_color_stop(0.0, "#0b1026").ok();
        gradient.add_color_stop(0.45, "#111c3f").ok();
        gradient.add_color_stop(1.0, "#1d0f2e").ok();
        let gradient_val = JsValue::from(gradient);
        ctx.set_fill_style(&gradient_val);
        ctx.fill_rect(0.0, 0.0, self.width, self.height);

        ctx.set_fill_style(&JsValue::from_str("rgba(255,255,255,0.08)"));
        for i in 0..12 {
            let y = self.height * 0.08 + i as f64 * 48.0;
            ctx.fill_rect(0.0, y, self.width, 1.0);
        }

        for star in &self.stars {
            let twinkle = (self.time * 1.8 + star.phase).sin() * 0.5 + 0.6;
            ctx.set_global_alpha(twinkle.clamp(0.2, 0.9));
            ctx.set_fill_style(&JsValue::from_str("#ffffff"));
            ctx.fill_rect(star.x, star.y, star.r, star.r);
        }
        ctx.set_global_alpha(1.0);

        ctx.save();
        ctx.set_shadow_blur(22.0);
        for brick in &self.bricks {
            if !brick.alive {
                continue;
            }
            let grad = ctx.create_linear_gradient(brick.x, brick.y, brick.x, brick.y + brick.h);
            let top = format!("hsla({}, 88%, 68%, 0.95)", brick.hue);
            let bottom = format!("hsla({}, 80%, 46%, 0.95)", brick.hue);
            grad.add_color_stop(0.0, &top).ok();
            grad.add_color_stop(1.0, &bottom).ok();
            let grad_val = JsValue::from(grad);
            ctx.set_fill_style(&grad_val);
            ctx.set_shadow_color(&format!("hsla({}, 90%, 60%, 0.7)", brick.hue));
            ctx.fill_rect(brick.x, brick.y, brick.w, brick.h);

            ctx.set_shadow_blur(0.0);
            ctx.set_fill_style(&JsValue::from_str("rgba(255,255,255,0.25)"));
            ctx.fill_rect(brick.x + 2.0, brick.y + 2.0, brick.w - 4.0, 4.0);
            ctx.set_shadow_blur(22.0);
        }
        ctx.restore();

        ctx.save();
        ctx.set_shadow_blur(30.0);
        ctx.set_shadow_color("rgba(67, 214, 255, 0.7)");
        let paddle_grad = ctx.create_linear_gradient(
            self.paddle_x,
            self.paddle_y,
            self.paddle_x,
            self.paddle_y + self.paddle_h,
        );
        paddle_grad.add_color_stop(0.0, "#7ee8ff").ok();
        paddle_grad.add_color_stop(1.0, "#1aa0ff").ok();
        let paddle_val = JsValue::from(paddle_grad);
        ctx.set_fill_style(&paddle_val);
        ctx.fill_rect(self.paddle_x, self.paddle_y, self.paddle_w, self.paddle_h);
        ctx.restore();

        ctx.save();
        ctx.set_global_composite_operation("lighter").ok();
        for p in &self.particles {
            let alpha = (p.life * 1.6).clamp(0.0, 0.6);
            ctx.set_global_alpha(alpha);
            ctx.set_fill_style(&JsValue::from_str("rgba(110, 227, 255, 1.0)"));
            ctx.begin_path();
            let _ = ctx.arc(p.x, p.y, p.radius, 0.0, std::f64::consts::PI * 2.0);
            ctx.fill();
        }
        ctx.restore();
        ctx.set_global_alpha(1.0);

        ctx.save();
        ctx.set_shadow_blur(28.0);
        ctx.set_shadow_color("rgba(255, 255, 255, 0.9)");
        let ball_grad = ctx
            .create_radial_gradient(
                self.ball_x - 3.0,
                self.ball_y - 3.0,
                2.0,
                self.ball_x,
                self.ball_y,
                self.ball_r * 2.2,
            )
            .ok();
        if let Some(ball_grad) = ball_grad {
            ball_grad.add_color_stop(0.0, "#ffffff").ok();
            ball_grad.add_color_stop(0.6, "#9df3ff").ok();
            ball_grad
                .add_color_stop(1.0, "rgba(120, 180, 255, 0.1)")
                .ok();
            ctx.set_fill_style(&JsValue::from(ball_grad));
        } else {
            ctx.set_fill_style(&JsValue::from_str("#d8f7ff"));
        }
        ctx.begin_path();
        let _ = ctx.arc(
            self.ball_x,
            self.ball_y,
            self.ball_r,
            0.0,
            std::f64::consts::PI * 2.0,
        );
        ctx.fill();
        ctx.restore();

        ctx.set_fill_style(&JsValue::from_str("rgba(255,255,255,0.9)"));
        ctx.set_font("16px system-ui, sans-serif");
        ctx.fill_text(&format!("Score: {}", self.score), 24.0, 32.0)
            .ok();
        ctx.fill_text(&format!("Lives: {}", self.lives), self.width - 120.0, 32.0)
            .ok();

        if !self.started {
            ctx.set_fill_style(&JsValue::from_str("rgba(255,255,255,0.75)"));
            ctx.set_font("18px system-ui, sans-serif");
            ctx.fill_text(
                "Click / tap or press Space to launch",
                self.width * 0.5 - 180.0,
                self.height * 0.6,
            )
            .ok();
        }
    }
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    let _ = web_sys::window()
        .expect("no window")
        .request_animation_frame(f.as_ref().unchecked_ref());
}

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let document = window
        .document()
        .ok_or_else(|| JsValue::from_str("no document"))?;
    let body = document
        .body()
        .ok_or_else(|| JsValue::from_str("no body"))?;

    body.style().set_property("margin", "0")?;
    body.style().set_property("overflow", "hidden")?;
    body.style().set_property("background", "#0b1026")?;

    let canvas = document
        .create_element("canvas")?
        .dyn_into::<HtmlCanvasElement>()?;
    canvas.style().set_property("display", "block")?;
    body.append_child(&canvas)?;

    let ctx = canvas
        .get_context("2d")?
        .ok_or_else(|| JsValue::from_str("no context"))?
        .dyn_into::<CanvasRenderingContext2d>()?;

    let game = Rc::new(RefCell::new(Game::new(window.clone(), canvas.clone(), ctx)));

    game.borrow_mut().resize();

    {
        let game = game.clone();
        let closure = Closure::wrap(Box::new(move || {
            game.borrow_mut().resize();
        }) as Box<dyn FnMut()>);
        window.add_event_listener_with_callback("resize", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    {
        let game = game.clone();
        let closure = Closure::wrap(Box::new(move |event: MouseEvent| {
            let mut game = game.borrow_mut();
            game.pointer_active = true;
            game.paddle_target_x = event.client_x() as f64 - game.paddle_w * 0.5;
        }) as Box<dyn FnMut(MouseEvent)>);
        window.add_event_listener_with_callback("mousemove", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    {
        let game = game.clone();
        let closure = Closure::wrap(Box::new(move |event: TouchEvent| {
            let touch = event.touches().item(0);
            if let Some(touch) = touch {
                let mut game = game.borrow_mut();
                game.pointer_active = true;
                game.paddle_target_x = touch.client_x() as f64 - game.paddle_w * 0.5;
            }
        }) as Box<dyn FnMut(TouchEvent)>);
        window.add_event_listener_with_callback("touchmove", closure.as_ref().unchecked_ref())?;
        window.add_event_listener_with_callback("touchstart", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    {
        let game = game.clone();
        let closure = Closure::wrap(Box::new(move |event: KeyboardEvent| {
            let mut game = game.borrow_mut();
            match event.key().as_str() {
                "ArrowLeft" | "a" | "A" => {
                    game.input_left = true;
                    game.pointer_active = false;
                }
                "ArrowRight" | "d" | "D" => {
                    game.input_right = true;
                    game.pointer_active = false;
                }
                " " => {
                    if !game.started {
                        game.launch_ball();
                    }
                }
                _ => {}
            }
        }) as Box<dyn FnMut(KeyboardEvent)>);
        window.add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    {
        let game = game.clone();
        let closure = Closure::wrap(Box::new(move |event: KeyboardEvent| {
            let mut game = game.borrow_mut();
            match event.key().as_str() {
                "ArrowLeft" | "a" | "A" => game.input_left = false,
                "ArrowRight" | "d" | "D" => game.input_right = false,
                _ => {}
            }
            let _ = event;
        }) as Box<dyn FnMut(KeyboardEvent)>);
        window.add_event_listener_with_callback("keyup", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    {
        let game = game.clone();
        let closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
            let mut game = game.borrow_mut();
            if !game.started {
                game.launch_ball();
            }
        }) as Box<dyn FnMut(MouseEvent)>);
        window.add_event_listener_with_callback("mousedown", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    let perf = window
        .performance()
        .ok_or_else(|| JsValue::from_str("no performance"))?;

    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    let game_loop = game.clone();

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        let now = perf.now();
        game_loop.borrow_mut().tick(now);
        if let Some(cb) = f.borrow().as_ref() {
            request_animation_frame(cb);
        }
    }) as Box<dyn FnMut()>));

    if let Some(cb) = g.borrow().as_ref() {
        request_animation_frame(cb);
    }

    Ok(())
}
