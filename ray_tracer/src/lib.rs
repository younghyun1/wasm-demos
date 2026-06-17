use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

use wasm_bindgen::Clamped;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData, MouseEvent, Window};

mod wgpu_renderer;

// ---- JS helpers ----
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => { log(&format!($($t)*)) };
}

fn window() -> Window {
    web_sys::window().unwrap()
}
fn document() -> web_sys::Document {
    window().document().unwrap()
}
fn perf() -> web_sys::Performance {
    window().performance().unwrap()
}
fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    window()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .unwrap();
}

// ---- Vec3 ----
#[derive(Clone, Copy)]
struct V3 {
    x: f32,
    y: f32,
    z: f32,
}

impl V3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }
    fn dot(self, b: Self) -> f32 {
        self.x * b.x + self.y * b.y + self.z * b.z
    }
    fn cross(self, b: Self) -> Self {
        Self {
            x: self.y * b.z - self.z * b.y,
            y: self.z * b.x - self.x * b.z,
            z: self.x * b.y - self.y * b.x,
        }
    }
    fn len(self) -> f32 {
        self.dot(self).sqrt()
    }
    fn norm(self) -> Self {
        let l = self.len();
        if l < 1e-10 { self } else { self * (1.0 / l) }
    }
    fn reflect(self, n: Self) -> Self {
        self - n * (2.0 * self.dot(n))
    }
    fn refract(self, n: Self, eta: f32) -> Self {
        let cos_i = (-self).dot(n).min(1.0);
        let r_perp = (self + n * cos_i) * eta;
        let r_par = n * -(1.0 - r_perp.dot(r_perp)).abs().sqrt();
        r_perp + r_par
    }
    fn max_comp(self) -> f32 {
        self.x.max(self.y.max(self.z))
    }
    fn clamp01(self) -> Self {
        Self {
            x: self.x.clamp(0.0, 1.0),
            y: self.y.clamp(0.0, 1.0),
            z: self.z.clamp(0.0, 1.0),
        }
    }
}

impl Add for V3 {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            x: self.x + b.x,
            y: self.y + b.y,
            z: self.z + b.z,
        }
    }
}
impl Sub for V3 {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self {
            x: self.x - b.x,
            y: self.y - b.y,
            z: self.z - b.z,
        }
    }
}
impl Mul<f32> for V3 {
    type Output = Self;
    fn mul(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}
impl Mul<V3> for V3 {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            x: self.x * b.x,
            y: self.y * b.y,
            z: self.z * b.z,
        }
    }
}
impl Div<f32> for V3 {
    type Output = Self;
    fn div(self, s: f32) -> Self {
        self * (1.0 / s)
    }
}
impl Neg for V3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

// ---- PCG RNG ----
struct Rng(u32);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        let old = self.0;
        self.0 = old.wrapping_mul(747796405).wrapping_add(2891336453);
        let word = ((old >> ((old >> 28).wrapping_add(4))) ^ old).wrapping_mul(277803737);
        (word >> 22) ^ word
    }
    fn f32(&mut self) -> f32 {
        self.next_u32() as f32 / 4294967296.0
    }
    fn unit_disk(&mut self) -> (f32, f32) {
        let r = self.f32().sqrt();
        let a = self.f32() * std::f32::consts::TAU;
        (r * a.cos(), r * a.sin())
    }
    fn cos_hemisphere(&mut self, n: V3) -> V3 {
        let r1 = self.f32();
        let r2 = self.f32();
        let phi = std::f32::consts::TAU * r1;
        let cos_theta = r2.sqrt();
        let sin_theta = (1.0 - r2).sqrt();
        let up = if n.y.abs() > 0.99 {
            V3::new(1.0, 0.0, 0.0)
        } else {
            V3::new(0.0, 1.0, 0.0)
        };
        let t = up.cross(n).norm();
        let b = n.cross(t);
        (t * phi.cos() * sin_theta + b * phi.sin() * sin_theta + n * cos_theta).norm()
    }
}

// ---- Ray / Hit ----
struct Ray {
    o: V3,
    d: V3,
}

struct Hit {
    #[allow(dead_code)]
    t: f32,
    pos: V3,
    normal: V3,
    mat_id: u8,
    front: bool,
}

const EPS: f32 = 0.001;
const INF: f32 = 1e20;
const MAX_BOUNCE: u32 = 10;
const FRAME_BUDGET_MS: f64 = 12.0;
const PIXEL_BATCH: usize = 256;
const FIREFLY_CLAMP: f32 = 16.0;

// ---- Materials ----
// 0=diffuse, 1=metal, 2=dielectric, 3=emissive
struct Mat {
    kind: u8,
    albedo: V3,
    param: f32,
}

fn get_mat(id: u8) -> Mat {
    match id {
        0 => Mat {
            kind: 0,
            albedo: V3::new(0.73, 0.73, 0.73),
            param: 0.0,
        },
        1 => Mat {
            kind: 0,
            albedo: V3::new(0.63, 0.26, 0.30),
            param: 0.0,
        },
        2 => Mat {
            kind: 0,
            albedo: V3::new(0.20, 0.52, 0.50),
            param: 0.0,
        },
        3 => Mat {
            kind: 3,
            albedo: V3::new(1.0, 0.9, 0.7),
            param: 18.0,
        },
        4 => Mat {
            kind: 1,
            albedo: V3::new(0.95, 0.78, 0.45),
            param: 0.02,
        },
        5 => Mat {
            kind: 2,
            albedo: V3::splat(1.0),
            param: 1.5,
        },
        6 => Mat {
            kind: 0,
            albedo: V3::new(0.16, 0.28, 0.72),
            param: 0.0,
        },
        7 => Mat {
            kind: 0,
            albedo: V3::new(0.72, 0.55, 0.22),
            param: 0.0,
        },
        8 => Mat {
            kind: 1,
            albedo: V3::new(0.86, 0.88, 0.95),
            param: 0.14,
        },
        9 => Mat {
            kind: 0,
            albedo: V3::splat(0.5),
            param: 0.0,
        },
        10 => Mat {
            kind: 0,
            albedo: V3::new(0.78, 0.78, 0.80),
            param: 0.0,
        },
        11 => Mat {
            kind: 0,
            albedo: V3::new(0.70, 0.72, 0.75),
            param: 0.0,
        },
        12 => Mat {
            kind: 0,
            albedo: V3::new(0.74, 0.34, 0.62),
            param: 0.0,
        },
        13 => Mat {
            kind: 0,
            albedo: V3::new(0.80, 0.71, 0.28),
            param: 0.0,
        },
        14 => Mat {
            kind: 1,
            albedo: V3::new(0.80, 0.84, 0.92),
            param: 0.04,
        },
        15 => Mat {
            kind: 0,
            albedo: V3::new(0.80, 0.45, 0.20),
            param: 0.0,
        },
        16 => Mat {
            kind: 3,
            albedo: V3::new(1.0, 0.55, 0.22),
            param: 6.0,
        },
        _ => Mat {
            kind: 0,
            albedo: V3::splat(0.5),
            param: 0.0,
        },
    }
}

// ---- Intersections ----
fn hit_sphere(center: V3, radius: f32, ray: &Ray, t_max: f32) -> Option<(f32, V3, bool)> {
    let oc = ray.o - center;
    let a = ray.d.dot(ray.d);
    let hb = oc.dot(ray.d);
    let c = oc.dot(oc) - radius * radius;
    let disc = hb * hb - a * c;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let mut t = (-hb - sq) / a;
    if t < EPS || t > t_max {
        t = (-hb + sq) / a;
        if t < EPS || t > t_max {
            return None;
        }
    }
    let pos = ray.o + ray.d * t;
    let outward = (pos - center) / radius;
    let front = ray.d.dot(outward) < 0.0;
    let normal = if front { outward } else { -outward };
    Some((t, normal, front))
}

fn hit_quad(corner: V3, u: V3, v: V3, ray: &Ray, t_max: f32) -> Option<(f32, V3, bool)> {
    let n = u.cross(v);
    let normal = n.norm();
    let denom = normal.dot(ray.d);
    if denom.abs() < 1e-8 {
        return None;
    }
    let d = normal.dot(corner);
    let t = (d - normal.dot(ray.o)) / denom;
    if t < EPS || t > t_max {
        return None;
    }
    let p = ray.o + ray.d * t;
    let planar = p - corner;
    let n_sq = n.dot(n);
    let alpha = planar.cross(v).dot(n) / n_sq;
    let beta = u.cross(planar).dot(n) / n_sq;
    if alpha < 0.0 || alpha > 1.0 || beta < 0.0 || beta > 1.0 {
        return None;
    }
    let front = denom < 0.0;
    let norm = if front { normal } else { -normal };
    Some((t, norm, front))
}

fn trace_scene(ray: &Ray) -> Option<Hit> {
    let mut closest = INF;
    let mut result: Option<Hit> = None;
    const HW: f32 = 4.2;
    const HD: f32 = 3.0;
    const RH: f32 = 5.6;

    macro_rules! check {
        ($test:expr, $mat:expr) => {
            if let Some((t, normal, front)) = $test {
                if t < closest {
                    closest = t;
                    result = Some(Hit {
                        t,
                        pos: ray.o + ray.d * t,
                        normal,
                        mat_id: $mat,
                        front,
                    });
                }
            }
        };
    }

    // Room shell
    check!(
        hit_quad(
            V3::new(-HW, 0.0, -HD),
            V3::new(2.0 * HW, 0.0, 0.0),
            V3::new(0.0, 0.0, 2.0 * HD),
            ray,
            closest
        ),
        9
    );
    check!(
        hit_quad(
            V3::new(-HW, RH, -HD),
            V3::new(2.0 * HW, 0.0, 0.0),
            V3::new(0.0, 0.0, 2.0 * HD),
            ray,
            closest
        ),
        10
    );
    check!(
        hit_quad(
            V3::new(-HW, 0.0, -HD),
            V3::new(2.0 * HW, 0.0, 0.0),
            V3::new(0.0, RH, 0.0),
            ray,
            closest
        ),
        11
    );
    check!(
        hit_quad(
            V3::new(-HW, 0.0, -HD),
            V3::new(0.0, 0.0, 2.0 * HD),
            V3::new(0.0, RH, 0.0),
            ray,
            closest
        ),
        1
    );
    check!(
        hit_quad(
            V3::new(HW, 0.0, -HD),
            V3::new(0.0, 0.0, 2.0 * HD),
            V3::new(0.0, RH, 0.0),
            ray,
            closest
        ),
        2
    );
    // Area light
    check!(
        hit_quad(
            V3::new(-1.3, RH - 0.02, -0.9),
            V3::new(2.6, 0.0, 0.0),
            V3::new(0.0, 0.0, 1.8),
            ray,
            closest
        ),
        3
    );
    // Spheres
    check!(hit_sphere(V3::new(1.5, 1.0, 0.2), 1.0, ray, closest), 5);
    check!(hit_sphere(V3::new(-1.8, 0.8, -0.4), 0.8, ray, closest), 4);
    check!(hit_sphere(V3::new(2.7, 0.42, -1.6), 0.42, ray, closest), 8);
    check!(hit_sphere(V3::new(-1.5, 0.55, 1.5), 0.55, ray, closest), 6);
    check!(hit_sphere(V3::new(0.2, 0.35, 1.9), 0.35, ray, closest), 7);
    check!(hit_sphere(V3::new(-3.0, 0.3, 1.4), 0.3, ray, closest), 16);
    // Boxes (metal pillar + diffuse riser)
    check!(
        hit_box(
            V3::new(-3.2, 0.0, -2.4),
            V3::new(-2.4, 2.2, -1.6),
            ray,
            closest
        ),
        14
    );
    check!(
        hit_box(V3::new(2.0, 0.0, 1.4), V3::new(2.8, 0.5, 2.2), ray, closest),
        15
    );
    let _ = closest;
    result
}

fn hit_box(bmin: V3, bmax: V3, ray: &Ray, t_max: f32) -> Option<(f32, V3, bool)> {
    let inv = V3::new(1.0 / ray.d.x, 1.0 / ray.d.y, 1.0 / ray.d.z);
    let t0 = (bmin - ray.o) * inv;
    let t1 = (bmax - ray.o) * inv;
    let tsmall = V3::new(t0.x.min(t1.x), t0.y.min(t1.y), t0.z.min(t1.z));
    let tbig = V3::new(t0.x.max(t1.x), t0.y.max(t1.y), t0.z.max(t1.z));
    let tn = tsmall.x.max(tsmall.y).max(tsmall.z);
    let tf = tbig.x.min(tbig.y).min(tbig.z);
    if tn > tf || tf < EPS {
        return None;
    }
    let (t, on_far) = if tn < EPS { (tf, true) } else { (tn, false) };
    if t < EPS || t > t_max {
        return None;
    }
    let mut n = if !on_far {
        if tn == tsmall.x {
            V3::new(-ray.d.x.signum(), 0.0, 0.0)
        } else if tn == tsmall.y {
            V3::new(0.0, -ray.d.y.signum(), 0.0)
        } else {
            V3::new(0.0, 0.0, -ray.d.z.signum())
        }
    } else if tf == tbig.x {
        V3::new(ray.d.x.signum(), 0.0, 0.0)
    } else if tf == tbig.y {
        V3::new(0.0, ray.d.y.signum(), 0.0)
    } else {
        V3::new(0.0, 0.0, ray.d.z.signum())
    };
    let front = ray.d.dot(n) < 0.0;
    if !front {
        n = -n;
    }
    Some((t, n, front))
}

fn checker(p: V3) -> V3 {
    let ix = (p.x * 0.7).floor() as i32;
    let iz = (p.z * 0.7).floor() as i32;
    if (ix + iz) & 1 == 0 {
        V3::new(0.82, 0.80, 0.76)
    } else {
        V3::new(0.18, 0.19, 0.22)
    }
}

// Next event estimation: sample the ceiling area light directly.
fn sample_light(p: V3, n: V3, rng: &mut Rng) -> V3 {
    let q = V3::new(-1.3, 5.6 - 0.02, -0.9)
        + V3::new(2.6, 0.0, 0.0) * rng.f32()
        + V3::new(0.0, 0.0, 1.8) * rng.f32();
    let to = q - p;
    let dist2 = to.dot(to);
    let dist = dist2.sqrt();
    let wi = to * (1.0 / dist);
    let cos_s = n.dot(wi);
    let cos_l = V3::new(0.0, -1.0, 0.0).dot(-wi);
    if cos_s <= 0.0 || cos_l <= 0.0 {
        return V3::splat(0.0);
    }
    let shadow = Ray {
        o: p + n * EPS,
        d: wi,
    };
    match trace_scene(&shadow) {
        Some(h) if h.mat_id == 3 => {
            let g = (cos_s * cos_l) / dist2;
            V3::new(18.0, 16.2, 12.6) * (g * 4.68 / std::f32::consts::PI)
        }
        _ => V3::splat(0.0),
    }
}

// ---- Fresnel ----
fn schlick(cosine: f32, ior: f32) -> f32 {
    let mut r0 = (1.0 - ior) / (1.0 + ior);
    r0 *= r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

// ---- Path trace ----
fn path_trace(primary: &Ray, rng: &mut Rng) -> (V3, u32) {
    let mut ray = Ray {
        o: primary.o,
        d: primary.d,
    };
    let mut throughput = V3::splat(1.0);
    let mut radiance = V3::splat(0.0);
    let mut rays = 0u32;
    let mut specular = true;

    for bounce in 0..MAX_BOUNCE {
        rays = rays.wrapping_add(1);
        let hit = match trace_scene(&ray) {
            Some(h) => h,
            None => break,
        };

        let mat = get_mat(hit.mat_id);

        // Emissive: main light (id 3) is covered by NEE on diffuse paths, so only
        // add it directly on specular/camera arrival to avoid double counting.
        if mat.kind == 3 {
            if specular || hit.mat_id != 3 {
                radiance = radiance + throughput * mat.albedo * mat.param;
            }
            break;
        }

        match mat.kind {
            0 => {
                // Diffuse + direct light sampling
                let albedo = if hit.mat_id == 9 {
                    checker(hit.pos)
                } else {
                    mat.albedo
                };
                radiance = radiance + throughput * albedo * sample_light(hit.pos, hit.normal, rng);
                ray.o = hit.pos + hit.normal * EPS;
                ray.d = rng.cos_hemisphere(hit.normal);
                throughput = throughput * albedo;
                specular = false;
            }
            1 => {
                // Metal
                let reflected = ray.d.norm().reflect(hit.normal);
                let fuzz = mat.param;
                ray.o = hit.pos + hit.normal * EPS;
                ray.d = (reflected
                    + V3::new(rng.f32() - 0.5, rng.f32() - 0.5, rng.f32() - 0.5) * fuzz)
                    .norm();
                if ray.d.dot(hit.normal) < 0.0 {
                    break;
                }
                throughput = throughput * mat.albedo;
                specular = true;
            }
            2 => {
                // Dielectric
                let ratio = if hit.front {
                    1.0 / mat.param
                } else {
                    mat.param
                };
                let unit = ray.d.norm();
                let cos_theta = (-unit).dot(hit.normal).min(1.0);
                let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
                let cannot_refract = ratio * sin_theta > 1.0;
                let should_reflect = schlick(cos_theta, ratio) > rng.f32();

                if cannot_refract || should_reflect {
                    ray.d = unit.reflect(hit.normal);
                } else {
                    ray.d = unit.refract(hit.normal, ratio);
                }
                ray.o = hit.pos + ray.d * EPS;
                specular = true;
            }
            _ => break,
        }

        // Russian roulette
        if bounce > 4 {
            let p = throughput.max_comp().max(0.05);
            if rng.f32() > p {
                break;
            }
            throughput = throughput / p;
        }
    }

    (radiance, rays)
}

// ---- Camera ----
struct Camera {
    eye: V3,
    right: V3,
    up: V3,
    forward: V3,
    ndc_bias_x: f32,
    ndc_bias_y: f32,
    ndc_scale_x: f32,
    ndc_scale_y: f32,
    focus_dist: f32,
    aperture: f32,
}

fn build_camera(w: u32, h: u32, yaw: f32, pitch: f32) -> Camera {
    let aspect = w as f32 / h as f32;
    let fov_scale = (62.0_f32.to_radians() * 0.5).tan();
    let target = V3::new(0.0, 2.0, 0.0);
    let dist = 9.5;
    let eye = target
        + V3::new(
            dist * pitch.cos() * yaw.sin(),
            dist * pitch.sin(),
            dist * pitch.cos() * yaw.cos(),
        );
    let forward = (target - eye).norm();
    let right = forward.cross(V3::new(0.0, 1.0, 0.0)).norm();
    let up = right.cross(forward);
    let focus_dist = (target - eye).len();
    let ndc_x_extent = aspect * fov_scale;

    Camera {
        eye,
        right,
        up,
        forward,
        ndc_bias_x: -ndc_x_extent,
        ndc_bias_y: fov_scale,
        ndc_scale_x: (2.0 * ndc_x_extent) / w as f32,
        ndc_scale_y: (2.0 * fov_scale) / h as f32,
        focus_dist,
        aperture: 0.035,
    }
}

fn make_ray(px: u32, py: u32, camera: &Camera, rng: &mut Rng) -> Ray {
    let fx = (px as f32 + rng.f32()) * camera.ndc_scale_x + camera.ndc_bias_x;
    let fy = camera.ndc_bias_y - (py as f32 + rng.f32()) * camera.ndc_scale_y;
    let dir = (camera.forward + camera.right * fx + camera.up * fy).norm();

    // Thin-lens depth of field
    let focus_point = camera.eye + dir * camera.focus_dist;
    let (dx, dy) = rng.unit_disk();
    let origin =
        camera.eye + camera.right * (dx * camera.aperture) + camera.up * (dy * camera.aperture);
    let dir = (focus_point - origin).norm();

    Ray { o: origin, d: dir }
}

// ---- Tone mapping (ACES) ----
fn tonemap(c: V3) -> V3 {
    let mapped = V3 {
        x: (c.x * (2.51 * c.x + 0.03)) / (c.x * (2.43 * c.x + 0.59) + 0.14),
        y: (c.y * (2.51 * c.y + 0.03)) / (c.y * (2.43 * c.y + 0.59) + 0.14),
        z: (c.z * (2.51 * c.z + 0.03)) / (c.z * (2.43 * c.z + 0.59) + 0.14),
    }
    .clamp01();
    V3 {
        x: mapped.x.powf(1.0 / 2.2),
        y: mapped.y.powf(1.0 / 2.2),
        z: mapped.z.powf(1.0 / 2.2),
    }
}

// ---- State ----
struct State {
    canvas: HtmlCanvasElement,
    ctx: CanvasRenderingContext2d,
    render_w: u32,
    render_h: u32,
    accum: Vec<f32>,
    pixels: Vec<u8>,
    sample_counts: Vec<u32>,
    rng_states: Vec<u32>,
    iteration: u32,
    next_pixel: usize,
    frame: u32,
    last_pass_ms: f64,
    last_pass_rays: u64,
    last_pass_samples: u64,
    pass_start_time: f64,
    pass_rays_accum: u64,
    pass_samples_accum: u64,

    stats_window_start: f64,
    window_passes: u32,
    window_samples: u64,
    window_rays: u64,
    passes_per_sec: f64,
    samples_per_sec: f64,
    rays_per_sec: f64,

    mouse_down: bool,
    last_mouse: (f64, f64),
    cam_yaw: f32,
    cam_pitch: f32,
    last_time: f64,
}

impl State {
    fn seed_for_pixel(idx: u32, salt: u32) -> u32 {
        idx.wrapping_mul(747796405)
            .wrapping_add(2891336453)
            .wrapping_add(salt.rotate_left(7))
    }

    fn reset_accumulation(&mut self) {
        self.accum.fill(0.0);
        self.pixels.fill(0);
        self.sample_counts.fill(0);
        self.iteration = 0;
        self.next_pixel = 0;
        self.pass_start_time = perf().now();
        self.pass_rays_accum = 0;
        self.pass_samples_accum = 0;
        self.last_pass_ms = 0.0;
        self.last_pass_rays = 0;
        self.last_pass_samples = 0;
    }

    fn reseed_rng_states(&mut self, salt: u32) {
        for (i, state) in self.rng_states.iter_mut().enumerate() {
            *state = Self::seed_for_pixel(i as u32, salt);
        }
    }

    fn resize(&mut self) {
        let dpr = window().device_pixel_ratio();
        let ww = window().inner_width().unwrap().as_f64().unwrap();
        let wh = window().inner_height().unwrap().as_f64().unwrap();

        self.canvas.set_width((ww * dpr) as u32);
        self.canvas.set_height((wh * dpr) as u32);

        // Full-resolution internal rendering.
        let rw = (ww * dpr) as u32;
        let rh = (wh * dpr) as u32;
        if rw != self.render_w || rh != self.render_h {
            self.render_w = rw;
            self.render_h = rh;
            let n = (rw * rh) as usize;
            self.accum = vec![0.0; n * 3];
            self.pixels = vec![0; n * 4];
            self.sample_counts = vec![0; n];
            self.rng_states = vec![0; n];
            self.reseed_rng_states(self.frame.wrapping_mul(977));
            self.iteration = 0;
            self.next_pixel = 0;
            self.pass_start_time = perf().now();
            self.pass_rays_accum = 0;
            self.pass_samples_accum = 0;
        }
    }

    fn trace_pixel(&mut self, idx: usize, camera: &Camera) -> u32 {
        let mut rng = Rng(self.rng_states[idx]);
        let x = (idx as u32) % self.render_w;
        let y = (idx as u32) / self.render_w;
        let ray = make_ray(x, y, camera, &mut rng);
        let (color, rays) = path_trace(&ray, &mut rng);
        let color = V3::new(
            color.x.min(FIREFLY_CLAMP),
            color.y.min(FIREFLY_CLAMP),
            color.z.min(FIREFLY_CLAMP),
        );
        self.rng_states[idx] = rng.0;

        let spp = self.sample_counts[idx] + 1;
        self.sample_counts[idx] = spp;
        let inv_spp = 1.0 / spp as f32;

        let ai = idx * 3;
        self.accum[ai] += (color.x - self.accum[ai]) * inv_spp;
        self.accum[ai + 1] += (color.y - self.accum[ai + 1]) * inv_spp;
        self.accum[ai + 2] += (color.z - self.accum[ai + 2]) * inv_spp;

        let c = tonemap(V3::new(
            self.accum[ai],
            self.accum[ai + 1],
            self.accum[ai + 2],
        ));
        let pi = idx * 4;
        self.pixels[pi] = (c.x * 255.0) as u8;
        self.pixels[pi + 1] = (c.y * 255.0) as u8;
        self.pixels[pi + 2] = (c.z * 255.0) as u8;
        self.pixels[pi + 3] = 255;
        rays
    }

    fn draw_hud(&self) {
        let total_pixels = (self.render_w * self.render_h) as usize;
        let progress = if total_pixels == 0 {
            0.0
        } else {
            self.next_pixel as f64 / total_pixels as f64
        };
        let text = format!(
            "CPU iter {} ({:.1}%) | pass {:.2} ms | passes/s {:.2} | samples/s {:.2}M | rays/s {:.2}M | rays/pass {}",
            self.iteration,
            progress * 100.0,
            self.last_pass_ms,
            self.passes_per_sec,
            self.samples_per_sec / 1_000_000.0,
            self.rays_per_sec / 1_000_000.0,
            self.last_pass_rays
        );

        let dpr = window().device_pixel_ratio();
        let pad = (8.0 * dpr).max(8.0);
        let font_px = (12.0 * dpr).max(12.0);
        self.ctx.set_font(&format!(
            "{}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
            font_px as u32
        ));
        self.ctx.set_text_baseline("top");

        let width = ((text.len() as f64) * font_px * 0.62 + pad * 2.0).max(260.0);
        let height = font_px + pad * 1.6;

        self.ctx.set_fill_style_str("rgba(0, 0, 0, 0.55)");
        self.ctx.fill_rect(pad, pad, width, height);
        self.ctx.set_fill_style_str("rgba(245, 245, 245, 0.95)");
        let _ = self.ctx.fill_text(&text, pad * 1.5, pad * 1.3);
    }

    fn frame(&mut self) {
        let now = perf().now();
        self.last_time = now;
        self.frame = self.frame.wrapping_add(1);
        self.resize();

        let w = self.render_w;
        let h = self.render_h;
        if w == 0 || h == 0 {
            return;
        }

        let camera = build_camera(w, h, self.cam_yaw, self.cam_pitch);
        let total_pixels = (w * h) as usize;
        let min_pixels = ((total_pixels / 192).max(1024)).min(16384);
        let start = perf().now();

        let mut traced = 0usize;
        let mut rays_this_frame = 0u64;
        while traced < min_pixels || perf().now() - start < FRAME_BUDGET_MS {
            let remain = total_pixels - self.next_pixel;
            let batch = remain.min(PIXEL_BATCH);
            for _ in 0..batch {
                rays_this_frame += self.trace_pixel(self.next_pixel, &camera) as u64;
                self.next_pixel += 1;
                self.pass_samples_accum += 1;
            }
            traced += batch;
            self.pass_rays_accum += rays_this_frame;
            self.window_samples += batch as u64;
            self.window_rays += rays_this_frame;
            rays_this_frame = 0;

            if self.next_pixel >= total_pixels {
                let now = perf().now();
                self.next_pixel = 0;
                self.iteration = self.iteration.wrapping_add(1);
                self.last_pass_ms = now - self.pass_start_time;
                self.last_pass_rays = self.pass_rays_accum;
                self.last_pass_samples = self.pass_samples_accum;
                self.pass_start_time = now;
                self.pass_rays_accum = 0;
                self.pass_samples_accum = 0;
                self.window_passes = self.window_passes.wrapping_add(1);
            }

            if traced >= total_pixels {
                break;
            }
        }

        let now = perf().now();
        let elapsed = now - self.stats_window_start;
        if elapsed >= 1000.0 {
            let sec = elapsed / 1000.0;
            self.passes_per_sec = self.window_passes as f64 / sec;
            self.samples_per_sec = self.window_samples as f64 / sec;
            self.rays_per_sec = self.window_rays as f64 / sec;
            self.stats_window_start = now;
            self.window_passes = 0;
            self.window_samples = 0;
            self.window_rays = 0;
        }

        let data =
            ImageData::new_with_u8_clamped_array_and_sh(Clamped(&self.pixels), w, h).unwrap();
        self.ctx.put_image_data(&data, 0.0, 0.0).unwrap();
        self.draw_hud();
    }
}

fn start_cpu() {
    std::panic::set_hook(Box::new(|info| {
        log(&format!("PANIC: {}", info));
    }));

    let canvas: HtmlCanvasElement = document()
        .create_element("canvas")
        .unwrap()
        .unchecked_into();
    canvas.style().set_property("position", "fixed").unwrap();
    canvas.style().set_property("top", "0").unwrap();
    canvas.style().set_property("left", "0").unwrap();
    canvas.style().set_property("width", "100%").unwrap();
    canvas.style().set_property("height", "100%").unwrap();
    canvas
        .style()
        .set_property("image-rendering", "auto")
        .unwrap();
    document().body().unwrap().set_inner_html("");
    document()
        .body()
        .unwrap()
        .style()
        .set_property("margin", "0")
        .unwrap();
    document()
        .body()
        .unwrap()
        .style()
        .set_property("overflow", "hidden")
        .unwrap();
    document().body().unwrap().append_child(&canvas).unwrap();

    let dpr = window().device_pixel_ratio();
    let ww = window().inner_width().unwrap().as_f64().unwrap();
    let wh = window().inner_height().unwrap().as_f64().unwrap();
    let rw = (ww * dpr) as u32;
    let rh = (wh * dpr) as u32;
    canvas.set_width(rw);
    canvas.set_height(rh);

    let ctx: CanvasRenderingContext2d = canvas.get_context("2d").unwrap().unwrap().unchecked_into();
    let n = (rw * rh) as usize;
    let mut rng_states = vec![0; n];
    for (i, state) in rng_states.iter_mut().enumerate() {
        *state = State::seed_for_pixel(i as u32, 0xA53B_1F27);
    }

    let state = Rc::new(RefCell::new(State {
        canvas: canvas.clone(),
        ctx,
        render_w: rw,
        render_h: rh,
        accum: vec![0.0; n * 3],
        pixels: vec![0; n * 4],
        sample_counts: vec![0; n],
        rng_states,
        iteration: 0,
        next_pixel: 0,
        frame: 0,
        last_pass_ms: 0.0,
        last_pass_rays: 0,
        last_pass_samples: 0,
        pass_start_time: perf().now(),
        pass_rays_accum: 0,
        pass_samples_accum: 0,
        stats_window_start: perf().now(),
        window_passes: 0,
        window_samples: 0,
        window_rays: 0,
        passes_per_sec: 0.0,
        samples_per_sec: 0.0,
        rays_per_sec: 0.0,
        mouse_down: false,
        last_mouse: (0.0, 0.0),
        cam_yaw: 0.0,
        cam_pitch: 0.15,
        last_time: perf().now(),
    }));

    // Mouse input
    {
        let s = state.clone();
        let cb = Closure::<dyn FnMut(MouseEvent)>::new(move |e: MouseEvent| {
            let mut st = s.borrow_mut();
            st.mouse_down = true;
            st.last_mouse = (e.client_x() as f64, e.client_y() as f64);
        });
        canvas
            .add_event_listener_with_callback("mousedown", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }
    {
        let s = state.clone();
        let cb = Closure::<dyn FnMut(MouseEvent)>::new(move |_: MouseEvent| {
            s.borrow_mut().mouse_down = false;
        });
        canvas
            .add_event_listener_with_callback("mouseup", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }
    {
        let s = state.clone();
        let cb = Closure::<dyn FnMut(MouseEvent)>::new(move |e: MouseEvent| {
            let mut st = s.borrow_mut();
            if st.mouse_down {
                let dx = e.client_x() as f64 - st.last_mouse.0;
                let dy = e.client_y() as f64 - st.last_mouse.1;
                st.cam_yaw += dx as f32 * 0.005;
                st.cam_pitch = (st.cam_pitch + dy as f32 * 0.005).clamp(-1.2, 1.2);
                st.last_mouse = (e.client_x() as f64, e.client_y() as f64);
                st.reset_accumulation();
                let salt = st.frame.wrapping_mul(3907).wrapping_add(11);
                st.reseed_rng_states(salt);
            }
        });
        canvas
            .add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }

    // Animation loop
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    let s = state.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        s.borrow_mut().frame();
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());

    console_log!(
        "Ray tracer running at full resolution ({}x{}) — drag to orbit camera",
        rw,
        rh
    );
}

#[wasm_bindgen(start)]
pub fn start() {
    std::panic::set_hook(Box::new(|info| {
        log(&format!("PANIC: {}", info));
    }));

    spawn_local(async move {
        if let Err(e) = wgpu_renderer::start().await {
            log(&format!(
                "wgpu(webgl2) init failed, falling back to CPU renderer: {}",
                e
            ));
            start_cpu();
        }
    });
}
