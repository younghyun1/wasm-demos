use std::cell::RefCell;
use std::rc::Rc;

use bytemuck::{Pod, Zeroable};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{Document, HtmlCanvasElement, HtmlElement, MouseEvent, Window};

const MAX_BOUNCE: u32 = 10;
const FOV_DEG: f32 = 62.0;
const CAMERA_DIST: f32 = 9.5;
const CAMERA_LENS_RADIUS: f32 = 0.035;
const MAX_TRACE_PASSES_PER_FRAME: u32 = 24;
const RENDER_SCALE_MAX: f64 = 1.5;

const TRACE_WGSL: &str = r#"
struct Uniforms {
    resolution: vec4<f32>,
    seed_data: vec4<u32>,
    cam_eye: vec4<f32>,
    cam_forward: vec4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    cam_params: vec4<f32>,
};

@group(0) @binding(0) var prev_accum: texture_2d<f32>;
@group(0) @binding(1) var<uniform> u: Uniforms;

const EPS: f32 = 0.001;
const INF: f32 = 1e20;
const PI2: f32 = 6.28318530718;
const MAX_BOUNCE: u32 = 10u;

const MAT_DIFFUSE: u32 = 0u;
const MAT_METAL: u32 = 1u;
const MAT_DIELECTRIC: u32 = 2u;
const MAT_EMISSIVE: u32 = 3u;

const PI: f32 = 3.14159265359;
const FIREFLY_CLAMP: f32 = 16.0;
const ROOM_HALF_W: f32 = 4.2;
const ROOM_HALF_D: f32 = 3.0;
const ROOM_H: f32 = 5.6;
const LIGHT_MIN: vec3<f32> = vec3<f32>(-1.3, ROOM_H - 0.02, -0.9);
const LIGHT_U: vec3<f32> = vec3<f32>(2.6, 0.0, 0.0);
const LIGHT_V: vec3<f32> = vec3<f32>(0.0, 0.0, 1.8);
const LIGHT_NRM: vec3<f32> = vec3<f32>(0.0, -1.0, 0.0);
const LIGHT_AREA: f32 = 4.68;
const LIGHT_EMIT: vec3<f32> = vec3<f32>(18.0, 16.2, 12.6);

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct HitGeom {
    hit: bool,
    t: f32,
    n: vec3<f32>,
    front: bool,
};

struct Hit {
    hit: bool,
    t: f32,
    p: vec3<f32>,
    n: vec3<f32>,
    mat_id: u32,
    front: bool,
};

struct Material {
    kind: u32,
    albedo: vec3<f32>,
    param: f32,
};

fn pcg_hash(state: ptr<function, u32>) -> u32 {
    let old = *state;
    *state = old * 747796405u + 2891336453u;
    let word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand1(state: ptr<function, u32>) -> f32 {
    return f32(pcg_hash(state)) * (1.0 / 4294967296.0);
}

fn sample_disk(state: ptr<function, u32>) -> vec2<f32> {
    let r = sqrt(rand1(state));
    let a = PI2 * rand1(state);
    return vec2<f32>(cos(a), sin(a)) * r;
}

fn sample_cos_hemisphere(n: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    let r1 = rand1(state);
    let r2 = rand1(state);
    let phi = PI2 * r1;
    let cos_theta = sqrt(r2);
    let sin_theta = sqrt(1.0 - r2);

    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.99);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return normalize(t * (cos(phi) * sin_theta) + b * (sin(phi) * sin_theta) + n * cos_theta);
}

fn hit_sphere(center: vec3<f32>, radius: f32, ro: vec3<f32>, rd: vec3<f32>, t_max: f32) -> HitGeom {
    let oc = ro - center;
    let a = dot(rd, rd);
    let hb = dot(oc, rd);
    let c = dot(oc, oc) - radius * radius;
    let disc = hb * hb - a * c;
    if (disc < 0.0) {
        return HitGeom(false, 0.0, vec3<f32>(0.0), true);
    }

    let sq = sqrt(disc);
    var t = (-hb - sq) / a;
    if (t < EPS || t > t_max) {
        t = (-hb + sq) / a;
        if (t < EPS || t > t_max) {
            return HitGeom(false, 0.0, vec3<f32>(0.0), true);
        }
    }

    let p = ro + rd * t;
    let outward = (p - center) / radius;
    let front = dot(rd, outward) < 0.0;
    let normal = select(-outward, outward, front);
    return HitGeom(true, t, normal, front);
}

fn hit_quad(corner: vec3<f32>, u_vec: vec3<f32>, v_vec: vec3<f32>, ro: vec3<f32>, rd: vec3<f32>, t_max: f32) -> HitGeom {
    let n = cross(u_vec, v_vec);
    let normal = normalize(n);
    let denom = dot(normal, rd);
    if (abs(denom) < 1e-8) {
        return HitGeom(false, 0.0, vec3<f32>(0.0), true);
    }

    let d = dot(normal, corner);
    let t = (d - dot(normal, ro)) / denom;
    if (t < EPS || t > t_max) {
        return HitGeom(false, 0.0, vec3<f32>(0.0), true);
    }

    let p = ro + rd * t;
    let planar = p - corner;
    let n_sq = dot(n, n);
    let alpha = dot(cross(planar, v_vec), n) / n_sq;
    let beta = dot(cross(u_vec, planar), n) / n_sq;
    if (alpha < 0.0 || alpha > 1.0 || beta < 0.0 || beta > 1.0) {
        return HitGeom(false, 0.0, vec3<f32>(0.0), true);
    }

    let front = denom < 0.0;
    let norm = select(-normal, normal, front);
    return HitGeom(true, t, norm, front);
}

fn hit_box(bmin: vec3<f32>, bmax: vec3<f32>, ro: vec3<f32>, rd: vec3<f32>, t_max: f32) -> HitGeom {
    let inv = vec3<f32>(1.0) / rd;
    let t0 = (bmin - ro) * inv;
    let t1 = (bmax - ro) * inv;
    let tsmall = min(t0, t1);
    let tbig = max(t0, t1);
    let tn = max(max(tsmall.x, tsmall.y), tsmall.z);
    let tf = min(min(tbig.x, tbig.y), tbig.z);
    if (tn > tf || tf < EPS) {
        return HitGeom(false, 0.0, vec3<f32>(0.0), true);
    }
    var t = tn;
    var on_far = false;
    if (t < EPS) {
        t = tf;
        on_far = true;
    }
    if (t < EPS || t > t_max) {
        return HitGeom(false, 0.0, vec3<f32>(0.0), true);
    }
    var n = vec3<f32>(0.0);
    if (!on_far) {
        if (tn == tsmall.x) {
            n = vec3<f32>(-sign(rd.x), 0.0, 0.0);
        } else if (tn == tsmall.y) {
            n = vec3<f32>(0.0, -sign(rd.y), 0.0);
        } else {
            n = vec3<f32>(0.0, 0.0, -sign(rd.z));
        }
    } else {
        if (tf == tbig.x) {
            n = vec3<f32>(sign(rd.x), 0.0, 0.0);
        } else if (tf == tbig.y) {
            n = vec3<f32>(0.0, sign(rd.y), 0.0);
        } else {
            n = vec3<f32>(0.0, 0.0, sign(rd.z));
        }
    }
    let front = dot(rd, n) < 0.0;
    let nn = select(-n, n, front);
    return HitGeom(true, t, nn, front);
}

fn checker(p: vec3<f32>) -> vec3<f32> {
    let ix = i32(floor(p.x * 0.7));
    let iz = i32(floor(p.z * 0.7));
    let parity = (ix + iz) & 1;
    return select(vec3<f32>(0.18, 0.19, 0.22), vec3<f32>(0.82, 0.80, 0.76), parity == 0);
}

fn trace_scene(ro: vec3<f32>, rd: vec3<f32>) -> Hit {
    var closest = INF;
    var result = Hit(false, 0.0, vec3<f32>(0.0), vec3<f32>(0.0), 0u, true);

    // Room shell
    var h = hit_quad(vec3<f32>(-ROOM_HALF_W, 0.0, -ROOM_HALF_D), vec3<f32>(2.0 * ROOM_HALF_W, 0.0, 0.0), vec3<f32>(0.0, 0.0, 2.0 * ROOM_HALF_D), ro, rd, closest);
    if (h.hit) { closest = h.t; result = Hit(true, h.t, ro + rd * h.t, h.n, 9u, h.front); }
    h = hit_quad(vec3<f32>(-ROOM_HALF_W, ROOM_H, -ROOM_HALF_D), vec3<f32>(2.0 * ROOM_HALF_W, 0.0, 0.0), vec3<f32>(0.0, 0.0, 2.0 * ROOM_HALF_D), ro, rd, closest);
    if (h.hit) { closest = h.t; result = Hit(true, h.t, ro + rd * h.t, h.n, 10u, h.front); }
    h = hit_quad(vec3<f32>(-ROOM_HALF_W, 0.0, -ROOM_HALF_D), vec3<f32>(2.0 * ROOM_HALF_W, 0.0, 0.0), vec3<f32>(0.0, ROOM_H, 0.0), ro, rd, closest);
    if (h.hit) { closest = h.t; result = Hit(true, h.t, ro + rd * h.t, h.n, 11u, h.front); }
    h = hit_quad(vec3<f32>(-ROOM_HALF_W, 0.0, -ROOM_HALF_D), vec3<f32>(0.0, 0.0, 2.0 * ROOM_HALF_D), vec3<f32>(0.0, ROOM_H, 0.0), ro, rd, closest);
    if (h.hit) { closest = h.t; result = Hit(true, h.t, ro + rd * h.t, h.n, 1u, h.front); }
    h = hit_quad(vec3<f32>(ROOM_HALF_W, 0.0, -ROOM_HALF_D), vec3<f32>(0.0, 0.0, 2.0 * ROOM_HALF_D), vec3<f32>(0.0, ROOM_H, 0.0), ro, rd, closest);
    if (h.hit) { closest = h.t; result = Hit(true, h.t, ro + rd * h.t, h.n, 2u, h.front); }

    // Area light
    h = hit_quad(LIGHT_MIN, LIGHT_U, LIGHT_V, ro, rd, closest);
    if (h.hit) { closest = h.t; result = Hit(true, h.t, ro + rd * h.t, h.n, 3u, h.front); }

    // Spheres
    h = hit_sphere(vec3<f32>(1.5, 1.0, 0.2), 1.0, ro, rd, closest);
    if (h.hit) { closest = h.t; result = Hit(true, h.t, ro + rd * h.t, h.n, 5u, h.front); }
    h = hit_sphere(vec3<f32>(-1.8, 0.8, -0.4), 0.8, ro, rd, closest);
    if (h.hit) { closest = h.t; result = Hit(true, h.t, ro + rd * h.t, h.n, 4u, h.front); }
    h = hit_sphere(vec3<f32>(2.7, 0.42, -1.6), 0.42, ro, rd, closest);
    if (h.hit) { closest = h.t; result = Hit(true, h.t, ro + rd * h.t, h.n, 8u, h.front); }
    h = hit_sphere(vec3<f32>(-1.5, 0.55, 1.5), 0.55, ro, rd, closest);
    if (h.hit) { closest = h.t; result = Hit(true, h.t, ro + rd * h.t, h.n, 6u, h.front); }
    h = hit_sphere(vec3<f32>(0.2, 0.35, 1.9), 0.35, ro, rd, closest);
    if (h.hit) { closest = h.t; result = Hit(true, h.t, ro + rd * h.t, h.n, 7u, h.front); }
    h = hit_sphere(vec3<f32>(-3.0, 0.3, 1.4), 0.3, ro, rd, closest);
    if (h.hit) { closest = h.t; result = Hit(true, h.t, ro + rd * h.t, h.n, 16u, h.front); }

    // Boxes (metal pillar + diffuse riser)
    h = hit_box(vec3<f32>(-3.2, 0.0, -2.4), vec3<f32>(-2.4, 2.2, -1.6), ro, rd, closest);
    if (h.hit) { closest = h.t; result = Hit(true, h.t, ro + rd * h.t, h.n, 14u, h.front); }
    h = hit_box(vec3<f32>(2.0, 0.0, 1.4), vec3<f32>(2.8, 0.5, 2.2), ro, rd, closest);
    if (h.hit) { closest = h.t; result = Hit(true, h.t, ro + rd * h.t, h.n, 15u, h.front); }

    return result;
}

fn get_material(mat_id: u32) -> Material {
    switch mat_id {
        case 0u: { return Material(MAT_DIFFUSE, vec3<f32>(0.73), 0.0); }
        case 1u: { return Material(MAT_DIFFUSE, vec3<f32>(0.63, 0.26, 0.30), 0.0); }
        case 2u: { return Material(MAT_DIFFUSE, vec3<f32>(0.20, 0.52, 0.50), 0.0); }
        case 3u: { return Material(MAT_EMISSIVE, vec3<f32>(1.0, 0.9, 0.7), 18.0); }
        case 4u: { return Material(MAT_METAL, vec3<f32>(0.95, 0.78, 0.45), 0.02); }
        case 5u: { return Material(MAT_DIELECTRIC, vec3<f32>(1.0), 1.5); }
        case 6u: { return Material(MAT_DIFFUSE, vec3<f32>(0.16, 0.28, 0.72), 0.0); }
        case 7u: { return Material(MAT_DIFFUSE, vec3<f32>(0.72, 0.55, 0.22), 0.0); }
        case 8u: { return Material(MAT_METAL, vec3<f32>(0.86, 0.88, 0.95), 0.14); }
        case 9u: { return Material(MAT_DIFFUSE, vec3<f32>(0.5), 0.0); }
        case 10u: { return Material(MAT_DIFFUSE, vec3<f32>(0.78, 0.78, 0.80), 0.0); }
        case 11u: { return Material(MAT_DIFFUSE, vec3<f32>(0.70, 0.72, 0.75), 0.0); }
        case 12u: { return Material(MAT_DIFFUSE, vec3<f32>(0.74, 0.34, 0.62), 0.0); }
        case 13u: { return Material(MAT_DIFFUSE, vec3<f32>(0.80, 0.71, 0.28), 0.0); }
        case 14u: { return Material(MAT_METAL, vec3<f32>(0.80, 0.84, 0.92), 0.04); }
        case 15u: { return Material(MAT_DIFFUSE, vec3<f32>(0.80, 0.45, 0.20), 0.0); }
        case 16u: { return Material(MAT_EMISSIVE, vec3<f32>(1.0, 0.55, 0.22), 6.0); }
        default: { return Material(MAT_DIFFUSE, vec3<f32>(0.5), 0.0); }
    }
}

fn schlick(cosine: f32, ior: f32) -> f32 {
    var r0 = (1.0 - ior) / (1.0 + ior);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

fn sample_light(p: vec3<f32>, n: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    let q = LIGHT_MIN + LIGHT_U * rand1(state) + LIGHT_V * rand1(state);
    let to = q - p;
    let dist2 = dot(to, to);
    let dist = sqrt(dist2);
    let wi = to / dist;
    let cos_s = dot(n, wi);
    let cos_l = dot(LIGHT_NRM, -wi);
    if (cos_s <= 0.0 || cos_l <= 0.0) {
        return vec3<f32>(0.0);
    }
    let shadow = trace_scene(p + n * EPS, wi);
    if (!shadow.hit || shadow.mat_id != 3u) {
        return vec3<f32>(0.0);
    }
    let g = (cos_s * cos_l) / dist2;
    return LIGHT_EMIT * (g * LIGHT_AREA / PI);
}

fn path_trace(ro0: vec3<f32>, rd0: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    var ro = ro0;
    var rd = rd0;
    var throughput = vec3<f32>(1.0);
    var radiance = vec3<f32>(0.0);
    var specular = true;

    for (var bounce = 0u; bounce < MAX_BOUNCE; bounce = bounce + 1u) {
        let hit = trace_scene(ro, rd);
        if (!hit.hit) {
            break;
        }

        let mat = get_material(hit.mat_id);
        if (mat.kind == MAT_EMISSIVE) {
            // Main light (id 3) is handled by NEE on diffuse paths; only add it
            // directly when reached via a specular/camera ray to avoid double counting.
            if (specular || hit.mat_id != 3u) {
                radiance = radiance + throughput * mat.albedo * mat.param;
            }
            break;
        }

        if (mat.kind == MAT_DIFFUSE) {
            var albedo = mat.albedo;
            if (hit.mat_id == 9u) {
                albedo = checker(hit.p);
            }
            radiance = radiance + throughput * albedo * sample_light(hit.p, hit.n, state);
            ro = hit.p + hit.n * EPS;
            rd = sample_cos_hemisphere(hit.n, state);
            throughput = throughput * albedo;
            specular = false;
        } else if (mat.kind == MAT_METAL) {
            let reflected = reflect(normalize(rd), hit.n);
            ro = hit.p + hit.n * EPS;
            rd = normalize(reflected + vec3<f32>(rand1(state) - 0.5, rand1(state) - 0.5, rand1(state) - 0.5) * mat.param);
            if (dot(rd, hit.n) < 0.0) {
                break;
            }
            throughput = throughput * mat.albedo;
            specular = true;
        } else if (mat.kind == MAT_DIELECTRIC) {
            let ratio = select(mat.param, 1.0 / mat.param, hit.front);
            let unit = normalize(rd);
            let cos_theta = min(dot(-unit, hit.n), 1.0);
            let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
            let cannot_refract = ratio * sin_theta > 1.0;
            let reflect_prob = schlick(cos_theta, ratio);
            if (cannot_refract || reflect_prob > rand1(state)) {
                rd = reflect(unit, hit.n);
            } else {
                rd = refract(unit, hit.n, ratio);
            }
            ro = hit.p + rd * EPS;
            specular = true;
        }

        if (bounce > 4u) {
            var p = max(max(throughput.x, throughput.y), throughput.z);
            p = max(p, 0.05);
            if (rand1(state) > p) {
                break;
            }
            throughput = throughput / p;
        }
    }

    return radiance;
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );

    var out: VsOut;
    out.pos = vec4<f32>(p[index], 0.0, 1.0);
    out.uv = p[index] * 0.5 + vec2<f32>(0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let px = vec2<u32>(u32(in.pos.x), u32(in.pos.y));
    var rng_state = px.x ^ (px.y * 4093u) ^ (u.seed_data.z * 26699u) ^ u.seed_data.x;
    rng_state = pcg_hash(&rng_state);

    let jitter = vec2<f32>(rand1(&rng_state), rand1(&rng_state)) - vec2<f32>(0.5);
    let pixel = in.pos.xy + jitter;

    let ndc_x = ((2.0 * pixel.x) / u.resolution.x - 1.0) * u.cam_params.x;
    let ndc_y = ((2.0 * pixel.y) / u.resolution.y - 1.0) * u.cam_params.y;

    var dir = normalize(u.cam_forward.xyz + u.cam_right.xyz * ndc_x + u.cam_up.xyz * ndc_y);
    let focus_point = u.cam_eye.xyz + dir * u.cam_params.z;
    let lens = sample_disk(&rng_state) * u.cam_params.w;
    let origin = u.cam_eye.xyz + u.cam_right.xyz * lens.x + u.cam_up.xyz * lens.y;
    dir = normalize(focus_point - origin);

    var sample_color = path_trace(origin, dir, &rng_state);
    sample_color = min(sample_color, vec3<f32>(FIREFLY_CLAMP));
    let prev = textureLoad(prev_accum, vec2<i32>(i32(px.x), i32(px.y)), 0).rgb;
    let n = f32(u.seed_data.y);
    let accum = (prev * n + sample_color) / (n + 1.0);
    return vec4<f32>(accum, 1.0);
}
"#;

const PRESENT_WGSL: &str = r#"
@group(0) @binding(0) var accum_tex: texture_2d<f32>;
@group(0) @binding(1) var accum_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

fn tonemap(c: vec3<f32>) -> vec3<f32> {
    let mapped = clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
    return pow(mapped, vec3<f32>(1.0 / 2.2));
}

fn ign(p: vec2<f32>) -> f32 {
    let m = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(m.z * fract(dot(p, m.xy)));
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );

    var out: VsOut;
    out.pos = vec4<f32>(p[index], 0.0, 1.0);
    out.uv = p[index] * 0.5 + vec2<f32>(0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(accum_tex, accum_sampler, in.uv).rgb;
    let mapped = tonemap(c);

    // Dither prior to UNORM/SRGB quantization to reduce visible banding in smooth gradients.
    let d = (vec3<f32>(
        ign(in.pos.xy),
        ign(in.pos.yx + vec2<f32>(17.0, 31.0)),
        ign(in.pos.xy + vec2<f32>(47.0, 13.0))
    ) - vec3<f32>(0.5)) / 255.0;
    let out_c = clamp(mapped + d, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(out_c, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    resolution: [f32; 4],
    seed_data: [u32; 4],
    cam_eye: [f32; 4],
    cam_forward: [f32; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    cam_params: [f32; 4],
}

struct Camera {
    eye: [f32; 3],
    forward: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    tan_half_fov_x: f32,
    tan_half_fov_y: f32,
    focus_dist: f32,
}

struct AccumTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

#[derive(Default)]
struct Stats {
    last_pass_ms: f64,
    last_frame_ms: f64,
    last_frame_passes: u32,
    passes_per_sec: f64,
    samples_per_sec: f64,
    rays_per_sec_est: f64,
    rays_per_pass_est: u64,
    total_rays_est: u64,
    total_samples: u64,
    window_start: f64,
    window_passes: u32,
    window_samples: u64,
    window_rays_est: u64,
}

pub struct GpuState {
    canvas: HtmlCanvasElement,
    hud: HtmlElement,

    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,

    trace_pipeline: wgpu::RenderPipeline,
    present_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,

    trace_bind_group_layout: wgpu::BindGroupLayout,
    present_bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,

    accum_targets: [AccumTarget; 2],
    trace_bind_groups: [wgpu::BindGroup; 2],
    present_bind_groups: [wgpu::BindGroup; 2],

    read_idx: usize,
    write_idx: usize,
    frame: u32,
    iteration: u32,
    seed_salt: u32,

    mouse_down: bool,
    last_mouse: (f64, f64),
    cam_yaw: f32,
    cam_pitch: f32,

    last_frame_start: f64,
    adaptive_passes: u32,

    stats: Stats,
}

fn window() -> Result<Window, String> {
    web_sys::window().ok_or_else(|| "window() unavailable".to_string())
}

fn document() -> Result<Document, String> {
    window()?
        .document()
        .ok_or_else(|| "document() unavailable".to_string())
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    let _ = window()
        .and_then(|w| {
            w.request_animation_frame(f.as_ref().unchecked_ref())
                .map_err(|_| "requestAnimationFrame failed".to_string())
        })
        .unwrap();
}

fn perf_now() -> f64 {
    window()
        .ok()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}

fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_len(v: [f32; 3]) -> f32 {
    vec3_dot(v, v).sqrt()
}

fn vec3_norm(v: [f32; 3]) -> [f32; 3] {
    let l = vec3_len(v);
    if l <= 1e-8 {
        v
    } else {
        [v[0] / l, v[1] / l, v[2] / l]
    }
}

fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn build_camera(w: u32, h: u32, yaw: f32, pitch: f32) -> Camera {
    let aspect = w as f32 / h as f32;
    let tan_half_fov_y = (FOV_DEG.to_radians() * 0.5).tan();
    let tan_half_fov_x = tan_half_fov_y * aspect;

    let target = [0.0, 2.0, 0.0];
    let eye = vec3_add(
        target,
        [
            CAMERA_DIST * pitch.cos() * yaw.sin(),
            CAMERA_DIST * pitch.sin(),
            CAMERA_DIST * pitch.cos() * yaw.cos(),
        ],
    );

    let forward = vec3_norm(vec3_sub(target, eye));
    let right = vec3_norm(vec3_cross(forward, [0.0, 1.0, 0.0]));
    let up = vec3_cross(right, forward);
    let focus_dist = vec3_len(vec3_sub(target, eye));

    Camera {
        eye,
        forward,
        right,
        up,
        tan_half_fov_x,
        tan_half_fov_y,
        focus_dist,
    }
}

fn create_canvas_and_hud() -> Result<(HtmlCanvasElement, HtmlElement), String> {
    let doc = document()?;
    let body = doc
        .body()
        .ok_or_else(|| "document.body missing".to_string())?;

    body.set_inner_html("");
    body.style()
        .set_property("margin", "0")
        .map_err(|_| "failed to set body margin".to_string())?;
    body.style()
        .set_property("overflow", "hidden")
        .map_err(|_| "failed to set body overflow".to_string())?;
    body.style()
        .set_property("background", "#000")
        .map_err(|_| "failed to set body background".to_string())?;

    let canvas: HtmlCanvasElement = doc
        .create_element("canvas")
        .map_err(|_| "failed to create canvas".to_string())?
        .unchecked_into();
    canvas
        .style()
        .set_property("position", "fixed")
        .map_err(|_| "failed to style canvas".to_string())?;
    canvas
        .style()
        .set_property("top", "0")
        .map_err(|_| "failed to style canvas".to_string())?;
    canvas
        .style()
        .set_property("left", "0")
        .map_err(|_| "failed to style canvas".to_string())?;
    canvas
        .style()
        .set_property("width", "100%")
        .map_err(|_| "failed to style canvas".to_string())?;
    canvas
        .style()
        .set_property("height", "100%")
        .map_err(|_| "failed to style canvas".to_string())?;
    body.append_child(&canvas)
        .map_err(|_| "failed to append canvas".to_string())?;

    let hud: HtmlElement = doc
        .create_element("pre")
        .map_err(|_| "failed to create hud".to_string())?
        .unchecked_into();
    hud.style()
        .set_property("position", "fixed")
        .map_err(|_| "failed to style hud".to_string())?;
    hud.style()
        .set_property("top", "0")
        .map_err(|_| "failed to style hud".to_string())?;
    hud.style()
        .set_property("left", "0")
        .map_err(|_| "failed to style hud".to_string())?;
    hud.style()
        .set_property("margin", "0")
        .map_err(|_| "failed to style hud".to_string())?;
    hud.style()
        .set_property("padding", "10px 12px")
        .map_err(|_| "failed to style hud".to_string())?;
    hud.style()
        .set_property("color", "rgba(245, 245, 245, 0.95)")
        .map_err(|_| "failed to style hud".to_string())?;
    hud.style()
        .set_property("background", "rgba(0, 0, 0, 0.55)")
        .map_err(|_| "failed to style hud".to_string())?;
    hud.style()
        .set_property(
            "font",
            "12px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
        )
        .map_err(|_| "failed to style hud".to_string())?;
    hud.style()
        .set_property("pointer-events", "none")
        .map_err(|_| "failed to style hud".to_string())?;
    hud.style()
        .set_property("user-select", "none")
        .map_err(|_| "failed to style hud".to_string())?;
    hud.style()
        .set_property("z-index", "10")
        .map_err(|_| "failed to style hud".to_string())?;
    body.append_child(&hud)
        .map_err(|_| "failed to append hud".to_string())?;

    Ok((canvas, hud))
}

impl GpuState {
    async fn new(canvas: HtmlCanvasElement, hud: HtmlElement) -> Result<Self, String> {
        let dpr = window()?.device_pixel_ratio();
        let ww = window()?
            .inner_width()
            .map_err(|_| "innerWidth failed".to_string())?
            .as_f64()
            .ok_or_else(|| "innerWidth missing".to_string())?;
        let wh = window()?
            .inner_height()
            .map_err(|_| "innerHeight failed".to_string())?
            .as_f64()
            .ok_or_else(|| "innerHeight missing".to_string())?;
        let scale = dpr.min(RENDER_SCALE_MAX);
        let width = (ww * scale).max(1.0) as u32;
        let height = (wh * scale).max(1.0) as u32;
        canvas.set_width(width);
        canvas.set_height(height);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| format!("create_surface failed: {e}"))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .map_err(|e| format!("request_adapter failed: {e}"))?;

        let limits = wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                label: Some("ray-tracer-device"),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("request_device failed: {e}"))?;

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
            wgpu::PresentMode::Fifo
        } else {
            caps.present_modes[0]
        };

        let alpha_mode = caps
            .alpha_modes
            .iter()
            .copied()
            .find(|m| *m == wgpu::CompositeAlphaMode::Opaque)
            .unwrap_or(caps.alpha_modes[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trace uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let trace_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("trace bind layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let present_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("present bind layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("accum sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let trace_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("trace shader"),
            source: wgpu::ShaderSource::Wgsl(TRACE_WGSL.into()),
        });
        let present_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("present shader"),
            source: wgpu::ShaderSource::Wgsl(PRESENT_WGSL.into()),
        });

        let trace_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("trace pipeline layout"),
                bind_group_layouts: &[&trace_bind_group_layout],
                immediate_size: 0,
            });
        let trace_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("trace pipeline"),
            layout: Some(&trace_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &trace_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &trace_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let present_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("present pipeline layout"),
                bind_group_layouts: &[&present_bind_group_layout],
                immediate_size: 0,
            });
        let present_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("present pipeline"),
            layout: Some(&present_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &present_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &present_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let (accum_targets, trace_bind_groups, present_bind_groups) = Self::create_accum_resources(
            &device,
            width,
            height,
            &trace_bind_group_layout,
            &present_bind_group_layout,
            &sampler,
            &uniform_buffer,
        );

        let mut state = Self {
            canvas,
            hud,
            surface,
            device,
            queue,
            surface_config,
            trace_pipeline,
            present_pipeline,
            uniform_buffer,
            trace_bind_group_layout,
            present_bind_group_layout,
            sampler,
            accum_targets,
            trace_bind_groups,
            present_bind_groups,
            read_idx: 0,
            write_idx: 1,
            frame: 0,
            iteration: 0,
            seed_salt: 0xA53B_1F27,
            mouse_down: false,
            last_mouse: (0.0, 0.0),
            cam_yaw: 0.0,
            cam_pitch: 0.15,
            last_frame_start: 0.0,
            adaptive_passes: 4,
            stats: Stats::default(),
        };

        state.stats.window_start = perf_now();
        state.clear_accumulation();
        state.update_hud();

        Ok(state)
    }

    fn create_accum_texture(device: &wgpu::Device, width: u32, height: u32) -> AccumTarget {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("accum texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        AccumTarget {
            _texture: texture,
            view,
        }
    }

    fn create_accum_resources(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        trace_layout: &wgpu::BindGroupLayout,
        present_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        uniform_buffer: &wgpu::Buffer,
    ) -> ([AccumTarget; 2], [wgpu::BindGroup; 2], [wgpu::BindGroup; 2]) {
        let targets = [
            Self::create_accum_texture(device, width, height),
            Self::create_accum_texture(device, width, height),
        ];

        let trace_bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trace bg 0"),
            layout: trace_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&targets[0].view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let trace_bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trace bg 1"),
            layout: trace_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&targets[1].view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let present_bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("present bg 0"),
            layout: present_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&targets[0].view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        let present_bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("present bg 1"),
            layout: present_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&targets[1].view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        (targets, [trace_bg0, trace_bg1], [present_bg0, present_bg1])
    }

    fn clear_accumulation(&mut self) {
        self.iteration = 0;
        self.frame = 0;
        self.read_idx = 0;
        self.write_idx = 1;
        self.seed_salt = self.seed_salt.wrapping_add(0x9E37_79B9);

        self.stats.last_pass_ms = 0.0;
        self.stats.last_frame_ms = 0.0;
        self.stats.last_frame_passes = 0;
        self.stats.rays_per_pass_est = 0;
        self.stats.total_rays_est = 0;
        self.stats.total_samples = 0;
        self.stats.window_start = perf_now();
        self.stats.window_passes = 0;
        self.stats.window_samples = 0;
        self.stats.window_rays_est = 0;
        self.stats.passes_per_sec = 0.0;
        self.stats.samples_per_sec = 0.0;
        self.stats.rays_per_sec_est = 0.0;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("clear accum"),
            });
        for target in &self.accum_targets {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        self.queue.submit(Some(encoder.finish()));
    }

    fn resize_if_needed(&mut self) {
        let Ok(w) = window() else {
            return;
        };
        let dpr = w.device_pixel_ratio();
        let Ok(inner_w) = w.inner_width() else {
            return;
        };
        let Ok(inner_h) = w.inner_height() else {
            return;
        };
        let Some(ww) = inner_w.as_f64() else {
            return;
        };
        let Some(wh) = inner_h.as_f64() else {
            return;
        };

        let scale = dpr.min(RENDER_SCALE_MAX);
        let new_w = (ww * scale).max(1.0) as u32;
        let new_h = (wh * scale).max(1.0) as u32;
        if new_w == self.surface_config.width && new_h == self.surface_config.height {
            return;
        }

        self.canvas.set_width(new_w);
        self.canvas.set_height(new_h);
        self.surface_config.width = new_w;
        self.surface_config.height = new_h;
        self.surface.configure(&self.device, &self.surface_config);

        let (targets, trace_bgs, present_bgs) = Self::create_accum_resources(
            &self.device,
            new_w,
            new_h,
            &self.trace_bind_group_layout,
            &self.present_bind_group_layout,
            &self.sampler,
            &self.uniform_buffer,
        );
        self.accum_targets = targets;
        self.trace_bind_groups = trace_bgs;
        self.present_bind_groups = present_bgs;
        self.clear_accumulation();
    }

    fn update_stats(&mut self, pass_ms: f64) {
        self.stats.last_pass_ms = pass_ms;
        self.stats.window_passes = self.stats.window_passes.wrapping_add(1);

        let samples = self.surface_config.width as u64 * self.surface_config.height as u64;
        let rays_est = samples * MAX_BOUNCE as u64;

        self.stats.rays_per_pass_est = rays_est;
        self.stats.total_samples = self.stats.total_samples.wrapping_add(samples);
        self.stats.total_rays_est = self.stats.total_rays_est.wrapping_add(rays_est);

        self.stats.window_samples = self.stats.window_samples.wrapping_add(samples);
        self.stats.window_rays_est = self.stats.window_rays_est.wrapping_add(rays_est);

        let now = perf_now();
        let elapsed_ms = now - self.stats.window_start;
        if elapsed_ms >= 1000.0 {
            let sec = elapsed_ms / 1000.0;
            self.stats.passes_per_sec = self.stats.window_passes as f64 / sec;
            self.stats.samples_per_sec = self.stats.window_samples as f64 / sec;
            self.stats.rays_per_sec_est = self.stats.window_rays_est as f64 / sec;
            self.stats.window_start = now;
            self.stats.window_passes = 0;
            self.stats.window_samples = 0;
            self.stats.window_rays_est = 0;
            web_sys::console::log_1(
                &format!(
                    "[ray-tracer/wgpu-webgl2] pass {:.2}ms | passes/s {:.2} | samples/s {:.2}M | rays/s(est) {:.2}M",
                    self.stats.last_pass_ms,
                    self.stats.passes_per_sec,
                    self.stats.samples_per_sec / 1_000_000.0,
                    self.stats.rays_per_sec_est / 1_000_000.0
                )
                .into(),
            );
        }
    }

    fn update_hud(&self) {
        self.hud.set_text_content(Some(&format!(
            "backend: wgpu(webgl2)\nresolution: {}x{}\niteration: {}\npass ms: {:.2}\npasses/frame: {}\nframe ms: {:.2}\npasses/s: {:.2}\nsamples/s: {:.2}M\nrays/s (est): {:.2}M\nrays/pass (est): {}\ntotal rays (est): {}\ndrag: orbit camera",
            self.surface_config.width,
            self.surface_config.height,
            self.iteration,
            self.stats.last_pass_ms,
            self.stats.last_frame_passes,
            self.stats.last_frame_ms,
            self.stats.passes_per_sec,
            self.stats.samples_per_sec / 1_000_000.0,
            self.stats.rays_per_sec_est / 1_000_000.0,
            self.stats.rays_per_pass_est,
            self.stats.total_rays_est,
        )));
    }

    fn write_uniforms(&mut self) {
        let w = self.surface_config.width;
        let h = self.surface_config.height;
        let cam = build_camera(w, h, self.cam_yaw, self.cam_pitch);

        let uniforms = Uniforms {
            resolution: [w as f32, h as f32, 0.0, 0.0],
            seed_data: [self.seed_salt, self.iteration, self.frame, 0],
            cam_eye: [cam.eye[0], cam.eye[1], cam.eye[2], 0.0],
            cam_forward: [cam.forward[0], cam.forward[1], cam.forward[2], 0.0],
            cam_right: [cam.right[0], cam.right[1], cam.right[2], 0.0],
            cam_up: [cam.up[0], cam.up[1], cam.up[2], 0.0],
            cam_params: [
                cam.tan_half_fov_x,
                cam.tan_half_fov_y,
                cam.focus_dist,
                CAMERA_LENS_RADIUS,
            ],
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    fn trace_one_pass(&mut self) -> f64 {
        self.write_uniforms();
        let pass_start = perf_now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("trace encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("trace pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.accum_targets[self.write_idx].view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.trace_pipeline);
            pass.set_bind_group(0, &self.trace_bind_groups[self.read_idx], &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        let tmp = self.read_idx;
        self.read_idx = self.write_idx;
        self.write_idx = tmp;
        self.iteration = self.iteration.wrapping_add(1);
        self.frame = self.frame.wrapping_add(1);
        perf_now() - pass_start
    }

    fn present_current_surface(&mut self) {
        let surface_tex = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(_) => {
                self.surface.configure(&self.device, &self.surface_config);
                match self.surface.get_current_texture() {
                    Ok(frame) => frame,
                    Err(_) => return,
                }
            }
        };
        let surface_view = surface_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("present encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("present pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.present_pipeline);
            pass.set_bind_group(0, &self.present_bind_groups[self.read_idx], &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        surface_tex.present();
    }

    fn draw_frame(&mut self) {
        self.resize_if_needed();

        let now = perf_now();
        let frame_delta = if self.last_frame_start > 0.0 {
            now - self.last_frame_start
        } else {
            16.0
        };
        self.last_frame_start = now;

        // GPU work is async on WebGL2, so per-pass CPU timing is meaningless.
        // Adapt the pass count from real inter-frame wall time to hold ~60fps.
        if frame_delta > 18.0 && self.adaptive_passes > 1 {
            self.adaptive_passes -= 1;
        } else if frame_delta < 14.0 && self.adaptive_passes < MAX_TRACE_PASSES_PER_FRAME {
            self.adaptive_passes += 1;
        }
        // One pass per frame while dragging keeps camera latency low.
        let target_passes = if self.mouse_down {
            1
        } else {
            self.adaptive_passes
        };

        let frame_start = now;
        let mut passes_this_frame = 0u32;
        while passes_this_frame < target_passes {
            let pass_ms = self.trace_one_pass();
            passes_this_frame = passes_this_frame.wrapping_add(1);
            self.update_stats(pass_ms);
        }

        self.present_current_surface();
        self.stats.last_frame_passes = passes_this_frame;
        self.stats.last_frame_ms = perf_now() - frame_start;
        self.update_hud();
    }
}

fn install_input_handlers(state: &Rc<RefCell<GpuState>>) -> Result<(), String> {
    let canvas = state.borrow().canvas.clone();

    {
        let s = state.clone();
        let cb = Closure::<dyn FnMut(MouseEvent)>::new(move |e: MouseEvent| {
            let mut st = s.borrow_mut();
            st.mouse_down = true;
            st.last_mouse = (e.client_x() as f64, e.client_y() as f64);
        });
        canvas
            .add_event_listener_with_callback("mousedown", cb.as_ref().unchecked_ref())
            .map_err(|_| "failed to add mousedown".to_string())?;
        cb.forget();
    }

    {
        let s = state.clone();
        let cb = Closure::<dyn FnMut(MouseEvent)>::new(move |_: MouseEvent| {
            s.borrow_mut().mouse_down = false;
        });
        window()?
            .add_event_listener_with_callback("mouseup", cb.as_ref().unchecked_ref())
            .map_err(|_| "failed to add mouseup".to_string())?;
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
                st.clear_accumulation();
            }
        });
        window()?
            .add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref())
            .map_err(|_| "failed to add mousemove".to_string())?;
        cb.forget();
    }

    Ok(())
}

pub async fn start() -> Result<(), String> {
    let (canvas, hud) = create_canvas_and_hud()?;
    let state = Rc::new(RefCell::new(GpuState::new(canvas, hud).await?));

    install_input_handlers(&state)?;

    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    let s = state.clone();

    *g.borrow_mut() = Some(Closure::new(move || {
        s.borrow_mut().draw_frame();
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));

    request_animation_frame(g.borrow().as_ref().unwrap());

    let (w, h) = {
        let st = state.borrow();
        (st.surface_config.width, st.surface_config.height)
    };
    web_sys::console::log_1(
        &format!(
            "[ray-tracer/wgpu-webgl2] running at full resolution {}x{}",
            w, h
        )
        .into(),
    );

    Ok(())
}
