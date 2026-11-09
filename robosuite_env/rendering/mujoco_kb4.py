# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Render a MuJoCo ``mjvScene`` with our own GL program.

``mjr_render`` is fixed-function OpenGL with a pinhole camera, so a fisheye wrist
camera needs a renderer whose vertex stage we own. This module re-implements the
geom passes of MuJoCo 3.8.1 ``render_gl3.c`` and ``render_context.c``: per-vertex
lighting, texgen, skybox and MSAA colour, then a second pass at pixel centres for
segmentation ids and ray distance, where ``mjr_render`` segments. Scene features
it does not reproduce raise ``NotImplementedError`` instead of rendering differently.

With a KB4 camera, tessellation splits each triangle until no edge spans more than
``_MAX_STEP`` of view angle, so that straight edges bend as the lens bends them. Each
fragment then intersects its pixel's ray with the original triangle and interpolates
the vertex colour and texture coordinates there, as perspective-correct interpolation
does in ``mjr_render``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import ctypes
import functools

import mujoco  # first: with MUJOCO_GL=egl it sets PyOpenGL's platform to EGL
import cv2 as cv
import numpy as np
from OpenGL import GL

from .wrist_render_profiles import (
    DEFAULT_POLICY_WRIST_RENDER_PROFILE,
    camera_rays,
    get_wrist_render_profile,
    wrist_source_intrinsics,
)

_F32 = np.float32
_MAX_LIGHTS = 8
_RECORD_VEC4 = 14
_TEXGEN_2D, _UV_2D, _TEXGEN_CUBE = 1, 2, 3  # texture modes; 0 is untextured
_MESH_ARRAYS = ("mesh_vert", "mesh_normal", "mesh_texcoord", "mesh_face", "mesh_facenormal", "mesh_facetexcoord")
_MAX_STEP = np.radians(1.0)
_THETA_MARGIN = np.radians(2.0)  # drawn beyond theta_max, so that edge pixels average drawn samples

_VERTEX_SHADER = """
#version 460
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
struct Draw {
  mat4 model_view;
  vec4 normal_matrix[3];
  vec4 rgba;
  vec4 emission_specular;
  vec4 params;  // shininess exponent, texture mode, segid + 1, unlit
  vec4 texgen[3];
  ivec4 mesh;  // vertadr, normaladr, texcoordadr, 1 for a mesh
};
layout(std430, binding = 0) readonly buffer Draws { Draw draws[]; };
// MuJoCo's mesh arrays as stored in mjModel; a mesh draw's gl_VertexID is a face corner.
layout(std430, binding = 1) readonly buffer MeshVert { float mesh_vert[]; };
layout(std430, binding = 2) readonly buffer MeshNormal { float mesh_normal[]; };
layout(std430, binding = 3) readonly buffer MeshTexcoord { float mesh_texcoord[]; };
layout(std430, binding = 4) readonly buffer MeshFace { int mesh_face[]; };
layout(std430, binding = 5) readonly buffer MeshFaceNormal { int mesh_facenormal[]; };
layout(std430, binding = 6) readonly buffer MeshFaceTexcoord { int mesh_facetexcoord[]; };
uniform mat4 u_projection;
uniform int u_nlight;
uniform vec3 u_ambient;
uniform vec4 u_light_position[8];
uniform vec4 u_light_spot[8];
uniform vec4 u_light_attenuation[8];
uniform vec3 u_light_ambient[8];
uniform vec3 u_light_diffuse[8];
uniform vec3 u_light_specular[8];
out vec4 v_color;
out vec3 v_texcoord;
out vec3 v_eye;
flat out ivec2 v_mode_segid;
flat out int v_behind;

vec3 mesh_vertex(int i) {
  return vec3(mesh_vert[3 * i], mesh_vert[3 * i + 1], mesh_vert[3 * i + 2]);
}

void main() {
  Draw d = draws[gl_BaseInstance];
  vec3 position = a_position, normal = a_normal;
  vec2 uv = a_uv;
  if (d.mesh.w != 0) {
    // mjr_uploadMesh: without texcoords, the face normal replaces a vertex normal at a crease.
    int first = gl_VertexID - gl_VertexID % 3;
    vec3 p[3];
    for (int k = 0; k < 3; k++) p[k] = mesh_vertex(mesh_face[first + k] + d.mesh.x);
    position = p[gl_VertexID - first];
    int n = 3 * (mesh_facenormal[gl_VertexID] + d.mesh.y);
    normal = vec3(mesh_normal[n], mesh_normal[n + 1], mesh_normal[n + 2]);
    if (d.mesh.z >= 0) {
      int t = 2 * (mesh_facetexcoord[gl_VertexID] + d.mesh.z);
      uv = vec2(mesh_texcoord[t], mesh_texcoord[t + 1]);
    } else {
      vec3 face = cross(p[1] - p[0], p[2] - p[0]);
      float len = length(face);
      face = len < 1e-10 ? vec3(0.0, 0.0, 1.0) : face * (1.0 / len);
      if (dot(normal, face) < 0.8) normal = face;
    }
  }
  vec4 eye = d.model_view * vec4(position, 1.0);
  gl_Position = u_projection * eye;
  v_eye = eye.xyz;
  int mode = int(d.params.y);
  v_mode_segid = ivec2(mode, int(d.params.z));
  v_behind = int(d.normal_matrix[0].w);
  vec4 object = vec4(position, 1.0);
  v_texcoord = (mode == 1 || mode == 3)
      ? vec3(dot(d.texgen[0], object), dot(d.texgen[1], object), dot(d.texgen[2], object))
      : vec3(uv, 0.0);
  if (d.params.w > 0.5) {
    v_color = d.rgba;
    return;
  }
  // OpenGL 2.1 fixed-function lighting: local viewer, single colour, one side.
  vec3 n = normalize(mat3(d.normal_matrix[0].xyz, d.normal_matrix[1].xyz,
                          d.normal_matrix[2].xyz) * normal);
  vec3 to_eye = normalize(-eye.xyz);
  vec3 rgb = d.emission_specular.rgb + u_ambient * d.rgba.rgb;
  for (int i = 0; i < u_nlight; i++) {
    vec3 to_light;
    float factor = 1.0;
    if (u_light_position[i].w == 0.0) {
      to_light = normalize(u_light_position[i].xyz);
    } else {
      vec3 offset = u_light_position[i].xyz - eye.xyz;
      float dist = length(offset);
      to_light = offset / dist;
      vec4 k = u_light_attenuation[i];
      factor = 1.0 / (k.x + k.y * dist + k.z * dist * dist);
      float c = dot(-to_light, u_light_spot[i].xyz);
      factor *= c < u_light_spot[i].w ? 0.0 : (k.w > 0.0 ? pow(max(c, 0.0), k.w) : 1.0);
    }
    float nl = dot(n, to_light);
    vec3 c = d.rgba.rgb * (u_light_ambient[i] + max(nl, 0.0) * u_light_diffuse[i]);
    if (nl > 0.0) {
      float nh = max(dot(n, normalize(to_light + to_eye)), 0.0);
      float shine = d.params.x > 0.0 ? pow(nh, d.params.x) : 1.0;
      c += shine * d.emission_specular.w * u_light_specular[i];
    }
    rgb += factor * c;
  }
  v_color = clamp(vec4(rgb, d.rgba.a), 0.0, 1.0);
}
"""

_FRAGMENT_SHADER = """
#version 460
#ifdef KB4
flat in vec3 f_eye[3];
flat in vec4 f_color[3];
flat in vec3 f_texcoord[3];
flat in ivec2 f_mode_segid;
uniform sampler2D u_rays;  // unit eye-space ray of each pixel centre, rows from the bottom
uniform vec2 u_depth_range;  // near, far [m]
#else
in vec4 v_color;
in vec3 v_texcoord;
in vec3 v_eye;
flat in ivec2 v_mode_segid;
#endif
uniform sampler2D u_texture_2d;
uniform samplerCube u_texture_cube;
uniform vec4 u_fog;  // start, end, enabled
uniform vec3 u_fog_color;
layout(location = 0) out vec4 o_color;
layout(location = 1) out int o_segid;
layout(location = 2) out float o_distance;

void main() {
#ifdef KB4
  vec3 ray = texelFetch(u_rays, ivec2(gl_FragCoord.xy), 0).xyz;
  vec3 e1 = f_eye[1] - f_eye[0], e2 = f_eye[2] - f_eye[0], n = cross(e1, e2);
  float distance = dot(n, f_eye[0]) / dot(n, ray);
  if (!(distance > 0.0)) discard;  // only where a sub-triangle strays off a triangle seen edge-on
  vec3 eye = distance * ray, q = eye - f_eye[0];
  vec2 b = vec2(dot(cross(q, e2), n), dot(cross(e1, q), n)) / dot(n, n);
  vec3 w = vec3(1.0 - b.x - b.y, b);
  vec4 c = w.x * f_color[0] + w.y * f_color[1] + w.z * f_color[2];
  vec3 texcoord = w.x * f_texcoord[0] + w.y * f_texcoord[1] + w.z * f_texcoord[2];
  ivec2 mode_segid = f_mode_segid;
  float fog_z = max(max(abs(eye.x), abs(eye.y)), abs(eye.z));  // eye z of the axis-aligned 90-degree pinhole view that sees the point
#else
  vec4 c = v_color;
  vec3 texcoord = v_texcoord;
  ivec2 mode_segid = v_mode_segid;
  float distance = length(v_eye);
  float fog_z = abs(v_eye.z);
#endif
  if (mode_segid.x == 1 || mode_segid.x == 2) {
    c *= texture(u_texture_2d, texcoord.xy);
  } else if (mode_segid.x == 3) {
    c *= texture(u_texture_cube, texcoord);
  }
  if (u_fog.z > 0.5) {
    float f = clamp((u_fog.y - fog_z) / (u_fog.y - u_fog.x), 0.0, 1.0);
    c.rgb = mix(u_fog_color, c.rgb, f);
  }
  o_color = c;
  o_segid = mode_segid.y;
  o_distance = distance;
#ifdef EXACT_DEPTH
  gl_FragDepth = clamp((u_depth_range.y - distance) / (u_depth_range.y - u_depth_range.x), 0.0, 1.0);
#endif
}
"""

# KB4 only; an outer level of 0 discards the patch.
_CONTROL_SHADER = """
#version 460
layout(vertices = 3) out;
in vec4 v_color[];
in vec3 v_texcoord[];
in vec3 v_eye[];
flat in ivec2 v_mode_segid[];
flat in int v_behind[];
out vec4 c_color[];
out vec3 c_texcoord[];
out vec3 c_eye[];
out ivec2 c_mode_segid[];
uniform float u_step;  // largest view angle [rad] of a sub-triangle edge
uniform float u_theta_cut;  // largest drawn angle [rad] off the optical axis
uniform int u_cull;

float segment_distance(vec3 a, vec3 b) {
  vec3 ab = b - a;
  return length(a + clamp(-dot(a, ab) / dot(ab, ab), 0.0, 1.0) * ab);
}

// Both patches of an edge must give it the same level, or a crack opens, so the ends are sorted first.
float edge_level(vec3 a, vec3 b) {
  if (a.x > b.x || (a.x == b.x && (a.y > b.y || (a.y == b.y && a.z > b.z)))) {
    vec3 t = a;
    a = b;
    b = t;
  }
  return clamp(ceil(length(b - a) / (max(segment_distance(a, b), 1e-9) * u_step)), 1.0, 64.0);
}

float triangle_distance(vec3 a, vec3 b, vec3 c, vec3 n) {
  vec3 p = n * (dot(n, a) / dot(n, n));
  bool inside = dot(cross(b - a, p - a), n) >= 0.0 && dot(cross(c - b, p - b), n) >= 0.0
      && dot(cross(a - c, p - c), n) >= 0.0;
  return inside ? length(p) : min(min(segment_distance(a, b), segment_distance(b, c)), segment_distance(c, a));
}

float off_axis(vec3 p) {
  return atan(length(p.xy), -p.z);
}

void main() {
  c_color[gl_InvocationID] = v_color[gl_InvocationID];
  c_texcoord[gl_InvocationID] = v_texcoord[gl_InvocationID];
  c_eye[gl_InvocationID] = v_eye[gl_InvocationID];
  c_mode_segid[gl_InvocationID] = v_mode_segid[gl_InvocationID];
  if (gl_InvocationID != 0) return;
  vec3 a = v_eye[0], b = v_eye[1], c = v_eye[2];
  vec3 n = cross(b - a, c - a);
  bool front = dot(n, a) < 0.0;
  // no two points of the triangle are further apart in view angle than this
  float extent = max(max(length(b - a), length(c - b)), length(a - c)) / max(triangle_distance(a, b, c, n), 1e-9);
  // Two equal corners make the lens bend the edge into a sliver that covers pixels, and FMA leaves n nonzero there.
  if (a == b || b == c || c == a || (u_cull != 0 && front == (v_behind[0] != 0))
      || min(min(off_axis(a), off_axis(b)), off_axis(c)) - extent > u_theta_cut) {
    gl_TessLevelOuter[0] = gl_TessLevelOuter[1] = gl_TessLevelOuter[2] = 0.0;
    return;
  }
  gl_TessLevelOuter[0] = edge_level(b, c);
  gl_TessLevelOuter[1] = edge_level(c, a);
  gl_TessLevelOuter[2] = edge_level(a, b);
  gl_TessLevelInner[0] = clamp(ceil(extent / u_step), 1.0, 64.0);
}
"""

_EVALUATION_SHADER = """
#version 460
layout(triangles, equal_spacing, ccw) in;
in vec4 c_color[];
in vec3 c_texcoord[];
in vec3 c_eye[];
in ivec2 c_mode_segid[];
uniform vec4 u_focal;  // fx, fy, cx, cy [px]; pixel centres at integers, rows down
uniform vec4 u_distortion;  // k1..k4
uniform vec2 u_raster;  // width, height [px]
uniform float u_theta_cut;  // largest drawn angle [rad] off the optical axis
uniform vec2 u_depth_range;  // near, far [m]
flat out vec3 f_eye[3];
flat out vec4 f_color[3];
flat out vec3 f_texcoord[3];
flat out ivec2 f_mode_segid;

void main() {
  // Two patches sum a shared edge point in different orders; without precise, FMA can move it and open a crack.
  precise vec3 p = gl_TessCoord.x * c_eye[0] + gl_TessCoord.y * c_eye[1] + gl_TessCoord.z * c_eye[2];
  float rho = length(p.xy);
  float theta = atan(rho, -p.z);
  float t2 = theta * theta;
  vec4 k = u_distortion;
  float theta_d = theta * (1.0 + t2 * (k.x + t2 * (k.y + t2 * (k.z + t2 * k.w))));
  vec2 offset = rho > 0.0 ? p.xy * (theta_d / rho) : vec2(0.0);
  vec2 ndc = 2.0 * (u_focal.zw + vec2(u_focal.x * offset.x, -u_focal.y * offset.y) + 0.5) / u_raster.xy - 1.0;
  float depth = (u_depth_range.y - length(p)) / (u_depth_range.y - u_depth_range.x);
  gl_Position = vec4(ndc.x, -ndc.y, depth, 1.0);
  gl_ClipDistance[0] = u_theta_cut - theta;
  for (int i = 0; i < 3; i++) {
    f_eye[i] = c_eye[i];
    f_color[i] = c_color[i];
    f_texcoord[i] = c_texcoord[i];
  }
  f_mode_segid = c_mode_segid[0];
}
"""


@dataclass(frozen=True)
class SceneImage:
    """One render; rows start at the top like ``mujoco.Renderer.render``."""

    rgb: np.ndarray | None  # [H, W, 3] uint8; None from render(color=False)
    segid: np.ndarray  # [H, W] int32 index into scene.geoms, -1 where no geom
    distance: np.ndarray  # [H, W] float32 ray distance [m], inf where no geom


def _translate(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def _scale(x, y, z):
    return np.diag((x, y, z, 1.0))


def _triangles(*corners):
    """Interleave ``(position, normal)`` corner arrays into triangle rows."""

    return np.stack(
        [np.concatenate((p, n, np.zeros((len(p), 2), _F32)), axis=1) for p, n in corners],
        axis=1,
    ).reshape(-1, 8)


def _quads(v1, v2, v3, v4):
    # NVIDIA splits GL_QUADS along v1-v3; the other diagonal moves edge pixels.
    return np.concatenate((_triangles(v1, v2, v3), _triangles(v1, v3, v4)), axis=0)


def _arc(n):
    j = np.arange(n + 1, dtype=np.float64)
    az = (2.0 * np.pi * j / n).astype(_F32)
    return np.cos(az), np.sin(az)


def _sphere_vertex(cos_az, sin_az, el, sign):
    ce, se = np.cos(_F32(el)), np.sin(_F32(el))
    v = np.stack((cos_az * ce, sin_az * ce, np.full_like(cos_az, _F32(sign) + se)), axis=1)
    return v, v - np.asarray((0, 0, sign), _F32)


def _pole(cos_az, sin_az, el, sign, z_pole, flip):
    v, n = _sphere_vertex(cos_az, sin_az, el, sign)
    tip = np.tile(np.asarray((0, 0, z_pole), _F32), (len(v) - 1, 1))
    tip_n = np.tile(np.asarray((0, 0, 1 if flip > 0 else -1), _F32), (len(v) - 1, 1))
    a, b = (v[:-1], n[:-1]), (v[1:], n[1:])
    return _triangles(a, b, (tip, tip_n)) if flip > 0 else _triangles((tip, tip_n), b, a)


def _band(cos_az, sin_az, el1, el2, sign, flip):
    v1, n1 = _sphere_vertex(cos_az, sin_az, el1, sign)
    v2, n2 = _sphere_vertex(cos_az, sin_az, el2, sign)
    c = ((v1[:-1], n1[:-1]), (v1[1:], n1[1:]), (v2[1:], n2[1:]), (v2[:-1], n2[:-1]))
    return _quads(*c) if flip > 0 else _quads(*c[::-1])


def _sphere(nslice, nstack):
    cos_az, sin_az = _arc(nslice)
    half = nstack // 2
    parts = [
        _pole(cos_az, sin_az, (0.5 * np.pi * s * (half - 1)) / half, 0, s, s) for s in (-1, 1)
    ]
    for s in (-1, 1):
        parts += [
            _band(cos_az, sin_az, (0.5 * np.pi * s * i) / half, (0.5 * np.pi * s * (i + 1)) / half, 0, s)
            for i in range(half - 1)
        ]
    return np.concatenate(parts)


def _half_sphere(sign, nslice, nstack):
    cos_az, sin_az = _arc(nslice)
    parts = [_pole(cos_az, sin_az, (np.pi / 2 * sign * (nstack - 1)) / nstack, sign, 2 * sign, sign)]
    parts += [
        _band(cos_az, sin_az, (np.pi / 2 * sign * i) / nstack, (np.pi / 2 * sign * (i + 1)) / nstack, sign, sign)
        for i in range(nstack - 1)
    ]
    return np.concatenate(parts)


def _disk(sign, nslice, nstack):
    cos_az, sin_az = _arc(nslice)
    normal = np.asarray((0, 0, -1 if sign == 0 else sign), _F32)

    def ring(r):
        p = np.stack((cos_az * r, sin_az * r, np.full_like(cos_az, sign)), axis=1)
        n = np.tile(normal, (len(p), 1))
        return (p[:-1], n[:-1]), (p[1:], n[1:])

    tip = (np.tile(np.asarray((0, 0, sign), _F32), (nslice, 1)), np.tile(normal, (nslice, 1)))
    first = (*ring(_F32(1.0) / _F32(nstack)), tip)
    parts = [_triangles(*first) if sign > 0 else _triangles(*first[::-1])]
    for i in range(nstack - 1):
        inner, outer = ring(_F32(i + 1) / _F32(nstack)), ring(_F32(i + 2) / _F32(nstack))
        c = (outer[0], outer[1], inner[1], inner[0])
        parts.append(_quads(*c) if sign > 0 else _quads(*c[::-1]))
    return np.concatenate(parts)


def _cylinder(nslice, nstack):
    cos_az, sin_az = _arc(nslice)
    radius = np.sqrt(cos_az * cos_az + sin_az * sin_az)
    normal = np.stack((cos_az / radius, sin_az / radius, np.zeros_like(cos_az)), axis=1)
    parts = []
    for i in range(nstack):
        h1 = _F32(2 * i) / _F32(nstack) - _F32(1)
        h2 = _F32(2 * (i + 1)) / _F32(nstack) - _F32(1)
        r1 = np.stack((cos_az, sin_az, np.full_like(cos_az, h1)), axis=1)
        r2 = np.stack((cos_az, sin_az, np.full_like(cos_az, h2)), axis=1)
        a, b = (r1[:-1], normal[:-1]), (r1[1:], normal[1:])
        parts.append(_quads(a, b, (r2[1:], normal[1:]), (r2[:-1], normal[:-1])))
    return np.concatenate(parts)


def _box(numquads):
    d = _F32(2.0) / _F32(numquads)
    g = [d * _F32(i) - _F32(1) for i in range(numquads + 1)]
    faces = []
    for x in range(numquads):
        for y in range(numquads):
            a0, a1, b0, b1 = g[x], g[x + 1], g[y], g[y + 1]
            faces += [
                ((0, 0, 1), ((a0, b0, 1), (a1, b0, 1), (a1, b1, 1), (a0, b1, 1))),
                ((0, 0, -1), ((a0, b1, -1), (a1, b1, -1), (a1, b0, -1), (a0, b0, -1))),
                ((1, 0, 0), ((1, a0, b0), (1, a1, b0), (1, a1, b1), (1, a0, b1))),
                ((-1, 0, 0), ((-1, a0, b1), (-1, a1, b1), (-1, a1, b0), (-1, a0, b0))),
                ((0, -1, 0), ((a0, -1, b0), (a1, -1, b0), (a1, -1, b1), (a0, -1, b1))),
                ((0, 1, 0), ((a0, 1, b1), (a1, 1, b1), (a1, 1, b0), (a0, 1, b0))),
            ]
    return _quad_list(faces)


def _quad_list(faces):
    normal = np.asarray([f[0] for f in faces], _F32)
    corners = np.asarray([f[1] for f in faces], _F32)
    return _quads(*((corners[:, k], normal) for k in range(4)))


def _plane_default(numquads):
    d = 2.0 / numquads
    return _quad_list([
        ((0, 0, 1), tuple((d * (x + i) - 1, d * (y + j) - 1, 0) for i, j in ((0, 0), (1, 0), (1, 1), (0, 1))))
        for x in range(numquads) for y in range(numquads)
    ])


def _plane_grid(model, geom_id):
    """``makePlane`` grid for one plane geom, in metres."""

    zfar = model.vis.map.zfar * model.stat.extent
    grids = []
    for k in range(2):
        size = float(model.geom_size[geom_id, k])
        if size > 0:
            step = max(float(model.geom_size[geom_id, 2]), size / (mujoco.mjMAXPLANEGRID - 2))
            n = int(np.floor(size / max(mujoco.mjMINVAL, step)))
            n -= n % 2
            pad = size - n * step
            edges = []
            for x in range(n + 2):
                left = -size if x == 0 else -size + pad + (x - 1) * step * 2
                right = left + (pad if x in (0, n + 1) else step * 2)
                edges.append(left)
            edges.append(max(left, right))
            grids.append(np.asarray(edges))
        else:
            matid = model.geom_matid[geom_id]
            repeat = model.mat_texrepeat[matid, k] if matid >= 0 else 0
            sx = 2 / repeat if repeat > 0 else 2.1 * zfar / (mujoco.mjMAXPLANEGRID - 2)
            d = (2.1 * zfar + 2 * sx) / mujoco.mjMAXPLANEGRID
            grids.append(d * np.arange(mujoco.mjMAXPLANEGRID + 1) - d * (mujoco.mjMAXPLANEGRID // 2))
    gx, gy = grids
    return _quad_list([
        ((0, 0, 1), ((gx[x], gy[y], 0), (gx[x + 1], gy[y], 0), (gx[x + 1], gy[y + 1], 0), (gx[x], gy[y + 1], 0)))
        for x in range(len(gx) - 1) for y in range(len(gy) - 1)
    ])


def _rbound(geom):
    """``mjv_rbound``, which the Python bindings do not expose."""

    if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM:
        return float(geom.modelrbound)
    s = np.asarray(geom.size, np.float64)
    return {
        mujoco.mjtGeom.mjGEOM_SPHERE: s[0],
        mujoco.mjtGeom.mjGEOM_CAPSULE: s[0] + s[2],
        mujoco.mjtGeom.mjGEOM_CYLINDER: np.hypot(s[0], s[2]),
        mujoco.mjtGeom.mjGEOM_BOX: np.linalg.norm(s),
    }.get(geom.type, s.max())


def _program(*defines):
    """Compile and link the shaders; on failure, delete what was created."""

    version, _, body = _FRAGMENT_SHADER.lstrip().partition("\n")
    fragment = "\n".join((version, *(f"#define {name}" for name in defines), body))
    stages = [(GL.GL_VERTEX_SHADER, _VERTEX_SHADER), (GL.GL_FRAGMENT_SHADER, fragment)]
    if "KB4" in defines:
        stages[1:1] = [(GL.GL_TESS_CONTROL_SHADER, _CONTROL_SHADER), (GL.GL_TESS_EVALUATION_SHADER, _EVALUATION_SHADER)]
    shaders, program = [], GL.glCreateProgram()
    try:
        for kind, source in stages:
            shaders.append(GL.glCreateShader(kind))
            GL.glShaderSource(shaders[-1], source)
            GL.glCompileShader(shaders[-1])
            if not GL.glGetShaderiv(shaders[-1], GL.GL_COMPILE_STATUS):
                raise RuntimeError(GL.glGetShaderInfoLog(shaders[-1]).decode())
            GL.glAttachShader(program, shaders[-1])
        GL.glLinkProgram(program)
        if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
            raise RuntimeError(GL.glGetProgramInfoLog(program).decode())
    except BaseException:
        GL.glDeleteProgram(program)
        raise
    finally:
        for shader in shaders:
            GL.glDeleteShader(shader)
    return program


@functools.lru_cache(maxsize=8)
def _ray_table(camera, theta_limit):
    """``camera_rays`` in eye space (x right, y up, z back) as RGBA rows from the bottom, for ``u_rays``."""

    rays, _ = camera_rays(camera, theta_limit)
    table = np.zeros((camera.height, camera.width, 4), _F32)
    table[..., :3] = rays[::-1] * np.asarray((1, -1, -1), _F32)
    table.flags.writeable = False
    return table


@dataclass(frozen=True)
class _Target:
    width: int
    height: int
    color_fbo: int
    resolve_fbo: int
    segid_fbo: int


class SceneRenderer:
    """Draw ``mjvScene`` geoms as ``mjr_render`` does, from the scene's camera pose.

    ``context`` is the ``mujoco.Renderer`` whose EGL context we share; the caller
    keeps it current-thread exclusive, as for ``mjr_render``, and closes us first.
    """

    def __init__(self, model, width, height, context):
        self._model = model
        self._width, self._height = int(width), int(height)
        if self._width <= 0 or self._height <= 0:
            raise ValueError(f"image size {self._width}x{self._height}")
        self._context = context
        self._samples = int(context._mjr_context.offSamples)
        self._uniforms = {}
        self._targets = {}
        self._ray_textures = {}
        self._owned = []  # deleters of every GL object, run by close() or by a failed __init__
        with self._gl():
            try:
                self._programs = {}
                for name, defines in (("pinhole", ()), ("kb4", ("KB4",)), ("kb4_exact", ("KB4", "EXACT_DEPTH"))):
                    program = self._programs[name] = _program(*defines)
                    self._owned.append(lambda program=program: GL.glDeleteProgram(program))
                self._build_geometry(model)
                self._build_textures(model)
                self._target(self._width, self._height)
                self._ssbo = self._gen(GL.glGenBuffers, GL.glDeleteBuffers)
            except BaseException:
                self._delete_owned()
                raise

    def _gen(self, gen, delete, n=1):
        names = [int(name) for name in np.atleast_1d(gen(n))]
        self._owned.append(lambda: delete(n, names))
        return names[0] if n == 1 else names

    def _target(self, width, height):
        """Colour (MSAA), resolve and id framebuffers of one raster size, made on first use."""

        if (width, height) not in self._targets:
            size = (width, height)
            color_rb = self._renderbuffer(self._samples, GL.GL_RGBA8, *size)
            color_depth_rb = self._renderbuffer(self._samples, GL.GL_DEPTH32F_STENCIL8, *size)
            color_fbo = self._framebuffer(((GL.GL_COLOR_ATTACHMENT0, color_rb),
                                           (GL.GL_DEPTH_STENCIL_ATTACHMENT, color_depth_rb)))
            resolve_fbo = self._framebuffer(((GL.GL_COLOR_ATTACHMENT0, self._renderbuffer(0, GL.GL_RGBA8, *size)),))
            segid_rb = self._renderbuffer(0, GL.GL_R32I, *size)
            distance_rb = self._renderbuffer(0, GL.GL_R32F, *size)
            segid_depth_rb = self._renderbuffer(0, GL.GL_DEPTH32F_STENCIL8, *size)
            segid_fbo = self._framebuffer(((GL.GL_COLOR_ATTACHMENT1, segid_rb),
                                           (GL.GL_COLOR_ATTACHMENT2, distance_rb),
                                           (GL.GL_DEPTH_STENCIL_ATTACHMENT, segid_depth_rb)))
            self._targets[size] = _Target(width, height, color_fbo, resolve_fbo, segid_fbo)
        return self._targets[(width, height)]

    def _ray_texture(self, camera, theta_limit):
        key = (camera, theta_limit)
        if key not in self._ray_textures:
            table = _ray_table(camera, theta_limit)
            name = self._gen(GL.glGenTextures, GL.glDeleteTextures)
            GL.glActiveTexture(GL.GL_TEXTURE2)
            GL.glBindTexture(GL.GL_TEXTURE_2D, name)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, camera.width, camera.height, 0,
                            GL.GL_RGBA, GL.GL_FLOAT, np.ascontiguousarray(table))
            self._ray_textures[key] = name
        return self._ray_textures[key]

    def _renderbuffer(self, samples, fmt, width, height):
        rb = self._gen(GL.glGenRenderbuffers, GL.glDeleteRenderbuffers)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb)
        if samples:
            GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, samples, fmt, width, height)
        else:
            GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, fmt, width, height)
        return rb

    def _framebuffer(self, attachments):
        fbo = self._gen(GL.glGenFramebuffers, GL.glDeleteFramebuffers)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        for point, rb in attachments:
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, point, GL.GL_RENDERBUFFER, rb)
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"framebuffer incomplete: 0x{status:x}")
        return fbo

    def _delete_owned(self):
        while self._owned:
            self._owned.pop()()

    @contextlib.contextmanager
    def _gl(self):
        """Make the shared context current, then restore the state ``mjr_render`` assumes.

        ``mjr_render`` draws into the bound framebuffer, reads client arrays with
        buffer object 0 bound, and assumes ``glCullFace(GL_BACK)``.
        """

        self._context._gl_context.make_current()
        draw = GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING)
        read = GL.glGetIntegerv(GL.GL_READ_FRAMEBUFFER_BINDING)
        try:
            yield
        finally:
            GL.glUseProgram(0)
            GL.glBindVertexArray(0)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            GL.glActiveTexture(GL.GL_TEXTURE2)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glDisable(GL.GL_CLIP_DISTANCE0)  # the same switch as glClipPlane(GL_CLIP_PLANE0)
            GL.glCullFace(GL.GL_BACK)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, draw)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, read)

    def _build_geometry(self, model):
        q = model.vis.quality
        parts = {
            "sphere": _sphere(q.numslices, q.numstacks),
            "sphere_top": _half_sphere(1, q.numslices, q.numstacks // 2),
            "sphere_bottom": _half_sphere(-1, q.numslices, q.numstacks // 2),
            "cylinder": np.concatenate((_disk(-1, q.numslices, q.numstacks // 2),
                                        _cylinder(q.numslices, q.numstacks),
                                        _disk(1, q.numslices, q.numstacks // 2))),
            "cylinder_open": _cylinder(q.numslices, q.numstacks),
            "box": _box(q.numquads),
            "plane": _plane_default(q.numquads),
        }
        planes = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
        for plane_id, geom_id in enumerate(planes):
            parts[("plane", plane_id)] = _plane_grid(model, geom_id)
        self._ranges, first, chunks = {}, 0, []
        for key, rows in parts.items():
            self._ranges[key] = (first, len(rows))
            first += len(rows)
            chunks.append(rows.astype(_F32))
        data = np.ascontiguousarray(np.concatenate(chunks))
        self._vao, self._mesh_vao = self._gen(GL.glGenVertexArrays, GL.glDeleteVertexArrays, 2)
        GL.glBindVertexArray(self._vao)
        self._vbo = self._gen(GL.glGenBuffers, GL.glDeleteBuffers)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
        for location, (offset, count) in enumerate(((0, 3), (3, 3), (6, 2))):
            GL.glEnableVertexAttribArray(location)
            GL.glVertexAttribPointer(location, count, GL.GL_FLOAT, False, 32, ctypes.c_void_p(4 * offset))
        GL.glBindVertexArray(0)
        for mesh_id, (faceadr, facenum) in enumerate(zip(model.mesh_faceadr, model.mesh_facenum)):
            self._ranges[("mesh", mesh_id)] = (3 * int(faceadr), 3 * int(facenum))
        self._mesh_buffers = self._gen(GL.glGenBuffers, GL.glDeleteBuffers, len(_MESH_ARRAYS))
        for buffer, name in zip(self._mesh_buffers, _MESH_ARRAYS):
            array = getattr(model, name)
            data = np.ascontiguousarray(array) if array.size else np.zeros(4, array.dtype)  # cannot bind size 0
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, buffer)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)

    def _build_textures(self, model):
        """``mjr_uploadTexture`` with the same formats and filters."""

        self._textures = []
        self._skybox_texture = next(
            (t for t in range(model.ntex) if model.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX), -1)
        for tex_id in range(model.ntex):
            width, height = int(model.tex_width[tex_id]), int(model.tex_height[tex_id])
            nchannel = int(model.tex_nchannel[tex_id])
            data = model.tex_data[model.tex_adr[tex_id]:]
            name = self._gen(GL.glGenTextures, GL.glDeleteTextures)
            if model.tex_type[tex_id] == mujoco.mjtTexture.mjTEXTURE_2D:
                srgb = model.tex_colorspace[tex_id] == mujoco.mjtColorSpace.mjCOLORSPACE_SRGB
                fmt = {3: GL.GL_RGB, 4: GL.GL_RGBA}[nchannel]
                internal = {GL.GL_RGB: GL.GL_SRGB8, GL.GL_RGBA: GL.GL_SRGB8_ALPHA8}[fmt] if srgb else fmt
                GL.glBindTexture(GL.GL_TEXTURE_2D, name)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
                pixels = np.ascontiguousarray(data[: width * height * nchannel])
                GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal, width, height, 0, fmt, GL.GL_UNSIGNED_BYTE, pixels)
                GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
                self._textures.append((GL.GL_TEXTURE_2D, name))
            else:
                GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, name)
                for p in (GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T, GL.GL_TEXTURE_WRAP_R):
                    GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, p, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
                face = 3 * width * width
                for i in range(6):
                    start = 0 if width == height else i * face
                    pixels = np.ascontiguousarray(data[start:start + face])
                    GL.glTexImage2D(GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL.GL_RGB, width, width, 0,
                                    GL.GL_RGB, GL.GL_UNSIGNED_BYTE, pixels)
                GL.glGenerateMipmap(GL.GL_TEXTURE_CUBE_MAP)
                self._textures.append((GL.GL_TEXTURE_CUBE_MAP, name))

    def _uniform(self, program, name):
        if (program, name) not in self._uniforms:
            self._uniforms[program, name] = GL.glGetUniformLocation(program, name)
        return self._uniforms[program, name]

    def _camera(self, scene):
        cam = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
        if cam.orthographic:
            raise NotImplementedError("orthographic cameras")
        forward = np.asarray(cam.forward, _F32)
        f = forward / np.sqrt(np.dot(forward, forward))
        s = np.cross(f, np.asarray(cam.up, _F32))
        s /= np.sqrt(np.dot(s, s))
        u = np.cross(s, f)
        u /= np.sqrt(np.dot(u, u))
        pos = np.asarray(cam.pos, _F32)
        view = np.eye(4)
        view[:3, :3] = (s, u, -f)
        view[:3, 3] = (-np.dot(s, pos), -np.dot(u, pos), np.dot(f, pos))
        half_width = cam.frustum_width or (
            0.5 * self._width / self._height * (cam.frustum_top - cam.frustum_bottom))
        left, right = cam.frustum_center - half_width, cam.frustum_center + half_width
        bottom, top, near, far = cam.frustum_bottom, cam.frustum_top, cam.frustum_near, cam.frustum_far
        frustum = np.array([
            [2 * near / (right - left), 0, (right + left) / (right - left), 0],
            [0, 2 * near / (top - bottom), (top + bottom) / (top - bottom), 0],
            [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, -1, 0],
        ])
        # Reverse Z with glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE), as setView.
        projection = _translate(0, 0, 0.5) @ _scale(1, 1, -0.5) @ frustum
        return view, projection, cam, pos.astype(np.float64)

    def _parts(self, geom):
        """``(range key, object-to-world matrix)`` pairs, as ``renderGeom`` draws the geom."""

        size = np.asarray(geom.size, _F32).astype(np.float64)
        base = _translate(*geom.pos)
        base[:3, :3] = np.asarray(geom.mat, np.float64).reshape(3, 3)
        kind = geom.type
        if kind == mujoco.mjtGeom.mjGEOM_PLANE:
            if geom.dataid >= 0:
                return [(("plane", int(geom.dataid)), base)]
            return [("plane", base @ _scale(size[0], size[1], 1))]
        if kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            return [("sphere", base @ _scale(*size))]
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            av = float(_F32(0.5 * (size[0] + size[1])))
            body = base @ _scale(*size)
            top = body @ _scale(1, 1, av / size[2]) @ _translate(0, 0, (size[2] - av) / av)
            bottom = top @ _translate(0, 0, 2 * (av - size[2]) / av)
            return [("cylinder_open", body), ("sphere_top", top), ("sphere_bottom", bottom)]
        if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return [("cylinder", base @ _scale(*size))]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            return [("box", base @ _scale(*size))]
        if kind in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_SDF) and geom.dataid >= 0:
            if geom.dataid % 2:
                raise NotImplementedError("convex-hull mesh rendering (mjVIS_CONVEXHULL)")
            return [(("mesh", int(geom.dataid) // 2), base)]
        raise NotImplementedError(f"mjvGeom type {int(kind)}")

    def _texture(self, geom, record):
        """Fill texture mode and texgen planes; return the texture index or -1."""

        if geom.matid < 0:
            return -1
        model = self._model
        tex_id = int(model.mat_texid[geom.matid, mujoco.mjtTextureRole.mjTEXROLE_RGB])
        if tex_id < 0:
            return -1
        if geom.texcoord:
            if model.tex_type[tex_id] != mujoco.mjtTexture.mjTEXTURE_2D:
                raise NotImplementedError("mesh texcoords with a cube texture")
            record[9, 1] = _UV_2D
            return tex_id
        size = np.asarray(geom.size, _F32)
        uniform = bool(model.mat_texuniform[geom.matid])
        if model.tex_type[tex_id] == mujoco.mjtTexture.mjTEXTURE_2D:
            scl = np.asarray(model.mat_texrepeat[geom.matid], _F32)
            if geom.dataid >= 0:
                scl = np.where(size[:2] > 0, scl / np.maximum(_F32(mujoco.mjMINVAL), size[:2]), scl)
            if uniform:
                scl = np.where(size[:2] > 0, scl * size[:2], scl)
            record[9, 1] = _TEXGEN_2D
            record[10] = (0.5 * scl[0], 0, 0, -0.5)
            record[11] = (0, -0.5 * scl[1], 0, -0.5)
        else:
            scl = size if uniform else np.ones(3, _F32)
            record[9, 1] = _TEXGEN_CUBE
            record[10:13, :3] = np.diag(scl)
        return tex_id

    def render(self, scene, camera=None, theta_max=None, *, color=True) -> SceneImage:
        """Draw ``scene`` from its camera pose.

        Without ``camera``, the scene's pinhole frustum fills the ``width`` x ``height`` raster.
        A KB4 ``camera`` (``FisheyeIntrinsics``) draws its own raster instead, out to a little
        beyond ``theta_max`` [rad] off the axis; the caller masks the rays beyond ``theta_max``.
        With ``color=False`` only ids and distance are drawn, and ``rgb`` is ``None``.
        """

        if camera is not None and theta_max is None:
            raise ValueError("a KB4 camera needs theta_max")
        flags = scene.flags
        unsupported = [name for name, on in (
            ("segment", flags[mujoco.mjtRndFlag.mjRND_SEGMENT]),
            ("wireframe", flags[mujoco.mjtRndFlag.mjRND_WIREFRAME]),
            ("depth", flags[mujoco.mjtRndFlag.mjRND_DEPTH]),
            ("stereo", scene.stereo),
            ("enabletransform", scene.enabletransform),
            ("framewidth", scene.framewidth),
            ("skin", scene.nskin),
            ("flex", scene.nflex),
        ) if on]
        if unsupported:
            raise NotImplementedError(f"mjvScene options {unsupported}")
        view, projection, cam, headpos = self._camera(scene)
        geoms = [scene.geoms[i] for i in range(scene.ngeom)]
        records, draws, info = [], [], []
        reflective_seen = False
        for index, geom in enumerate(geoms):
            if geom.label:
                raise NotImplementedError("geom labels")
            behind = geom.type == mujoco.mjtGeom.mjGEOM_PLANE and float(
                np.dot(headpos - np.asarray(geom.pos), np.asarray(geom.mat).reshape(3, 3)[:, 2])) < 0
            rgba = np.asarray(geom.rgba, _F32).copy()
            transparent = rgba[3] < 0.995 or behind
            reflective = (geom.type in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_BOX)
                          and not transparent and geom.reflectance > 0 and not reflective_seen)
            reflective_seen |= reflective  # mjr_render keeps only the first reflective geom
            if reflective and flags[mujoco.mjtRndFlag.mjRND_REFLECTION]:
                raise NotImplementedError("reflective geoms with mjRND_REFLECTION")
            if behind:
                rgba[3] *= _F32(0.3)
            record = np.zeros((_RECORD_VEC4, 4), _F32)
            record[7] = rgba
            record[8, :3] = _F32(geom.emission) * rgba[:3]
            record[8, 3] = geom.specular
            record[9, 0] = _F32(geom.shininess) * _F32(128.0)
            record[9, 2] = geom.segid + 1
            texture = self._texture(geom, record)
            camdist = np.inf
            if transparent:
                camdist = 1e10 if geom.type == mujoco.mjtGeom.mjGEOM_PLANE else float(
                    np.linalg.norm(np.asarray(geom.pos) - headpos) - _rbound(geom))
            info.append((transparent, reflective, camdist))
            for key, world in self._parts(geom):
                model_view = (view @ world).astype(_F32)
                record = record.copy()
                record[0:4] = model_view.T
                record[4:7, :3] = np.linalg.inv(model_view[:3, :3].astype(np.float64)).astype(_F32)
                record[4, 3] = behind
                mesh = key[0] == "mesh"
                if mesh:
                    m = self._model
                    record.view(np.int32)[13] = (m.mesh_vertadr[key[1]], m.mesh_normaladr[key[1]],
                                                 m.mesh_texcoordadr[key[1]], 1)
                records.append(record)
                draws.append((index, *self._ranges[key], texture, behind, mesh))
        skybox = None
        if flags[mujoco.mjtRndFlag.mjRND_SKYBOX] and self._skybox_texture >= 0:
            if flags[mujoco.mjtRndFlag.mjRND_HAZE] and any(
                    g.type == mujoco.mjtGeom.mjGEOM_PLANE and g.size[0] == 0 and g.size[1] == 0 for g in geoms):
                raise NotImplementedError("haze over an infinite plane")
            # mjr_render's skybox: the unit cylinder at the head, scaled below far / sqrt(2) to escape far clipping.
            radius = float(_F32(cam.frustum_far) * _F32(0.7))
            record = np.zeros((_RECORD_VEC4, 4), _F32)
            record[0:4] = (view @ _translate(*headpos) @ _scale(radius, radius, radius)).astype(_F32).T
            record[4, 3] = 1
            record[7] = 1
            record[9] = (0, _TEXGEN_CUBE, 0, 1)
            record[10:13] = ((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0))
            skybox = len(records)
            records.append(record)
            draws.append((-1, *self._ranges["cylinder"], self._skybox_texture, True, False))
        data = np.ascontiguousarray(np.stack(records)) if records else np.zeros((1, _RECORD_VEC4, 4), _F32)
        by_geom = {}
        for record_index, draw in enumerate(draws):
            by_geom.setdefault(draw[0], []).append(record_index)
        with self._gl():
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._ssbo)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL.GL_STREAM_DRAW)
            for binding, buffer in enumerate((self._ssbo, *self._mesh_buffers)):
                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, binding, buffer)
            if camera is None:
                target = self._target(self._width, self._height)
                programs = (self._programs["pinhole"],) * 2
                self._primitive = GL.GL_TRIANGLES
            else:
                theta_cut = float(theta_max) + _THETA_MARGIN
                rays = self._ray_texture(camera, theta_cut)
                target = self._target(camera.width, camera.height)
                programs = (self._programs["kb4"], self._programs["kb4_exact"])
                self._primitive = GL.GL_PATCHES
                GL.glActiveTexture(GL.GL_TEXTURE2)
                GL.glBindTexture(GL.GL_TEXTURE_2D, rays)
                GL.glEnable(GL.GL_CLIP_DISTANCE0)
                GL.glPatchParameteri(GL.GL_PATCH_VERTICES, 3)
            for program in dict.fromkeys(programs):
                GL.glUseProgram(program)
                self._set_uniforms(program, scene, view, flags)
                if camera is None:
                    GL.glUniformMatrix4fv(self._uniform(program, "u_projection"), 1, True, projection.astype(_F32))
                else:
                    self._set_kb4_uniforms(program, camera, theta_cut, cam, flags)
            self._common_state(flags, target)
            if color:
                GL.glUseProgram(programs[0])
                self._color_pass(target, info, by_geom, draws, flags, skybox)
            GL.glUseProgram(programs[1])
            self._segment_pass(target, geoms, by_geom, draws)
            return self._read(target, color)

    def _set_uniforms(self, program, scene, view, flags):
        GL.glUniform1i(self._uniform(program, "u_texture_2d"), 0)
        GL.glUniform1i(self._uniform(program, "u_texture_cube"), 1)
        self._set_lights(program, scene, view)
        model = self._model
        extent = model.stat.extent
        GL.glUniform4f(self._uniform(program, "u_fog"), extent * model.vis.map.fogstart,
                       extent * model.vis.map.fogend, float(flags[mujoco.mjtRndFlag.mjRND_FOG]), 0)
        GL.glUniform3f(self._uniform(program, "u_fog_color"), *model.vis.rgba.fog[:3])

    def _set_kb4_uniforms(self, program, camera, theta_cut, cam, flags):
        GL.glUniform1i(self._uniform(program, "u_rays"), 2)
        GL.glUniform4f(self._uniform(program, "u_focal"), camera.fx, camera.fy, camera.cx, camera.cy)
        GL.glUniform4f(self._uniform(program, "u_distortion"), camera.k1, camera.k2, camera.k3, camera.k4)
        GL.glUniform2f(self._uniform(program, "u_raster"), camera.width, camera.height)
        GL.glUniform1f(self._uniform(program, "u_theta_cut"), theta_cut)
        GL.glUniform2f(self._uniform(program, "u_depth_range"), cam.frustum_near, cam.frustum_far)
        GL.glUniform1f(self._uniform(program, "u_step"), _MAX_STEP)
        GL.glUniform1i(self._uniform(program, "u_cull"), int(bool(flags[mujoco.mjtRndFlag.mjRND_CULL_FACE])))

    def _set_lights(self, program, scene, view):
        nlight = int(scene.nlight)
        if nlight > _MAX_LIGHTS:
            raise NotImplementedError(f"{nlight} lights; OpenGL fixed function has {_MAX_LIGHTS}")
        rows = {k: np.zeros((_MAX_LIGHTS, 4), _F32) for k in ("position", "spot", "attenuation")}
        colors = {k: np.zeros((_MAX_LIGHTS, 3), _F32) for k in ("ambient", "diffuse", "specular")}
        rotation = view[:3, :3]
        shadows = scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] and self._model.vis.quality.shadowsize
        for i in range(nlight):
            light = scene.lights[i]
            if light.castshadow and shadows:
                raise NotImplementedError("shadow-casting lights")
            if light.type == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL:
                rows["position"][i, :3] = rotation @ -np.asarray(light.dir, _F32)
                rows["spot"][i] = (0, 0, -1, -2)
                rows["attenuation"][i] = (1, 0, 0, 0)
            elif light.type == mujoco.mjtLightType.mjLIGHT_SPOT:
                rows["position"][i] = (view @ np.append(np.asarray(light.pos, _F32), 1.0)).astype(_F32)
                rows["spot"][i, :3] = rotation @ np.asarray(light.dir, _F32)
                rows["spot"][i, :3] /= np.linalg.norm(rows["spot"][i, :3])
                if not (0 <= light.cutoff <= 90 or light.cutoff == 180) or not 0 <= light.exponent <= 128:
                    raise NotImplementedError("spot cutoff or exponent outside the OpenGL range")
                rows["spot"][i, 3] = -2 if light.cutoff == 180 else np.cos(np.radians(light.cutoff))
                rows["attenuation"][i] = (*light.attenuation, 0 if light.cutoff == 180 else light.exponent)
            else:
                raise NotImplementedError(f"light type {int(light.type)}")
            for name in colors:
                colors[name][i] = getattr(light, name)
        GL.glUniform1i(self._uniform(program, "u_nlight"), nlight)
        GL.glUniform3f(self._uniform(program, "u_ambient"), *([0.0] * 3 if scene.nlight else [0.3] * 3))
        for name, value in rows.items():
            GL.glUniform4fv(self._uniform(program, f"u_light_{name}"), _MAX_LIGHTS, value)
        for name, value in colors.items():
            GL.glUniform3fv(self._uniform(program, f"u_light_{name}"), _MAX_LIGHTS, value)

    def _common_state(self, flags, target):
        GL.glViewport(0, 0, target.width, target.height)
        GL.glClipControl(GL.GL_LOWER_LEFT, GL.GL_ZERO_TO_ONE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_GEQUAL)
        GL.glDepthRange(0, 1)
        GL.glClearDepth(0)
        GL.glClearColor(0, 0, 0, 0)
        GL.glFrontFace(GL.GL_CCW)
        GL.glCullFace(GL.GL_BACK)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        for cap in (GL.GL_SCISSOR_TEST, GL.GL_STENCIL_TEST, GL.GL_ALPHA_TEST, GL.GL_POLYGON_OFFSET_FILL):
            GL.glDisable(cap)
        self._cull = bool(flags[mujoco.mjtRndFlag.mjRND_CULL_FACE])

    def _draw(self, record_index, draw, bound):
        _, first, count, texture, behind, mesh = draw
        vao = self._mesh_vao if mesh else self._vao
        if bound.get("vao") != vao:
            GL.glBindVertexArray(vao)
            bound["vao"] = vao
        if texture >= 0 and bound.get("texture") != texture:
            target, name = self._textures[texture]
            GL.glActiveTexture(GL.GL_TEXTURE0 if target == GL.GL_TEXTURE_2D else GL.GL_TEXTURE1)
            GL.glBindTexture(target, name)
            bound["texture"] = texture
        if self._cull:
            face = GL.GL_FRONT if behind else GL.GL_BACK
            if bound.get("cull") != face:
                GL.glCullFace(face)
                bound["cull"] = face
        GL.glDrawArraysInstancedBaseInstance(self._primitive, first, count, 1, record_index)

    def _draw_geoms(self, order, by_geom, draws, bound):
        for index in order:
            for record_index in by_geom.get(index, ()):
                self._draw(record_index, draws[record_index], bound)

    def _color_pass(self, target, info, by_geom, draws, flags, skybox):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target.color_fbo)
        GL.glDrawBuffers(1, [GL.GL_COLOR_ATTACHMENT0])
        GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        (GL.glEnable if self._cull else GL.glDisable)(GL.GL_CULL_FACE)
        bound = {}
        reflective = [i for i, (t, r, _) in enumerate(info) if r]
        opaque = [i for i, (t, r, _) in enumerate(info) if not t and not r]
        transparent = sorted((i for i, (t, _, _) in enumerate(info) if t), key=lambda i: info[i][2])
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)
        self._draw_geoms(reflective, by_geom, draws, bound)
        GL.glDisable(GL.GL_BLEND)
        self._draw_geoms(opaque, by_geom, draws, bound)
        if skybox is not None:
            self._draw(skybox, draws[skybox], bound)
        if transparent:
            GL.glDepthMask(GL.GL_FALSE)
            GL.glEnable(GL.GL_BLEND)
            additive = flags[mujoco.mjtRndFlag.mjRND_ADDITIVE]
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE if additive else GL.GL_ONE_MINUS_SRC_ALPHA)
            self._draw_geoms(transparent, by_geom, draws, bound)
            if not additive:
                self._draw_geoms(transparent[::-1], by_geom, draws, bound)
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDisable(GL.GL_BLEND)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, target.color_fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, target.resolve_fbo)
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
        w, h = target.width, target.height
        GL.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)

    def _segment_pass(self, target, geoms, by_geom, draws):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target.segid_fbo)
        GL.glDrawBuffers(3, [GL.GL_NONE, GL.GL_COLOR_ATTACHMENT1, GL.GL_COLOR_ATTACHMENT2])
        GL.glDisable(GL.GL_MULTISAMPLE)
        GL.glDisable(GL.GL_BLEND)
        GL.glClearBufferiv(GL.GL_COLOR, 1, np.zeros(4, np.int32))
        GL.glClearBufferfv(GL.GL_COLOR, 2, np.full(4, np.inf, _F32))
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        order = [i for i, geom in enumerate(geoms) if geom.segid >= 0]
        self._draw_geoms(order, by_geom, draws, {})
        GL.glEnable(GL.GL_MULTISAMPLE)

    def _read(self, target, color):
        w, h = target.width, target.height
        rgb = None
        if color:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, target.resolve_fbo)
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
            rgb = np.frombuffer(GL.glReadPixels(0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE), np.uint8)
            rgb = rgb.reshape(h, w, 3)[::-1].copy()
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, target.segid_fbo)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
        segid = np.frombuffer(GL.glReadPixels(0, 0, w, h, GL.GL_RED_INTEGER, GL.GL_INT), np.int32).reshape(h, w)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT2)
        distance = np.frombuffer(GL.glReadPixels(0, 0, w, h, GL.GL_RED, GL.GL_FLOAT), _F32).reshape(h, w)
        return SceneImage(rgb=rgb, segid=segid[::-1] - 1, distance=distance[::-1].copy())

    def close(self):
        if self._owned:
            with self._gl():
                self._delete_owned()


@dataclass(frozen=True)
class WristSourceImage:
    """MuJoCo-only wrist image on the pre-crop raster of a wrist profile; rows start at the top."""

    rgb: np.ndarray  # [H, W, 3] uint8, area mean of the supersampled colour; 0 outside the circle
    segid: np.ndarray  # [H, W] int32 at pixel centres, index into scene.geoms; -1 where no geom or outside
    distance: np.ndarray  # [H, W] float32 ray distance [m] at pixel centres; inf where no geom, nan outside
    supersampled: SceneImage


class WristSourceRenderer:
    """One KB4 MuJoCo pass per wrist image, on the raster of ``wrist_source_intrinsics(profile)``.

    ``scene`` comes from ``update_scene`` with the wrist's front camera and the profile's
    edits, as in ``WristFisheye.capture_mujoco_stage``. Colour is drawn ``supersample`` times
    finer and area-averaged; ids and distance are drawn at the pixel centres.
    """

    def __init__(self, model, context, profile=DEFAULT_POLICY_WRIST_RENDER_PROFILE, supersample=1):
        self.camera = wrist_source_intrinsics(profile)
        self.theta_max = float(np.radians(get_wrist_render_profile(profile).source_fov_deg / 2.0))
        self.valid = camera_rays(self.camera, self.theta_max)[1]
        self.supersample = float(supersample)
        size = [max(1, round(self.supersample * n)) for n in (self.camera.width, self.camera.height)]
        self.supersampled_camera = self.camera if self.supersample == 1 else self.camera.resized(*size)
        self._renderer = SceneRenderer(model, *size, context)

    def render(self, scene) -> WristSourceImage:
        fine = self._renderer.render(scene, self.supersampled_camera, self.theta_max)
        exact, rgb = fine, fine.rgb
        if self.supersampled_camera is not self.camera:
            exact = self._renderer.render(scene, self.camera, self.theta_max, color=False)
            rgb = cv.resize(fine.rgb, (self.camera.width, self.camera.height), interpolation=cv.INTER_AREA)
        valid = self.valid
        return WristSourceImage(
            rgb=np.where(valid[..., None], rgb, 0).astype(np.uint8),
            segid=np.where(valid, exact.segid, -1).astype(np.int32),
            distance=np.where(valid, exact.distance, np.nan).astype(_F32),
            supersampled=fine,
        )

    def close(self):
        self._renderer.close()
