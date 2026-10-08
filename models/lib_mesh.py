"""Shared helpers: load a case's SDF volumes, extract smoothed meshes and
render them with consistent flat-gray studio lighting. Used by both
build_cases.py (interactive 3D viewer data) and render_marketing_images.py
(static comparison PNGs), so the two outputs look like the same material.
"""
import numpy as np
from skimage import measure
import trimesh

GRID = 64  # volumes are 64^3 SDF grids; voxel index maps linearly to [-1, 1]
BASE_COLOR = np.array([0.827, 0.847, 0.875])  # #D3D8DF, matches the three.js viewer material


def load_volume(path):
    return np.load(path).squeeze().astype(np.float32)


def mesh_from_volume(vol, smooth_iters=2, step_size=1):
    """Marching cubes at the zero level set, then just enough Laplacian
    smoothing to take the edge off the 64^3 voxel facets without erasing
    fissures and cusps (that erasure is what made earlier renders look
    overly smooth — the network's own blurriness should be the only
    source of lost detail, not our post-processing)."""
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.0, step_size=step_size)
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    trimesh.smoothing.filter_laplacian(m, lamb=0.5, iterations=smooth_iters)
    return np.asarray(m.vertices, dtype=np.float32), np.asarray(m.faces, dtype=np.uint32)


def normalize_to_unit_cube(verts):
    """Grid index -> [-1, 1], identical for every mesh/case so the three
    panels of a case stay registered with each other."""
    return (verts / (GRID - 1)) * 2 - 1


def _cam_dir(elev, azim):
    e, a = np.radians(elev), np.radians(azim)
    v = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return v / np.linalg.norm(v)


def face_shade_colors(verts, faces, elev, azim):
    """Two-light shading (a camera-relative key light plus a soft fill) with
    a narrow specular highlight layered on top. The diffuse term alone reads
    as flat/plasticky and hides fissure detail; the highlight gives concave
    and convex surface changes a visible glint, like a glossy plaster cast,
    so cusps and grooves stay legible even where the diffuse shading is flat."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    key = _cam_dir(elev + 18, azim - 20)
    fill = _cam_dir(elev - 35, azim + 150)
    view = _cam_dir(elev, azim)
    diff_key = np.clip(n @ key, 0, 1)
    diff_fill = np.clip(n @ fill, 0, 1)
    shade = 0.32 + 0.78 * diff_key + 0.10 * diff_fill
    shade = np.clip(shade, 0.22, 1.25)
    color = np.clip(shade[:, None] * BASE_COLOR[None, :], 0, 1)
    half = key + view
    half = half / np.linalg.norm(half)
    spec = np.clip(n @ half, 0, 1) ** 30
    return np.clip(color + spec[:, None] * 0.3, 0, 1)
