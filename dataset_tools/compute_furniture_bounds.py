import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

GAZEBO_MODELS = os.path.expanduser("~/gazebo_models")

# model_dir -> (unit_scale for any mesh files found, default 1.0)
TARGETS = {
    "bookshelf": 1.0,
    "cabinet": 1.0,
    "table": 1.0,
    "cafe_table": 1.0,
    "washer": 1.0,
    "person_standing": 1.0,
    "person_walking": 1.0,
}


def mesh_bounds(path):
    scene_or_mesh = trimesh.load(path, force='scene')
    mesh = scene_or_mesh.to_geometry() if isinstance(scene_or_mesh, trimesh.Scene) else scene_or_mesh
    return mesh.bounds  # (2,3) min/max in mesh local frame


def get_unit_scale(dae_path):
    try:
        content = open(dae_path).read()
        m = re.search(r'<unit[^>]*meter="([\d.]+)"', content)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return 1.0


def parse_pose(elem):
    if elem is None:
        return [0.0] * 6
    return [float(v) for v in elem.text.split()]


def combined_bounds_from_sdf(sdf_path, model_dir):
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    mins = []
    maxs = []

    for geom_parent_tag in ("collision", "visual"):
        for elem in root.iter(geom_parent_tag):
            pose = parse_pose(elem.find("pose"))
            px, py, pz = pose[0], pose[1], pose[2]
            geom = elem.find("geometry")
            if geom is None:
                continue

            box = geom.find("box")
            cyl = geom.find("cylinder")
            sph = geom.find("sphere")
            mesh = geom.find("mesh")

            if box is not None:
                sx, sy, sz = [float(v) for v in box.find("size").text.split()]
                mins.append([px - sx/2, py - sy/2, pz - sz/2])
                maxs.append([px + sx/2, py + sy/2, pz + sz/2])
            elif cyl is not None:
                r = float(cyl.find("radius").text)
                length = float(cyl.find("length").text)
                mins.append([px - r, py - r, pz - length/2])
                maxs.append([px + r, py + r, pz + length/2])
            elif sph is not None:
                r = float(sph.find("radius").text)
                mins.append([px - r, py - r, pz - r])
                maxs.append([px + r, py + r, pz + r])
            elif mesh is not None:
                uri = mesh.find("uri").text
                rel = uri.replace("model://", "")
                # rel = "<model_name>/meshes/xxx.dae" - resolve relative to gazebo_models root
                mesh_path = os.path.join(GAZEBO_MODELS, rel)
                if not os.path.exists(mesh_path):
                    continue
                scale_elem = mesh.find("scale")
                scale = [1.0, 1.0, 1.0]
                if scale_elem is not None:
                    scale = [float(v) for v in scale_elem.text.split()]
                unit = get_unit_scale(mesh_path) if mesh_path.endswith(".dae") else 1.0
                b = mesh_bounds(mesh_path)
                b = b * unit
                b[:, 0] *= scale[0]
                b[:, 1] *= scale[1]
                b[:, 2] *= scale[2]
                mins.append([px + b[0][0], py + b[0][1], pz + b[0][2]])
                maxs.append([px + b[1][0], py + b[1][1], pz + b[1][2]])

    if not mins:
        return None

    mins = np.array(mins)
    maxs = np.array(maxs)
    overall_min = mins.min(axis=0)
    overall_max = maxs.max(axis=0)
    return overall_min, overall_max


results = {}

for model_dir in TARGETS:
    sdf_path = os.path.join(GAZEBO_MODELS, model_dir, "model.sdf")
    if not os.path.exists(sdf_path):
        candidates = [f for f in os.listdir(os.path.join(GAZEBO_MODELS, model_dir)) if f.endswith(".sdf")]
        sdf_path = os.path.join(GAZEBO_MODELS, model_dir, candidates[0])

    bounds = combined_bounds_from_sdf(sdf_path, model_dir)
    if bounds is None:
        print(f"{model_dir}: NO GEOMETRY FOUND")
        continue

    bmin, bmax = bounds
    size = bmax - bmin
    center = (bmin + bmax) / 2.0
    half = size / 2.0

    print(f"{model_dir:18s} size={size.round(3).tolist()} center_offset={center.round(3).tolist()}")
    results[model_dir] = {
        "half": half.round(4).tolist(),
        "offset": center.round(4).tolist(),
    }

import json
print()
print(json.dumps(results, indent=2))
